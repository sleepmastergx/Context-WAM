"""Batched paradigm-C action inference for the legacy baseline ``FastWAM`` class.

Author: Rui Heng Yang

Merged from the dispatcher fleet's ``dispatcher/policy/models/fastwam_batching.py``
(``infer_baseline_action_batch``, verified on the released checkpoints: batch-one
bitwise-equal to the legacy single path, slot-position invariant, 2.04x throughput
at batch 2) so the batched forward lives in this repository instead of the serving
layer. The upstream ``FastWAM.infer_action`` entrypoint is restricted to one
sample; most of the class is already batch-general. This module follows the same
single-prefill KV-cache route as ``infer_action`` and adds per-slot seeds.

Seed semantics match ``FastWAMDecoupled.infer_action_batch``: ``seeds`` draws each
row from its OWN generator (a row's noise depends only on its own seed, never on
batch composition or arrival order — required by serving layers that derive one
seed per request); scalar ``seed`` keeps the original broadcast behaviour.

bf16 caveat, unchanged from the fleet measurements: batched matmuls are not
bitwise equal to batch-1 (~7.8e-03 max action delta on GB10), so batched runs are
statistically equivalent to unbatched ones, not bit-reproducible against them.
"""

from __future__ import annotations

import time
from typing import Optional

import torch


def baseline_infer_action_batch(
    model,
    *,
    input_image: torch.Tensor,
    action_horizon: int,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    proprio: Optional[torch.Tensor] = None,
    num_inference_steps: int = 20,
    sigma_shift: Optional[float] = None,
    seed: Optional[int] = None,
    seeds: Optional[list[Optional[int]]] = None,
    rand_device: str = "cpu",
) -> dict:
    """Return ``{"action": [B, T, D] float32 CPU, "timing_ms": {...}}`` in one forward pass.

    Limited to the baseline (paradigm C) KV-cache API used by the released
    checkpoints; the caller is responsible for guarding against paradigm A/B
    variants. Dynamic-batcher OOM recovery remains the serving layer's concern.
    """
    if input_image.ndim != 4 or input_image.shape[1] != 3:
        raise ValueError(f"input_image must be [B,3,H,W], got {tuple(input_image.shape)}")
    batch_size, _, height, width = input_image.shape
    if batch_size < 1:
        raise ValueError("Fast-WAM batching requires at least one sample")
    if height % 16 or width % 16:
        raise ValueError(f"Fast-WAM image dimensions must be multiples of 16, got {height}x{width}")
    if context.ndim != 3 or context_mask.ndim != 2:
        raise ValueError(
            "context/context_mask must be [B,L,D]/[B,L], got "
            f"{tuple(context.shape)}/{tuple(context_mask.shape)}"
        )
    if context.shape[0] != batch_size or context_mask.shape[0] != batch_size:
        raise ValueError("text context batch does not match image batch")
    if proprio is not None and (proprio.ndim != 2 or proprio.shape[0] != batch_size):
        raise ValueError(f"proprio must be [B,D] with B={batch_size}, got {tuple(proprio.shape)}")

    mot = model.mot
    if not hasattr(mot, "prefill_video_cache") or not hasattr(
        mot, "forward_action_with_video_cache"
    ):
        raise TypeError("this Fast-WAM variant does not expose the baseline batched KV-cache API")

    # Noise draw. ``seeds`` (per-slot, precedence) draws each row from its own generator,
    # exactly matching B independent infer_action calls; scalar ``seed`` broadcasts one draw.
    noise_shape = (1, action_horizon, model.action_expert.action_dim)
    if seeds is not None:
        if len(seeds) != batch_size:
            raise ValueError(f"got {len(seeds)} seeds for batch size {batch_size}")
        rows = []
        for row_seed in seeds:
            generator = (
                None
                if row_seed is None
                else torch.Generator(device=rand_device).manual_seed(row_seed)
            )
            rows.append(
                torch.randn(
                    noise_shape, generator=generator, device=rand_device, dtype=torch.float32
                )
            )
        noise = torch.cat(rows, dim=0)
    else:
        generator = (
            None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        )
        noise = torch.randn(
            noise_shape, generator=generator, device=rand_device, dtype=torch.float32
        ).expand(batch_size, -1, -1)
    latents_action = noise.contiguous().to(device=model.device, dtype=model.torch_dtype)

    def _sync_ms(t0: float) -> float:
        if latents_action.is_cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    t_prefill = time.perf_counter()
    input_image = input_image.to(device=model.device, dtype=model.torch_dtype)
    # WanVideoVAE38.encode(list) serializes the list; its underlying single_encode is
    # batch-general and accepts [B, C, T, H, W].
    first_frame_latents = model.vae.single_encode(input_image.unsqueeze(2), model.device)

    context = context.to(device=model.device, dtype=model.torch_dtype, non_blocking=True)
    context_mask = context_mask.to(device=model.device, dtype=torch.bool, non_blocking=True)
    if proprio is not None:
        proprio = proprio.to(device=model.device, dtype=model.torch_dtype)
        context, context_mask = model._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )

    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    timestep_video = torch.zeros(
        (batch_size,), dtype=first_frame_latents.dtype, device=model.device
    )
    video_pre = model.video_expert.pre_dit(
        x=first_frame_latents,
        timestep=timestep_video,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=fuse_flag,
    )
    video_seq_len = int(video_pre["tokens"].shape[1])
    attention_mask = model._build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=action_horizon,
        video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        device=video_pre["tokens"].device,
    )
    video_kv_cache = mot.prefill_video_cache(
        video_tokens=video_pre["tokens"],
        video_freqs=video_pre["freqs"],
        video_t_mod=video_pre["t_mod"],
        video_context_payload={
            "context": video_pre["context"],
            "mask": video_pre["context_mask"],
        },
        video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
    )
    video_prefill_ms = _sync_ms(t_prefill)

    t_denoise = time.perf_counter()
    timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=model.device,
        dtype=latents_action.dtype,
        shift_override=sigma_shift,
    )
    for step_t, step_delta in zip(timesteps, deltas, strict=True):
        timestep_action = (
            step_t.unsqueeze(0).to(dtype=latents_action.dtype, device=model.device)
        ).expand(batch_size)
        pred_action = model._predict_action_noise_with_cache(
            latents_action=latents_action,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        latents_action = model.infer_action_scheduler.step(
            pred_action, step_delta, latents_action
        )
    denoise_ms = _sync_ms(t_denoise)

    return {
        "action": latents_action.detach().to(device="cpu", dtype=torch.float32),
        "timing_ms": {
            "video_prefill_ms": video_prefill_ms,
            "denoise_ms": denoise_ms,
        },
    }
