# Author: Rui Heng Yang

"""FastWAMDecoupled -- Decoupled MoT variant with asymmetric expert layers.

Author: Rui Heng Yang

This module implements a FastWAM subclass where the video expert and action
expert have different numbers of transformer layers (e.g., video=30, action=5).
The action expert cross-attends to a learned mix of all video-layer K/V
tensors. ``kv_source_mapping`` is still used for action-layer initialization
and inspection. Supports kv_source_mode: final_only, uniform_end,
uniform_middle, and fused_kv. Uses paradigm C (first-frame video tokens
visible to action).
The action expert cross-attends to either video K/V selected by
``kv_source_mapping`` or fused K/V from ``kv_source_mode="fused_mlp"``.
Supports kv_source_mode: final_only, uniform_end, uniform_middle, and fused_mlp.
Uses paradigm C (first-frame video tokens visible to action).

Training uses ``MoTDecoupled.forward_decoupled()``.
Inference uses ``MoTDecoupled.prefill_video_kv()`` +
``MoTDecoupled.forward_action_with_video_kv()`` for the action path, and
standalone video expert forward passes for the video path.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM

logger = get_logger(__name__)


class FastWAMDecoupled(FastWAM):
    """FastWAM variant with decoupled (asymmetric-layer) video and action experts.

    The video expert runs all its layers independently. The action expert
    (with potentially fewer layers) cross-attends to either selected video K/V
    from ``kv_source_mapping`` or fused video K/V from
    ``kv_source_mode="fused_mlp"``. Supports kv_source_mode: final_only,
    uniform_end, uniform_middle, and fused_mlp. Paradigm C: action sees
    first-frame video tokens only.

    Overrides:
        - ``_build_mot_attention_mask`` -- returns a tuple of two masks
        - ``training_loss`` -- uses ``mot.forward_decoupled()``
        - ``infer_action`` -- uses ``mot.prefill_video_kv()`` +
          ``mot.forward_action_with_video_kv()``
        - ``infer`` -- runs full video denoising + action denoising independently
        - ``infer_joint`` -- raises NotImplementedError
    """

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build separate attention masks for the decoupled two-phase forward.

        Returns a tuple of two masks instead of a single joint mask:
        - ``video_mask``: ``[Sv, Sv]`` -- video self-attention (same as baseline).
        - ``action_mask``: ``[Sa, Sv + Sa]`` -- action rows only. First-frame
          video columns are True, remaining video columns False, all action
          columns True (paradigm C topology).

        Args:
            video_seq_len: Number of video tokens ``Sv``.
            action_seq_len: Number of action tokens ``Sa``.
            video_tokens_per_frame: Number of video tokens per frame. Used to
                determine which columns correspond to the first frame.
            device: Device for the mask tensors.

        Returns:
            Tuple of ``(video_mask, action_mask)``.
        """
        # Video self-attention mask: [Sv, Sv]
        video_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

        # Action attention mask: [Sa, Sv + Sa]
        # Columns: [video_first_frame | video_other_frames | action]
        action_mask = torch.zeros(
            (action_seq_len, video_seq_len + action_seq_len),
            dtype=torch.bool,
            device=device,
        )
        # Action can see first-frame video tokens.
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        action_mask[:, :first_frame_tokens] = True
        # Action can see all action tokens (full self-attention).
        action_mask[:, video_seq_len:] = True

        return video_mask, action_mask

    def training_loss(self, sample: dict, tiled: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute training loss using the decoupled two-phase forward pass.

        Same loss structure as the baseline ``FastWAM.training_loss()`` but uses
        ``self.mot.forward_decoupled()`` instead of ``self.mot()`` and passes
        separate video/action masks.

        Args:
            sample: Training sample dict from the dataloader.
            tiled: Whether to use tiled VAE encoding.

        Returns:
            Tuple of ``(loss_total, loss_dict)`` where ``loss_dict`` has keys
            ``"loss_video"`` and ``"loss_action"``.
        """
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        freeze_video_backbone = bool(
            getattr(self, "freeze_video_backbone_for_training", False)
        )

        # Sample action noise and timestep (independent from video).
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        if freeze_video_backbone:
            first_frame_latents = inputs["first_frame_latents"]
            if first_frame_latents is None:
                raise ValueError(
                    "`freeze_video_backbone_for_training` requires first-frame "
                    "conditioning latents."
                )

            timestep_video = torch.zeros(
                (batch_size,),
                device=self.device,
                dtype=first_frame_latents.dtype,
            )
            with torch.no_grad():
                video_pre = self.video_expert.pre_dit(
                    x=first_frame_latents,
                    timestep=timestep_video,
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                )
                video_mask = self.video_expert.build_video_to_video_mask(
                    video_seq_len=video_pre["tokens"].shape[1],
                    video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                    device=video_pre["tokens"].device,
                )
                video_kv_per_layer = self.mot.prefill_video_kv(
                    video_tokens=video_pre["tokens"],
                    video_freqs=video_pre["freqs"],
                    video_t_mod=video_pre["t_mod"],
                    video_context_payload={
                        "context": video_pre["context"],
                        "mask": video_pre["context_mask"],
                    },
                    video_attention_mask=video_mask,
                )

            action_pre = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )
            _, action_mask = self._build_mot_attention_mask(
                video_seq_len=video_pre["tokens"].shape[1],
                action_seq_len=action_pre["tokens"].shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=action_pre["tokens"].device,
            )
            action_tokens_out = self.mot.forward_action_with_video_kv(
                video_kv_per_layer=video_kv_per_layer,
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                } if action_pre["context"] is not None else None,
                action_attention_mask=action_mask,
            )
            pred_action = self.action_expert.post_dit(action_tokens_out, action_pre)

            action_loss_token = F.mse_loss(
                pred_action.float(), target_action.float(), reduction="none"
            ).mean(dim=2)
            if action_is_pad is not None:
                valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
                valid_sum = valid.sum(dim=1).clamp(min=1.0)
                action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
            else:
                action_loss_per_sample = action_loss_token.mean(dim=1)

            action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
                action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
            )
            loss_action = (action_loss_per_sample * action_weight).mean()
            loss_total = self.loss_lambda_action * loss_action
            return loss_total, {
                "loss_video": 0.0,
                "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            }

        # Sample video noise and timestep.
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        # Replace first frame with clean latent (conditioning).
        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        # Pre-dit: embed tokens, compute RoPE, time modulation, context.
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        # Build separate masks for the decoupled forward.
        video_mask, action_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )

        # Video forward captures all KV layers, then action mixes them per layer.
        # CRITICAL: Use forward_decoupled(), NOT self.mot() which raises NotImplementedError.
        tokens_out = self.mot.forward_decoupled(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_masks={
                "video": video_mask,
                "action": action_mask,
            },
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                } if action_pre["context"] is not None else None,
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
            # Passed only when present: this training_loss is shared with plain
            # MoTDecoupled (static / fused_mlp / fused_kv), whose
            # forward_decoupled() has no such parameter. Only the FasterWAM
            # aligned subclass that hosts the EEF modes accepts it.
            **(
                {"eef_anchor_token": inputs["eef_anchor_token"]}
                if inputs.get("eef_anchor_token") is not None
                else {}
            ),
        )

        # Post-dit: project tokens back to prediction space.
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        # Strip first-frame prediction (not a denoising target when fusing).
        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        # Video loss: per-sample MSE weighted by scheduler training weight.
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        # Action loss: per-token MSE with pad masking.
        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)  # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_action_noise_with_video_kv(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_per_layer: list[dict[str, torch.Tensor]],
        action_attention_mask: torch.Tensor,
        eef_anchor_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict action velocity using all video-layer K/V.

        Args:
            latents_action: Noisy action latents ``[B, Sa, action_dim]``.
            timestep_action: Action timestep ``[B]``.
            context: Text/proprio context ``[B, L, D]``.
            context_mask: Context attention mask ``[B, L]``.
            video_kv_per_layer: List of per-layer video K/V, each a dict
                ``{"k": [B, Sv, H*Dh], "v": [B, Sv, H*Dh]}``. The list length
                depends on the KV-source mode: it is ``video_num_layers`` (one
                entry per video layer) for the selected / ``fused_kv`` modes, but
                ``action_num_layers`` (one fused entry per action layer) for the
                ``fused_mlp`` mode, where ``prefill_video_kv`` has already fused
                the N video layers into M action-layer K/V.
            action_attention_mask: Rectangular mask ``[Sa, Sv + Sa]``.

        Returns:
            Predicted action velocity ``[B, Sa, action_dim]``.
        """
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_kv(
            video_kv_per_layer=video_kv_per_layer,
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            } if action_pre["context"] is not None else None,
            action_attention_mask=action_attention_mask,
            **(
                {"eef_anchor_token": eef_anchor_token}
                if eef_anchor_token is not None
                else {}
            ),
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    def _predict_action_noise_with_final_kv(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        final_video_kv: dict[str, torch.Tensor],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Backward-compat wrapper. Prefer ``_predict_action_noise_with_video_kv()`` instead."""
        if isinstance(final_video_kv, dict):
            video_kv_per_layer = [final_video_kv] * self.mot.video_num_layers
        else:
            video_kv_per_layer = final_video_kv
        return self._predict_action_noise_with_video_kv(
            latents_action=latents_action,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            video_kv_per_layer=video_kv_per_layer,
            action_attention_mask=action_attention_mask,
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        eef_anchor_token: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Infer actions using the decoupled KV cache path.

        Prefills the video expert once to obtain all video-layer K/V, then
        denoises actions for
        ``num_inference_steps`` Euler steps using the cached K/V.

        Args:
            prompt: Text prompt (mutually exclusive with context/context_mask).
            input_image: Conditioning image ``[1, 3, H, W]`` or ``[3, H, W]``.
            action_horizon: Number of action steps to predict.
            proprio: Optional proprioceptive state ``[D]`` or ``[1, D]``.
            context: Precomputed text context ``[B, L, D]``.
            context_mask: Context attention mask ``[B, L]``.
            negative_prompt: Unused (kept for API compatibility).
            text_cfg_scale: Unused.
            num_inference_steps: Number of Euler denoising steps.
            sigma_shift: Optional override for inference sigma shift.
            seed: Random seed for action noise.
            rand_device: Device to use for random number generation.
            tiled: Whether to use tiled VAE encoding.

        Returns:
            Dict ``{"action": tensor}`` with action shape ``[action_horizon, action_dim]``.
        """
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        # --- Input validation (same as baseline) ---
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 "
                f"but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        # --- Initialize action noise ---
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        def _sync_ms(t0: float) -> float:
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled
        )
        vae_ms = _sync_ms(t0)

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], "
                    f"got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        text_enc_ms = _sync_ms(t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        proprio_ms = _sync_ms(t0)

        result = self._infer_action_core(
            first_frame_latents=first_frame_latents,
            latents_action=latents_action,
            context=context,
            context_mask=context_mask,
            action_horizon=action_horizon,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            eef_anchor_token=eef_anchor_token,
        )

        core_timing = result.get("timing_ms", {})
        result["timing_ms"] = {
            "text_enc_ms": text_enc_ms,
            "proprio_ms": proprio_ms,
            "vae_ms": vae_ms,
            "video_prefill_ms": core_timing.get("video_prefill_ms", 0.0),
            "denoise_ms": core_timing.get("denoise_ms", 0.0),
            "num_denoise_steps": num_inference_steps,
        }
        return result

    @torch.no_grad()
    def _encode_input_image_latents_batch(self, input_image: torch.Tensor) -> torch.Tensor:
        """VAE-encode ``[B, 3, H, W]`` conditioning frames in a single pass.

        ``WanVideoVAE38.encode()`` takes a *list* of videos and loops over it
        (`wan_video_vae.py:1218-1233`), so routing a batch through it would run
        ``B`` sequential encodes. The underlying ``model.encode()`` is already
        batch-general, so this calls ``single_encode()`` once with a real
        ``[B, C, T, H, W]`` tensor instead.
        """
        video = input_image.unsqueeze(2)  # [B, 3, T=1, H, W]
        return self.vae.single_encode(video, self.device)

    @torch.no_grad()
    def infer_action_batch(
        self,
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
        tiled: bool = False,
        eef_anchor_token: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Serve ``B`` independent action requests in one forward pass.

        Every sample is denoised from the *same* initial noise tensor, broadcast
        across the batch. Because `infer_action()` already seeds a fresh
        generator per call from a run-constant ``seed``, this reproduces the
        batch-one initial condition exactly in every slot, and makes a sample's
        result independent of both its batch slot and the batch composition.

        Only precomputed ``context``/``context_mask`` are accepted: batching
        `encode_prompt` would require the text encoder to be resident, which the
        LIBERO server explicitly avoids (``model.load_text_encoder=false``).

        Args:
            input_image: Conditioning images ``[B, 3, H, W]``.
            action_horizon: Number of action steps to predict.
            context: Precomputed text context ``[B, L, D]``.
            context_mask: Context attention mask ``[B, L]``.
            proprio: Optional proprioceptive state ``[B, D]``.
            num_inference_steps: Number of Euler denoising steps.
            sigma_shift: Optional override for inference sigma shift.
            seed: Seed for the shared action noise draw (all slots identical).
            seeds: One seed per batch slot, drawn INDEPENDENTLY per row. Takes precedence
                over ``seed``. Use this when each request must be its own sample: a row's
                noise then depends only on its own seed, not on batch composition or
                arrival order. Length must equal the batch size; ``None`` in a slot means
                an unseeded draw for that slot.
            rand_device: Device used for the noise draw.
            tiled: Must be false; batched inference does not support tiled VAE encoding.

        Returns:
            Dict with ``action`` of shape ``[B, action_horizon, action_dim]``
            and a ``timing_ms`` breakdown for the whole batch.
        """
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action_batch` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [B,3,H,W], got {tuple(input_image.shape)}"
            )
        batch_size, _, height, width = input_image.shape
        if batch_size < 1:
            raise ValueError("`infer_action_batch` requires a non-empty batch")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 "
                f"but got HxW=({height},{width})"
            )
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], "
                f"got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if context.shape[0] != batch_size or context_mask.shape[0] != batch_size:
            raise ValueError(
                f"`context/context_mask` batch {context.shape[0]}/{context_mask.shape[0]} "
                f"does not match `input_image` batch {batch_size}"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim != 2 or proprio.shape[0] != batch_size:
                raise ValueError(
                    f"`proprio` must be [B,D] with B={batch_size}, got {tuple(proprio.shape)}"
                )
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        # Noise draw. Two modes, and the distinction is a correctness one:
        #
        # `seeds` (per-slot) draws each row from its OWN generator, so a row's noise depends only
        # on its own seed -- never on batch composition or arrival order. A serving layer that
        # derives a seed per request (e.g. blake2b(episode_id + step_index)) needs this: without
        # it, two independent episodes batched together share one noise draw, and the same job
        # replayed at a different batch size produces different results.
        #
        # `seed` (scalar) is the original behaviour, kept for backward compatibility: one draw
        # broadcast to every slot -- batch-invariant, but the slots are duplicates of each other
        # rather than independent samples.
        _shape = (1, action_horizon, self.action_expert.action_dim)
        if seeds is not None:
            if len(seeds) != batch_size:
                raise ValueError(
                    f"`seeds` has {len(seeds)} entries but batch_size is {batch_size}"
                )
            _rows = []
            for _s in seeds:
                _g = None if _s is None else torch.Generator(device=rand_device).manual_seed(_s)
                _rows.append(
                    torch.randn(_shape, generator=_g, device=rand_device, dtype=torch.float32)
                )
            noise = torch.cat(_rows, dim=0)
        else:
            generator = (
                None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
            )
            noise = torch.randn(
                _shape, generator=generator, device=rand_device, dtype=torch.float32
            ).expand(batch_size, -1, -1)
        latents_action = noise.contiguous().to(device=self.device, dtype=self.torch_dtype)

        def _sync_ms(t0: float) -> float:
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0

        if tiled:
            raise NotImplementedError("`infer_action_batch` does not support tiled VAE encoding.")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_batch(input_image)
        vae_ms = _sync_ms(t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        proprio_ms = _sync_ms(t0)

        result = self._infer_action_core(
            first_frame_latents=first_frame_latents,
            latents_action=latents_action,
            context=context,
            context_mask=context_mask,
            action_horizon=action_horizon,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            return_batch=True,
            eef_anchor_token=eef_anchor_token,
        )

        core_timing = result.get("timing_ms", {})
        result["timing_ms"] = {
            "text_enc_ms": 0.0,
            "proprio_ms": proprio_ms,
            "vae_ms": vae_ms,
            "video_prefill_ms": core_timing.get("video_prefill_ms", 0.0),
            "denoise_ms": core_timing.get("denoise_ms", 0.0),
            "num_denoise_steps": num_inference_steps,
            "batch_size": batch_size,
        }
        return result

    def _assert_inference_masks_match_training_masks(
        self,
        *,
        video_self_mask: torch.Tensor,
        action_attention_mask: torch.Tensor,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> None:
        """Fail if decoupled inference masks drift from the training mask builder.

        Inference intentionally constructs the video prefill and action-query
        masks locally because those tensors are consumed at different stages of
        the KV-cache path. This guard keeps that local construction tied to the
        shared paradigm-C topology used by training.
        """
        expected_video_mask, expected_action_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        if (
            video_self_mask.shape != expected_video_mask.shape
            or not torch.equal(video_self_mask, expected_video_mask)
        ):
            raise RuntimeError(
                "Inference video mask diverged from _build_mot_attention_mask(). "
                "This would make training and inference use different paradigm-C "
                "attention topology."
            )
        if (
            action_attention_mask.shape != expected_action_mask.shape
            or not torch.equal(action_attention_mask, expected_action_mask)
        ):
            raise RuntimeError(
                "Inference action mask diverged from _build_mot_attention_mask(). "
                "This would make training and inference use different paradigm-C "
                "attention topology."
            )

    def _infer_action_core(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        return_batch: bool = False,
        eef_anchor_token: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Core action inference logic: prefill video KV + Euler denoising loop.

        Factored out so that both ``infer_action()`` (encodes image itself) and
        ``infer()`` (already has encoded latents) can share this without
        re-encoding the conditioning image.

        Args:
            return_batch: When ``False`` (default) ``result["action"]`` is the
                single sample ``latents_action[0]``, preserving the batch-one
                contract of every existing caller. When ``True`` the full
                ``[B, Sa, action_dim]`` tensor is returned, for
                ``infer_action_batch()``.
        """
        def _sync_ms(t0: float) -> float:
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0

        if eef_anchor_token is not None:
            if not torch.is_tensor(eef_anchor_token):
                eef_anchor_token = torch.as_tensor(eef_anchor_token)
            expected = (latents_action.shape[0], 2, 2)
            if eef_anchor_token.shape != torch.Size(expected):
                raise ValueError(
                    f"eef_anchor_token must be {expected} with cameras "
                    f"(main,wrist) and coordinates (y,x), got "
                    f"{tuple(eef_anchor_token.shape)}"
                )
            if not torch.isfinite(eef_anchor_token).all():
                raise ValueError("eef_anchor_token contains nonfinite values")
            # One observation's anchor is reused for every Euler step below and
            # discarded when the actor replans from the next observation.
            eef_anchor_token = eef_anchor_token.to(
                device=self.device, dtype=torch.float32, non_blocking=True
            )

        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])

        # Build video self-attention mask for prefill.
        video_self_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        # Prefill: run all video layers and keep each layer's KV.
        video_kv_per_layer = self.mot.prefill_video_kv(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_self_mask,
        )
        video_prefill_ms = _sync_ms(t0)

        # Build action attention mask: [Sa, Sv + Sa].
        # For single-frame inference, Sv == tokens_per_frame, so all video
        # columns correspond to the first frame and are all-True.
        action_attention_mask = torch.zeros(
            (action_horizon, video_seq_len + action_horizon),
            dtype=torch.bool,
            device=video_pre["tokens"].device,
        )
        first_frame_tokens = min(tokens_per_frame, video_seq_len)
        action_attention_mask[:, :first_frame_tokens] = True
        action_attention_mask[:, video_seq_len:] = True
        self._assert_inference_masks_match_training_masks(
            video_self_mask=video_self_mask,
            action_attention_mask=action_attention_mask,
            video_seq_len=video_seq_len,
            action_seq_len=action_horizon,
            video_tokens_per_frame=tokens_per_frame,
            device=video_pre["tokens"].device,
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            # Expand the scalar schedule step to one timestep per batch element;
            # a no-op when B == 1, required when serving a batched request group.
            timestep_action = (
                step_t_action.unsqueeze(0)
                .to(dtype=latents_action.dtype, device=self.device)
                .expand(latents_action.shape[0])
            )
            pred_action = self._predict_action_noise_with_video_kv(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_per_layer=video_kv_per_layer,
                action_attention_mask=action_attention_mask,
                eef_anchor_token=eef_anchor_token,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )

        denoise_ms = _sync_ms(t0)

        return {
            "action": (
                latents_action if return_batch else latents_action[0]
            ).detach().to(device="cpu", dtype=torch.float32),
            "timing_ms": {
                "video_prefill_ms": video_prefill_ms,
                "denoise_ms": denoise_ms,
            },
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        eef_anchor_token: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Full inference: video denoising + action denoising (independent stages).

        This is the eval inference path called by ``trainer.evaluate()``. It must
        return ``{"video": list_of_PIL_frames, "action": action_tensor}`` to
        match the baseline interface.

        Stage 1: Denoise video using the standalone ``video_expert.forward()``
        through N Euler steps. Clamp the first frame at each step. Decode via VAE.

        Stage 2: Prefill all-layer video K/V from the clean first frame,
        then denoise actions using ``_predict_action_noise_with_video_kv()``.

        Args:
            prompt: Text prompt (mutually exclusive with context/context_mask).
            input_image: Conditioning image ``[1, 3, H, W]`` or ``[3, H, W]``.
            num_frames: Number of video frames to generate (must satisfy T%4==1).
            action: Optional GT action for video conditioning (unused in decoupled).
            action_horizon: Number of action steps to predict.
            proprio: Optional proprioceptive state.
            context: Precomputed text context.
            context_mask: Context attention mask.
            negative_prompt: Unused.
            text_cfg_scale: Unused.
            action_cfg_scale: Unused.
            num_inference_steps: Number of Euler denoising steps.
            sigma_shift: Optional override for inference sigma shift.
            seed: Random seed.
            rand_device: Device for random number generation.
            tiled: Whether to use tiled VAE encoding/decoding.
            eef_anchor_token: Optional ``[B, 2, 2]`` EEF anchors, required by the
                EEF-relative RoPE modes. Stage 1 does not take it: those modes
                only re-anchor the action-facing K RoPE, leaving the video
                expert's own self-attention RoPE untouched.

        Returns:
            Dict ``{"video": list_of_PIL_frames, "action": action_tensor}``.
        """
        self.eval()

        # --- Input validation ---
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 "
                f"but got HxW=({height},{width})"
            )
        if checked_t != num_frames:
            raise ValueError(
                f"`num_frames` must satisfy T % 4 == 1, got {num_frames}"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        # --- Initialize latents ---
        latent_t = (num_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)

        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        if action_horizon is not None:
            latents_action = torch.randn(
                (1, action_horizon, self.action_expert.action_dim),
                generator=action_generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
        else:
            latents_action = None

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled
        )
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        # --- Resolve text context ---
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], "
                    f"got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        # ====================================================================
        # Stage 1: Video denoising (standalone video expert, N Euler steps)
        # Uses video_expert.forward() which runs all 30 layers with its own
        # self-attention mask internally. This is the same approach as
        # FastWAMIDM.infer_joint() Stage 1.
        # ====================================================================
        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video in zip(infer_timesteps_video, infer_deltas_video):
            timestep_video = step_t_video.unsqueeze(0).to(
                dtype=latents_video.dtype, device=self.device
            )
            # Standalone video forward: pre_dit -> all blocks -> post_dit.
            pred_video = self.video_expert(
                x=latents_video,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=action,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta_video, latents_video
            )
            # Clamp first frame to conditioning latent at each step.
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        # Decode video latents to PIL frames.
        video_frames = self._decode_latents(latents_video, tiled=tiled)

        # ====================================================================
        # Stage 2: Action denoising (if action_horizon is specified)
        # Prefill video KV from clean first frame, then denoise action.
        # ====================================================================
        action_out = None
        if latents_action is not None and action_horizon is not None:
            action_result = self._infer_action_core(
                first_frame_latents=first_frame_latents,
                latents_action=latents_action,
                context=context,
                context_mask=context_mask,
                action_horizon=action_horizon,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                eef_anchor_token=eef_anchor_token,
            )
            action_out = action_result["action"]

        return {
            "video": video_frames,
            "action": action_out,
        }

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str] = None,
        input_image: Optional[torch.Tensor] = None,
        num_video_frames: int = 0,
        action_horizon: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Raises NotImplementedError -- decoupled MoT does not support joint inference.

        Use ``infer()`` (video + action independently) or ``infer_action()`` (action only).
        """
        raise NotImplementedError(
            "Decoupled MoT does not support joint inference. Use infer() instead."
        )
