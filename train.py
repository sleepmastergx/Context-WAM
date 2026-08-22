"""Train the MoveCube arms — portable, both arms, ONE sampler.

    control : fastwam_m5 baseline, no memory        --arm control
    ttt     : context-wam, sliding w=8 memory       --arm ttt

Both arms draw the SAME uniform-random exec windows (seeded per rank). The
TTT arm additionally runs the in-graph sliding chain (sliding_chain.py) each
step; that is the entire difference between the arms, so any gap is
attributable to the memory.

Distributed: accelerate + DeepSpeed ZeRO-1 (configs/accelerate_zero1_ds.yaml,
their recipe — ~5B trainable; replicated AdamW state alone would be 56 GiB).
The 5B model goes through the engine in bf16. The 2.25M TTT memory stays
OUTSIDE the engine in fp32 — DeepSpeed bf16 would quantize the inner TTT
update (a forward-pass gradient) to noise — with its own AdamW; its gradients
are reduced across ranks manually.

Launch (from the repo root — the Wan loader resolves ./checkpoints from cwd):
    accelerate launch --config_file configs/accelerate_zero1_ds.yaml \
        --num_processes $NGPU train.py --arm ttt --cache $CACHE_DIR

Synthetic CPU smoke (no accelerate, no weights, verifies the LOOP only):
    python train.py --arm ttt --synthetic --steps 3
"""
import argparse
import contextlib
import json
import os
import pathlib
import sys
import time

# ---- RunPod volume defaults ------------------------------------------------
# The training image exports all of these, but a pod template's env vars
# OVERRIDE the image's ENV, and a bare `docker run`/ssh shell may carry none of
# them — in which case `--cache` resolves to None and the run dies on argument
# parsing rather than on anything real. Filling them in here makes
# `git clone && python train.py --arm ttt` work with no env setup at all.
#
# setdefault, so an explicitly exported value always wins; and only when the
# path already exists, so a non-RunPod machine is untouched (there is no
# /workspace there, and CACHE_DIR stays unset exactly as before).
# Set before importing torch/transformers: HF_HOME is read at import time.
for _var, _path in (("HF_HOME", "/workspace/hf_cache"),
                    ("MODELSCOPE_CACHE", "/workspace/hf_cache/modelscope"),
                    ("DIFFSYNTH_MODEL_BASE_PATH", "/workspace/checkpoints/"),
                    ("CACHE_DIR", "/workspace/data/movecube_fastwam"),
                    ("OUT_ROOT", "/workspace/outputs")):
    if not os.environ.get(_var) and os.path.isdir(_path.rstrip("/") or "/"):
        os.environ[_var] = _path
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np
import torch
import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "context_wam"))

from sliding_chain import SlidingChain          # noqa: E402

ARM_CFG = {
    "control": "fastwam_ttt_m5_control",
    "ttt": "fastwam_ttt_m5",
    "ttt_tokens": "fastwam_ttt_m5_tokens",
    "original": "fastwam_original_m30",
}


class EMA:
    """Their trainer has none. Our DP runs evaluated EMA weights and it
    mattered; on 100 episodes it is the cheapest guard against a noisy
    from-scratch run. Shadows are fp32 and SAVED in every checkpoint."""

    def __init__(self, named_params, decay=0.9999):
        self.decay = float(decay)
        self.names = [n for n, _ in named_params]
        self.shadow = [p.detach().clone().float() for _, p in named_params]

    @torch.no_grad()
    def update(self, params):
        for s, p in zip(self.shadow, params):
            s.mul_(self.decay).add_(p.detach().float(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {n: s for n, s in zip(self.names, self.shadow)}


class ArmModule(torch.nn.Module):
    """Engine-facing container. The MODEL lives inside (DeepSpeed manages its
    params, bf16, ZeRO-1). The memory/chain attach as a plain attribute so the
    engine never touches the fp32 memory params — they are optimized
    separately in main()."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.chain = None            # SlidingChain (not an nn.Module) or None

    def forward(self, batch, idx=None):
        """idx: legacy path (chain rolled inside the forward, one chain per
        micro-batch). main() now rolls the chain ONCE per optimizer step and
        installs the readouts before each micro-batch (chain-once protocol),
        so it calls forward(batch) with idx=None."""
        stats = None
        if self.chain is not None and idx is not None:
            stats = self.chain.load_states(idx)   # in-graph; sets memory._m
        loss = self.model.training_loss(batch)
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        return loss, stats


class SyntheticWindowCache:
    """Duck-typed GPUWindowCache with realistic episode geometry, for the CPU
    smoke. Random latents/actions; video->exec boundary ~200."""

    def __init__(self, n_episodes=4, seed=0, device="cpu"):
        rng = np.random.default_rng(seed)
        lat, act, sta, ep, start, is_ex = [], [], [], [], [], []
        for e in range(n_episodes):
            T = int(rng.integers(280, 400))
            exec_start = int(rng.integers(180, 220))
            n = T - 33 + 1
            lat.append(torch.randn(n, 48, 3, 16, 32) * 0.5)
            A = torch.randn(T, 8)
            idx = np.clip(np.arange(n)[:, None] + np.arange(32)[None, :], 0, T - 1)
            act.append(A[torch.from_numpy(idx)])
            sta.append(torch.randn(n, 8))
            ep.append(torch.full((n,), e, dtype=torch.int32))
            start.append(torch.arange(n, dtype=torch.int32))
            is_ex.append(torch.arange(n) >= exec_start)
        self.latents = torch.cat(lat).to(device)
        self.actions = torch.cat(act).to(device)
        self.states = torch.cat(sta).to(device)
        self.ep = torch.cat(ep).to(device)
        self.start = torch.cat(start).to(device)
        self.is_exec = torch.cat(is_ex).to(device)
        self.context = torch.zeros(4, 4096)
        self.context_mask = torch.ones(4, dtype=torch.bool)
        self.source_video_shape = torch.tensor([[3, 9, 256, 512]])
        self.action_stats = None

    def exec_indices(self):
        return torch.nonzero(self.is_exec, as_tuple=False).squeeze(-1)

    def batch(self, idx):
        B = idx.shape[0]
        return {"video_latents": self.latents[idx],
                "action": self.actions[idx],
                # [B,1,P] — mirrors GPUWindowCache.batch(); the two caches are
                # duck-typed and must agree, or the synthetic smoke keeps
                # passing while the real model path raises on the shape.
                "proprio": self.states[idx].unsqueeze(1),
                "is_exec": self.is_exec[idx],
                "context": self.context.unsqueeze(0).expand(B, -1, -1),
                "context_mask": self.context_mask.unsqueeze(0).expand(B, -1),
                "source_video_shape": self.source_video_shape.expand(B, -1)}


def build_arm(cfg, device, synthetic):
    if not synthetic:
        from build_model import build
        from omegaconf import OmegaConf
        return build(OmegaConf.create(cfg), device=str(device))
    from per_layer_memory import (PerLayerEpisodeMemory,
                                  patch_mot_action_expert, tag_action_blocks)
    from synthetic_model import SyntheticFastWAM
    m = cfg["model"]
    model = SyntheticFastWAM(
        n_action_layers=m["action_dit_config"]["num_layers"],
        hidden_dim=m["action_dit_config"]["hidden_dim"]).to(device)
    memory = None
    if m.get("memory", {}).get("enabled", False):
        mm = m["memory"]
        memory = PerLayerEpisodeMemory(
            n_layers=mm["n_layers"], hidden_dim=mm["hidden_dim"],
            latent_channels=mm["latent_channels"],
            proprio_dim=m.get("proprio_dim", 8),
            d_k=mm["d_k"], d_v=mm["d_v"], d_hidden=mm["d_hidden"],
            d_out=mm["d_out"], chunk=mm["chunk"],
            gate_init=mm["gate_init"],
            write_input=mm.get("write_input", "pooled"),
            patch=mm.get("patch", 2)).to(device).float()
        tag_action_blocks(model.mot.mixtures["action"])
        patch_mot_action_expert(model.mot, memory)
    return model, memory


def make_sched(opt, total_steps):
    warmup = max(1, int(total_steps * 0.05))
    return torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, warmup),
         torch.optim.lr_scheduler.CosineAnnealingLR(
             opt, T_max=max(1, total_steps - warmup),
             eta_min=opt.param_groups[0]["lr"] * 0.01)],
        milestones=[warmup])


def save_ckpt(path, step, cfg, model_state, memory, ema, action_stats=None):
    torch.save({"step": step, "cfg": cfg,
                "model": model_state,
                "memory": memory.state_dict() if memory is not None else None,
                "ema": ema.state_dict() if ema is not None else None,
                # action/proprio normalization used at training time (None =
                # raw); the eval server inverts it -- never evaluate a
                # normalized-trained model as raw
                "action_stats": action_stats}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=tuple(ARM_CFG), required=True)
    ap.add_argument("--config", default=str(HERE / "configs/train_movecube.yaml"))
    ap.add_argument("--cache", default=os.environ.get("CACHE_DIR"),
                    help="window cache dir; defaults to $CACHE_DIR "
                         "(set by setup.sh's env.sh)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=None, help="cap for smoke runs")
    ap.add_argument("--resume", default=None,
                    help="weights checkpoint to resume from; restores step and "
                         "scheduler position (optimizer moments are not stored)")
    ap.add_argument("--synthetic", action="store_true",
                    help="CPU: fake data + stub model — verifies the LOOP only")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["model"] = yaml.safe_load(
        open(HERE / f"configs/model/{ARM_CFG[args.arm]}.yaml"))
    if "mot_checkpoint_mixed_attn" in cfg:
        cfg["model"]["mot_checkpoint_mixed_attn"] = bool(
            cfg["mot_checkpoint_mixed_attn"])
    if "video_gradient_checkpointing" in cfg:
        cfg["model"]["video_dit_config"]["use_gradient_checkpointing"] = bool(
            cfg["video_gradient_checkpointing"])
    if "action_gradient_checkpointing" in cfg:
        cfg["model"]["action_dit_config"]["use_gradient_checkpointing"] = bool(
            cfg["action_gradient_checkpointing"])
    is_ttt = args.arm.startswith("ttt")
    if is_ttt != bool(cfg["model"].get("memory", {}).get("enabled", False)):
        raise SystemExit(f"--arm {args.arm} does not match memory.enabled in "
                         f"{ARM_CFG[args.arm]}.yaml — refusing to guess")
    if args.resume:
        cfg["resume_from"] = str(pathlib.Path(args.resume).resolve())

    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    if grad_accum < 1:
        raise SystemExit("gradient_accumulation_steps must be >= 1")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(
            cfg.get("float32_matmul_precision", "high"))
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(cfg.get("allow_tf32", True))
        torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))

    out = pathlib.Path(args.out or os.path.join(
        os.environ.get("OUT_ROOT", str(HERE / "runs")), f"fwam_{args.arm}"))

    # ---------------------------------------------------------------- setup
    if args.synthetic:
        accelerator, device, world, rank, is_main = None, torch.device("cpu"), 1, 0, True
    else:
        from accelerate import Accelerator
        accelerator = Accelerator(
            mixed_precision=cfg.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=grad_accum,
        )
        device = accelerator.device
        world = accelerator.num_processes
        rank = accelerator.process_index
        is_main = accelerator.is_main_process
    if is_main:
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        with open(out / "run_config.yaml", "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ---------------------------------------------------------------- data
    if args.synthetic:
        cache = SyntheticWindowCache(n_episodes=4, seed=cfg["seed"], device=device)
    else:
        if not args.cache:
            raise SystemExit("pass --cache or `source env.sh` for $CACHE_DIR "
                             "(download: scripts/download_data.py)")
        from gpu_cache import GPUWindowCache
        # first train_episodes episodes train; the rest are held out for val,
        # mirroring the DP split discipline
        train_episodes = list(range(int(cfg["train_episodes"])))
        excluded = {int(e) for e in cfg.get("exclude_train_episodes", [])}
        train_episodes = [e for e in train_episodes if e not in excluded]
        if not train_episodes:
            raise SystemExit("training episode split is empty")
        cache_storage_device = str(cfg.get("cache_storage_device", device))
        # (the sliding chain moves its E x J inputs to the memory's device
        # itself, so a CPU-resident cache is fine for the ttt arm too)
        cache = GPUWindowCache(
            args.cache,
            device,
            split_episodes=train_episodes,
            storage_device=cache_storage_device,
            action_mode=str(cfg.get("action_mode", "raw")),
        )
    exec_idx = cache.exec_indices()
    effective_batch = int(cfg["batch_size"])
    batch_divisor = world * grad_accum
    configured_micro = (None if args.synthetic
                        else cfg.get("micro_batch_size_per_gpu"))
    if configured_micro is None:
        if effective_batch % batch_divisor:
            raise SystemExit(
                f"batch_size={effective_batch} must be divisible by "
                f"world_size({world}) * gradient_accumulation_steps({grad_accum})")
        per_rank = effective_batch // batch_divisor
    else:
        per_rank = int(configured_micro)
        actual_batch = per_rank * batch_divisor
        if actual_batch != effective_batch:
            raise SystemExit(
                f"batch_size={effective_batch}, but micro_batch_size_per_gpu="
                f"{per_rank} * world_size={world} * gradient_accumulation_steps="
                f"{grad_accum} gives {actual_batch}")
    if per_rank < 1:
        raise SystemExit("micro batch size per GPU must be >= 1")
    steps_per_epoch = max(1, len(exec_idx) // effective_batch)
    total_steps = args.steps or steps_per_epoch * cfg["num_epochs"]

    # ---------------------------------------------------------------- model
    model, memory = build_arm(cfg, device, args.synthetic)
    resume_step = 0
    if args.resume:
        resume_path = pathlib.Path(args.resume)
        if not resume_path.is_file():
            raise SystemExit(f"resume checkpoint not found: {resume_path}")
        payload = torch.load(
            resume_path, map_location="cpu", mmap=True, weights_only=False)
        resume_step = int(payload["step"])
        model.load_state_dict(payload["model"], strict=True)
        if is_ttt and payload.get("memory") is not None:
            memory.load_state_dict(payload["memory"], strict=True)
        del payload
        if is_main:
            print(f"resumed weights from {resume_path} at step "
                  f"{resume_step}; optimizer moments restart empty", flush=True)
    if (accelerator is not None and
            accelerator.distributed_type.name == "FSDP"):
        # FastWAM registers each expert twice: directly as
        # model.video_expert/action_expert and again under model.mot.mixtures.
        # FSDP cannot recursively wrap the same Module object through two
        # ownership paths. Keep the public attributes as non-registering
        # aliases; mot.mixtures remains the single registered owner.
        for expert_name in ("video_expert", "action_expert"):
            expert = model._modules.pop(expert_name)
            object.__setattr__(model, expert_name, expert)
    arm = ArmModule(model)
    if is_ttt:
        if memory is None:
            raise SystemExit("ttt arm built without a memory — config mismatch")
        arm.chain = SlidingChain(cache, memory, w=cfg["sliding_w"],
                                 checkpoint_every=cfg["chain_checkpoint_every"])

    if args.synthetic:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]
    else:
        from build_model import trainable_parameters
        groups = trainable_parameters(model, None)     # model only; memory below
        if memory is not None:
            memory.requires_grad_(True)
    fused_opt = bool(cfg.get("fused_optimizer", False) and device.type == "cuda")
    use_torch_zro = bool(
        cfg.get("zero_redundancy_optimizer", False)
        and accelerator is not None
        and accelerator.distributed_type.name == "MULTI_GPU"
    )
    if use_torch_zro:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        opt = ZeroRedundancyOptimizer(
            groups,
            optimizer_class=torch.optim.AdamW,
            lr=cfg["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=cfg["weight_decay"],
            fused=fused_opt,
        )
    else:
        opt = torch.optim.AdamW(
            groups,
            lr=cfg["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=cfg["weight_decay"],
            fused=fused_opt,
        )
    sched = make_sched(opt, total_steps)
    if resume_step:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(resume_step):
                sched.step()

    mem_opt = mem_sched = None
    if memory is not None:
        if args.synthetic:
            rest = [p for n, p in memory.named_parameters() if n != "alpha"]
            mem_groups = [{"params": rest},
                          {"params": [memory.alpha], "weight_decay": 0.0}]
        else:
            from build_model import memory_param_groups
            mem_groups = memory_param_groups(memory)
        # gradient accumulation: the chain runs once per optimizer step and
        # its readouts are handed to each micro-batch as detached leaves whose
        # grads accumulate; ONE chain backward then moves the summed gradient
        # into the write parameters (exact — see the loop below)
        mem_opt = torch.optim.AdamW(
            mem_groups,
            lr=cfg["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=cfg["weight_decay"],
            fused=fused_opt,
        )
        mem_sched = make_sched(mem_opt, total_steps)

    # ---------------------------------------------------------- accelerate
    if accelerator is not None:
        dsp = getattr(accelerator.state, "deepspeed_plugin", None)
        if dsp is not None:
            dsp.deepspeed_config["train_micro_batch_size_per_gpu"] = per_rank
            dsp.deepspeed_config["gradient_accumulation_steps"] = grad_accum
            dsp.deepspeed_config["train_batch_size"] = effective_batch
            dsp.deepspeed_config["gradient_clipping"] = cfg["max_grad_norm"]
            if cfg.get("deepspeed_wall_clock_breakdown", False):
                dsp.deepspeed_config["wall_clock_breakdown"] = True
                dsp.deepspeed_config["steps_per_print"] = 1
        if use_torch_zro:
            # Accelerate's optimizer wrapper calls state_dict() during
            # construction, which ZeroRedundancyOptimizer intentionally rejects
            # before an explicit cross-rank consolidation. Keep the native
            # optimizer/scheduler and prepare only the DDP model.
            arm = accelerator.prepare_model(arm)
        else:
            arm, opt, sched = accelerator.prepare(arm, opt, sched)
        raw = accelerator.unwrap_model(arm)
    else:
        dsp = None
        raw = arm

    if is_main:
        optimizer_chain = []
        current_opt = opt
        seen_optimizers = set()
        while current_opt is not None and id(current_opt) not in seen_optimizers:
            seen_optimizers.add(id(current_opt))
            defaults = getattr(current_opt, "defaults", {})
            fused = defaults.get("fused") if isinstance(defaults, dict) else None
            details = []
            if fused is not None:
                details.append(f"fused={fused}")
            if hasattr(current_opt, "clip_grad"):
                details.append(f"clip_grad={current_opt.clip_grad}")
            suffix = f"({', '.join(details)})" if details else ""
            optimizer_chain.append(type(current_opt).__name__ + suffix)
            current_opt = getattr(current_opt, "optimizer", None)
        print("optimizer=" + " -> ".join(optimizer_chain), flush=True)

    tracked = [(f"model.{n}", p) for n, p in raw.model.named_parameters()
               if p.requires_grad]
    if memory is not None:
        tracked += [(f"memory.{n}", p) for n, p in memory.named_parameters()]
    ema = EMA(tracked, cfg["ema"]["decay"]) if cfg["ema"]["enabled"] else None
    tracked_params = [p for _, p in tracked]
    mem_params = [p for _, p in tracked if _.startswith("memory.")] \
        if memory is not None else []

    n_tr = sum(p.numel() for p in tracked_params)
    if is_main:
        print(f"arm={args.arm} | trainable {n_tr/1e6:.2f}M "
              f"(memory {sum(p.numel() for p in mem_params)/1e6:.2f}M) | "
              f"{len(exec_idx)} exec windows | micro batch {per_rank}/rank x "
              f"{world} x accum {grad_accum} = {effective_batch} "
              f"| {steps_per_epoch} steps/epoch -> {total_steps} steps", flush=True)

    wb = None
    if is_main and not args.synthetic and cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb
            wb = wandb.init(project=cfg["wandb"]["project"],
                            group=cfg["wandb"]["group"],
                            name=f"fwam_{args.arm}", dir=str(out),
                            # task tag comes from the config, NOT hardcoded --
                            # it read "movecube" on every run until 2026-08-14,
                            # which mistagged the whole VideoUnmask study.
                            tags=[*cfg["wandb"].get("tags", ["fastwam"]),
                                  args.arm],
                            config={k: v for k, v in cfg.items() if k != "model"})
            print(f"wandb: {wb.url}", flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"wandb disabled ({e})", flush=True)

    # ---------------------------------------------------------------- loop
    # Sampling: ALL micro-batches of an optimizer step are drawn at once
    # (per_rank * grad_accum windows) so the memory chain can be rolled once
    # per step over their union; for grad_accum == 1 this is the historical
    # draw, so control/ttt pairs stay window-identical.
    #
    # Chain-once protocol (memory arm):
    #   ms, stats = chain.readouts(sel_all)          # in-graph, once per step
    #   leaves    = [m.detach().requires_grad_()]    # what the DiT sees
    #   for each micro-batch k:  memory._m = leaves[k-th slice]; loss_k.backward()
    #       -> d loss_k / d leaf accumulates on the leaves (scaled 1/accum by
    #          accelerate like every other grad)
    #   torch.autograd.backward(ms, [leaf.grad])     # ONE chain backward:
    #       sum_k dloss_k/dm_k * dm_k/dtheta == d(sum_k loss_k)/dtheta, exact.
    # The chain graph is therefore built once and traversed once per step,
    # whatever grad_accum is; the 5B DiT graph is still per micro-batch.
    g = torch.Generator(device="cpu").manual_seed(
        cfg["seed"] + rank + resume_step)
    t0 = time.time()
    timing_steps = int(cfg.get("timing_steps", 0))
    gstep = resume_step
    chain = raw.chain if memory is not None else None
    opt.zero_grad(set_to_none=True)
    if mem_opt is not None:
        mem_opt.zero_grad(set_to_none=True)
    while gstep < total_steps:
        timing_events = None
        if (is_main and torch.cuda.is_available() and
                gstep < timing_steps):
            timing_events = {
                name: torch.cuda.Event(enable_timing=True)
                for name in ("data_start", "data_end", "forward_end",
                             "backward_end", "optimizer_end")
            }
            timing_events["data_start"].record()
        sel_all = exec_idx[torch.randint(len(exec_idx),
                                         (per_rank * grad_accum,),
                                         generator=g).to(exec_idx.device)]

        # ---- memory chain: once per optimizer step, in-graph -----------
        stats, ms, leaves = None, None, None
        if chain is not None:
            ms, stats = chain.readouts(sel_all)           # list of [B_all, d_out]
            leaves = [m.detach().requires_grad_(True) for m in ms]

        accumulated_loss = 0.0
        engine_grad_norm = None
        for k in range(grad_accum):
            sel = sel_all[k * per_rank:(k + 1) * per_rank]
            batch = cache.batch(sel)
            if timing_events is not None and k == 0:
                timing_events["data_end"].record()
            if leaves is not None:
                chain.set_readouts(
                    [leaf[k * per_rank:(k + 1) * per_rank] for leaf in leaves])

            accumulate = (accelerator.accumulate(arm) if accelerator is not None
                          else contextlib.nullcontext())
            with accumulate:
                loss, _ = arm(batch)
                if timing_events is not None and k == grad_accum - 1:
                    timing_events["forward_end"].record()
                if not bool(torch.isfinite(loss.detach())):
                    raise FloatingPointError(
                        f"non-finite loss before backward at optimizer step "
                        f"{gstep + 1}, rank {rank}")
                accumulated_loss += float(loss.detach())
                sync_gradients = (accelerator.sync_gradients
                                  if accelerator is not None
                                  else k == grad_accum - 1)
                if accelerator is not None:
                    accelerator.backward(loss)      # scales by 1/grad_accum
                    if dsp is not None and sync_gradients:
                        engine_grad_norm = (
                            accelerator.deepspeed_engine_wrapped.get_global_grad_norm())
                else:
                    (loss / grad_accum).backward()
                if k == grad_accum - 1 and not sync_gradients:
                    raise RuntimeError(
                        "accelerate's accumulation counter is out of phase with "
                        "the optimizer-step loop (grad_accum mismatch)")
                if timing_events is not None and k == grad_accum - 1:
                    timing_events["backward_end"].record()

                # memory params live outside the engine: one chain backward
                # with the accumulated readout grads, then reduce + clip + step
                mem_gnorm = 0.0
                if memory is not None and sync_gradients:
                    grads = [leaf.grad if leaf.grad is not None
                             else torch.zeros_like(leaf) for leaf in leaves]
                    torch.autograd.backward(ms, grads)
                    if accelerator is not None and world > 1:
                        for p in mem_params:
                            if p.grad is not None:
                                p.grad = accelerator.reduce(p.grad, reduction="mean")
                    mem_gnorm = float(torch.nn.utils.clip_grad_norm_(
                        mem_params, cfg["max_grad_norm"]))
                    mem_opt.step()
                    mem_sched.step()
                    mem_opt.zero_grad(set_to_none=True)
                    ms = leaves = None
                if dsp is None and sync_gradients:
                    # DeepSpeed clips engine grads itself (gradient_clipping above)
                    if (accelerator is not None and
                            accelerator.distributed_type.name == "FSDP"):
                        accelerator.clip_grad_norm_(
                            arm.parameters(), cfg["max_grad_norm"])
                    else:
                        clip = [p for _, p in tracked
                                if not _.startswith("memory.")]
                    if accelerator is not None and accelerator.distributed_type.name != "FSDP":
                        accelerator.clip_grad_norm_(clip, cfg["max_grad_norm"])
                    elif accelerator is None:
                        torch.nn.utils.clip_grad_norm_(clip, cfg["max_grad_norm"])
                if sync_gradients:
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                if timing_events is not None and k == grad_accum - 1:
                    timing_events["optimizer_end"].record()

        gstep += 1
        step_loss = accumulated_loss / grad_accum
        if ema is not None:
            ema.update(tracked_params)

        if not args.synthetic and bool(cfg.get("fail_on_nonfinite", True)):
            probes = {
                "video.patch_embedding": raw.model.video_expert.patch_embedding.weight,
                "action.action_encoder": raw.model.action_expert.action_encoder.weight,
            }
            bad_probes = [
                name for name, value in probes.items()
                if not bool(torch.isfinite(value.detach()).all())
            ]
            if bad_probes:
                raise FloatingPointError(
                    f"non-finite parameters after optimizer step {gstep}, rank "
                    f"{rank}: {', '.join(bad_probes)}")

        step_timing = None
        if timing_events is not None:
            timing_events["optimizer_end"].synchronize()
            step_timing = {
                "data_ms": timing_events["data_start"].elapsed_time(
                    timing_events["data_end"]),
                "forward_ms": timing_events["data_end"].elapsed_time(
                    timing_events["forward_end"]),
                "backward_ms": timing_events["forward_end"].elapsed_time(
                    timing_events["backward_end"]),
                "optimizer_ms": timing_events["backward_end"].elapsed_time(
                    timing_events["optimizer_end"]),
            }

        if is_main and (gstep % cfg["log_every"] == 0 or gstep == 1):
            rec = {"step": gstep, "loss": step_loss,
                   "lr": sched.get_last_lr()[0],
                   "steps_per_s": ((gstep - resume_step)
                                   / max(time.time() - t0, 1e-9))}
            if engine_grad_norm is not None:
                rec["grad_norm"] = float(engine_grad_norm)
            if step_timing is not None:
                rec.update(step_timing)
            if torch.cuda.is_available():
                rec["vram_current_gib"] = (
                    torch.cuda.memory_allocated() / 2**30)
                rec["vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
            if memory is not None:
                rec["gate"] = float(torch.tanh(memory.alpha).abs().mean())
                rec["mem_gnorm"] = mem_gnorm
                if stats:
                    rec.update({k: stats[k] for k in
                                ("chain_J", "chain_E", "surprise")})
            print(json.dumps(rec), flush=True)
            with open(out / "log.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if wb is not None:
                wb.log({k: v for k, v in rec.items() if k != "step"}, step=gstep)

        save_every = int(cfg.get("save_every", 0))
        should_save = (
            (save_every > 0 and gstep % save_every == 0)
            or (bool(cfg.get("save_final", True)) and gstep == total_steps)
        )
        if should_save and accelerator is not None:
            accelerator.wait_for_everyone()
        model_state = None
        if should_save and accelerator is not None and \
                accelerator.distributed_type.name == "FSDP":
            arm_state = accelerator.get_state_dict(arm)
            if is_main:
                model_state = {
                    (name[len("model."):] if name.startswith("model.") else name): value
                    for name, value in arm_state.items()
                }
        elif is_main and should_save:
            model_state = raw.model.state_dict()
        if is_main and should_save:
            save_ckpt(out / "checkpoints" / f"step_{gstep}.pt",
                      gstep, cfg, model_state, memory, ema,
                      action_stats=getattr(cache, "action_stats", None))
        if should_save and accelerator is not None:
            accelerator.wait_for_everyone()

    if is_main:
        print(f"done: {total_steps} steps in {time.time()-t0:.0f}s", flush=True)
    if wb is not None:
        wb.finish()
    if accelerator is not None:
        accelerator.end_training()


if __name__ == "__main__":
    main()
