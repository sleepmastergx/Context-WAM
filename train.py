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

ARM_CFG = {"control": "fastwam_ttt_m5_control", "ttt": "fastwam_ttt_m5"}


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
        stats = None
        if self.chain is not None:
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
            gate_init=mm["gate_init"]).to(device).float()
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


def save_ckpt(path, step, cfg, model, memory, ema):
    torch.save({"step": step, "cfg": cfg,
                "model": model.state_dict(),
                "memory": memory.state_dict() if memory is not None else None,
                "ema": ema.state_dict() if ema is not None else None}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("control", "ttt"), required=True)
    ap.add_argument("--config", default=str(HERE / "configs/train_movecube.yaml"))
    ap.add_argument("--cache", default=os.environ.get("CACHE_DIR"),
                    help="window cache dir; defaults to $CACHE_DIR "
                         "(set by setup.sh's env.sh)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=None, help="cap for smoke runs")
    ap.add_argument("--synthetic", action="store_true",
                    help="CPU: fake data + stub model — verifies the LOOP only")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["model"] = yaml.safe_load(
        open(HERE / f"configs/model/{ARM_CFG[args.arm]}.yaml"))
    is_ttt = args.arm == "ttt"
    if is_ttt != bool(cfg["model"].get("memory", {}).get("enabled", False)):
        raise SystemExit(f"--arm {args.arm} does not match memory.enabled in "
                         f"{ARM_CFG[args.arm]}.yaml — refusing to guess")

    out = pathlib.Path(args.out or os.path.join(
        os.environ.get("OUT_ROOT", str(HERE / "runs")), f"fwam_{args.arm}"))

    # ---------------------------------------------------------------- setup
    if args.synthetic:
        accelerator, device, world, rank, is_main = None, torch.device("cpu"), 1, 0, True
    else:
        from accelerate import Accelerator
        accelerator = Accelerator(mixed_precision=cfg.get("mixed_precision", "bf16"))
        device = accelerator.device
        world = accelerator.num_processes
        rank = accelerator.process_index
        is_main = accelerator.is_main_process
    if is_main:
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)

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
        cache = GPUWindowCache(args.cache, device,
                               split_episodes=list(range(cfg["train_episodes"])))
    exec_idx = cache.exec_indices()
    per_rank = max(1, cfg["batch_size"] // world)
    steps_per_epoch = max(1, len(exec_idx) // (per_rank * world))
    total_steps = args.steps or steps_per_epoch * cfg["num_epochs"]

    # ---------------------------------------------------------------- model
    model, memory = build_arm(cfg, device, args.synthetic)
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
    opt = torch.optim.AdamW(groups, lr=cfg["learning_rate"], betas=(0.9, 0.95),
                            weight_decay=cfg["weight_decay"])
    sched = make_sched(opt, total_steps)

    mem_opt = mem_sched = None
    if memory is not None:
        if args.synthetic:
            rest = [p for n, p in memory.named_parameters() if n != "alpha"]
            mem_groups = [{"params": rest},
                          {"params": [memory.alpha], "weight_decay": 0.0}]
        else:
            from build_model import memory_param_groups
            mem_groups = memory_param_groups(memory)
        mem_opt = torch.optim.AdamW(mem_groups, lr=cfg["learning_rate"],
                                    betas=(0.9, 0.95),
                                    weight_decay=cfg["weight_decay"])
        mem_sched = make_sched(mem_opt, total_steps)

    # ---------------------------------------------------------- accelerate
    if accelerator is not None:
        dsp = getattr(accelerator.state, "deepspeed_plugin", None)
        if dsp is not None:
            dsp.deepspeed_config["train_micro_batch_size_per_gpu"] = per_rank
            dsp.deepspeed_config["gradient_accumulation_steps"] = 1
            dsp.deepspeed_config["train_batch_size"] = per_rank * world
            dsp.deepspeed_config["gradient_clipping"] = cfg["max_grad_norm"]
        arm, opt, sched = accelerator.prepare(arm, opt, sched)
        raw = accelerator.unwrap_model(arm)
    else:
        dsp = None
        raw = arm

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
              f"{len(exec_idx)} exec windows | batch {per_rank}/rank x {world} "
              f"| {steps_per_epoch} steps/epoch -> {total_steps} steps", flush=True)

    wb = None
    if is_main and not args.synthetic and cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb
            wb = wandb.init(project=cfg["wandb"]["project"],
                            group=cfg["wandb"]["group"],
                            name=f"fwam_{args.arm}", dir=str(out),
                            tags=["fastwam", "movecube", args.arm],
                            config={k: v for k, v in cfg.items() if k != "model"})
            print(f"wandb: {wb.url}", flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"wandb disabled ({e})", flush=True)

    # ---------------------------------------------------------------- loop
    g = torch.Generator(device="cpu").manual_seed(cfg["seed"] + rank)
    t0 = time.time()
    for gstep in range(1, total_steps + 1):
        sel = exec_idx[torch.randint(len(exec_idx), (per_rank,), generator=g)
                       .to(exec_idx.device)]
        batch = cache.batch(sel)

        opt.zero_grad(set_to_none=True)
        if mem_opt is not None:
            mem_opt.zero_grad(set_to_none=True)

        loss, stats = arm(batch, sel)
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()

        # memory params live outside the engine: reduce + clip + step manually
        mem_gnorm = 0.0
        if memory is not None:
            if accelerator is not None and world > 1:
                for p in mem_params:
                    if p.grad is not None:
                        p.grad = accelerator.reduce(p.grad, reduction="mean")
            mem_gnorm = float(torch.nn.utils.clip_grad_norm_(
                mem_params, cfg["max_grad_norm"]))
            mem_opt.step()
            mem_sched.step()
        if dsp is None:
            # DeepSpeed clips engine grads itself (gradient_clipping above)
            clip = [p for _, p in tracked if not _.startswith("memory.")]
            if accelerator is not None:
                accelerator.clip_grad_norm_(clip, cfg["max_grad_norm"])
            else:
                torch.nn.utils.clip_grad_norm_(clip, cfg["max_grad_norm"])
        opt.step()
        sched.step()
        if ema is not None:
            ema.update(tracked_params)

        if is_main and (gstep % cfg["log_every"] == 0 or gstep == 1):
            rec = {"step": gstep, "loss": float(loss.detach()),
                   "lr": sched.get_last_lr()[0],
                   "steps_per_s": gstep / max(time.time() - t0, 1e-9)}
            if torch.cuda.is_available():
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

        if is_main and (gstep % cfg["save_every"] == 0 or gstep == total_steps):
            save_ckpt(out / "checkpoints" / f"step_{gstep}.pt",
                      gstep, cfg, raw.model, memory, ema)

    if is_main:
        print(f"done: {total_steps} steps in {time.time()-t0:.0f}s", flush=True)
    if wb is not None:
        wb.finish()
    if accelerator is not None:
        accelerator.end_training()


if __name__ == "__main__":
    main()
