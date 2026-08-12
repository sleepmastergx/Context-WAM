"""Stage 2 on MoveCube: fresh UNet head on frozen stage-1 features.

    --no-ttt   arm 2  (control: no memory)
    --ttt      arm 4a (TTT concat small) with the flags below

Optimizer follows the RMBench stage-2 setup unchanged (AdamW 1e-4,
betas (0.95, 0.999), wd 1e-6, cosine + 500 warmup, EMA inv_gamma 1 power 0.75
max 0.9999, 600 epochs, 128 windows/step). Windows follow the STAGE-1 RoboMME
recipe: obs 2 / pred 16 / exec 8, sampled from execution frames only; the TTT
rollout always covers the full episode including the conditioning video.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset_mc import (EpisodeFeatureDatasetMC, GPUEpisodeBatcher,  # noqa: E402
                        gather_windows, sample_windows)
from model_mc import DPMemoryPolicyMC  # noqa: E402
from diffusers.optimization import get_scheduler  # noqa: E402


class EMA:
    """Power-schedule EMA matching diffusion_policy's EMAModel defaults
    (update_after_step 0, inv_gamma 1.0, power 0.75, max 0.9999)."""

    def __init__(self, model, inv_gamma=1.0, power=0.75, max_value=0.9999):
        self.model = model
        self.inv_gamma, self.power, self.max_value = inv_gamma, power, max_value
        self.step_count = 0

    @torch.no_grad()
    def step(self, src):
        self.step_count += 1
        decay = 1.0 - (1.0 + self.step_count / self.inv_gamma) ** -self.power
        decay = min(max(decay, 0.0), self.max_value)
        for p_ema, p in zip(self.model.parameters(), src.parameters()):
            p_ema.lerp_(p, 1.0 - decay)
        for b_ema, b in zip(self.model.buffers(), src.buffers()):
            b_ema.copy_(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--stats-path", required=True,
                    help="stage-1 run dir holding stats.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ttt", dest="use_ttt", action="store_true", default=True)
    ap.add_argument("--no-ttt", dest="use_ttt", action="store_false")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0e-6)
    ap.add_argument("--warmup", type=int, default=500)
    # stage-1 RoboMME window recipe
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--n-obs-steps", type=int, default=2)
    ap.add_argument("--n-action-steps", type=int, default=8)
    ap.add_argument("--episodes-per-step", type=int, default=8)
    ap.add_argument("--windows-per-episode", type=int, default=16)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    # TTT (4a = defaults)
    ap.add_argument("--d-k", type=int, default=128)
    ap.add_argument("--d-v", type=int, default=128)
    ap.add_argument("--d-hidden", type=int, default=512)
    ap.add_argument("--d-out", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--inject", default="concat", choices=["concat", "add"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--wandb-project", default="wam-ttt")
    ap.add_argument("--wandb-group", default="stage2")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false", default=True)
    ap.add_argument("--smoke", action="store_true",
                    help="CPU: 3 steps, 2 epochs, no wandb, tiny batch")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp = world > 1 and not args.smoke
    if ddp:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")
    elif args.smoke and not torch.cuda.is_available():
        dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)
    is_main = rank == 0
    if args.smoke:
        args.epochs, args.steps_per_epoch = 2, 3
        args.episodes_per_step, args.windows_per_episode = 2, 4
        args.wandb = False

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    out = pathlib.Path(args.out)
    if is_main:
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_ds = EpisodeFeatureDatasetMC(args.features, args.stats_path, "train")
    val_ds = EpisodeFeatureDatasetMC(args.features, args.stats_path, "val")
    print(f"train episodes {len(train_ds)} | val episodes {len(val_ds)} "
          f"| d_feat {train_ds.d_feat}", flush=True)

    spe = args.steps_per_epoch
    if spe is None:
        per_step = args.episodes_per_step * args.windows_per_episode * world
        total = sum((e - s) - xs for (s, e), xs
                    in zip(train_ds.episodes, train_ds.exec_starts))
        spe = max(1, total // per_step)
    loader = GPUEpisodeBatcher(train_ds, args.episodes_per_step, dev,
                               shuffle=True, seed=args.seed + rank,
                               windows_per_episode=args.windows_per_episode,
                               steps_per_epoch=spe)
    val_loader = GPUEpisodeBatcher(val_ds, 1, dev, shuffle=False, seed=0,
                                   windows_per_episode=64, steps_per_epoch=1)

    ttt_kwargs = dict(d_k=args.d_k, d_v=args.d_v, d_hidden=args.d_hidden,
                      d_out=args.d_out, chunk=args.chunk)
    def build():
        return DPMemoryPolicyMC(
            d_feat=train_ds.d_feat, action_dim=train_ds.d_action,
            horizon=args.horizon, n_obs_steps=args.n_obs_steps,
            n_action_steps=args.n_action_steps, use_ttt=args.use_ttt,
            ttt_kwargs=ttt_kwargs, inject=args.inject).to(dev)

    model = build()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_ttt = sum(p.numel() for p in model.ttt.parameters()) if args.use_ttt else 0
    print(f"use_ttt={args.use_ttt} | trainable {n_tr/1e6:.2f}M "
          f"(unet {sum(p.numel() for p in model.unet.parameters())/1e6:.2f}M, "
          f"ttt {n_ttt/1e6:.2f}M)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.95, 0.999),
                            eps=1e-8, weight_decay=args.weight_decay)
    sched = get_scheduler("cosine", opt, num_warmup_steps=args.warmup,
                          num_training_steps=len(loader) * args.epochs)
    ema_model = build()
    ema_model.load_state_dict(model.state_dict())
    ema = EMA(ema_model)

    raw = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
        print(f"[rank {rank}/{world}] DDP | effective batch = "
              f"{world*args.episodes_per_step*args.windows_per_episode} windows",
              flush=True)

    wb = None
    if args.wandb and is_main:
        import wandb
        wb = wandb.init(project=args.wandb_project,
                        name=args.run_name or out.name, group=args.wandb_group,
                        tags=["stage2", "ttt" if args.use_ttt else "control"],
                        dir=str(out), config=vars(args))
        print(f"wandb: {wb.url}", flush=True)

    log_path = out / "log.jsonl"
    gstep = 0
    for epoch in range(args.epochs):
        raw.train()
        t0, losses, surprises = time.time(), [], []
        for batch in loader:
            feat, action = batch["feat"], batch["action"]
            mask, lengths = batch["mask"], batch["length"]
            starts = sample_windows(lengths, batch["exec_start"],
                                    args.windows_per_episode)
            loss, diag = model(feat, action, mask, lengths, starts,
                               args.horizon, args.n_obs_steps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.step(raw)
            losses.append(loss.item())
            if diag:
                surprises.append(float(diag["surprise"].mean()))
            gstep += 1

        rec = {"epoch": epoch, "step": gstep,
               "train_loss": float(np.mean(losses)),
               "lr": sched.get_last_lr()[0],
               "sec": round(time.time() - t0, 1)}
        if surprises:
            rec["surprise"] = float(np.mean(surprises))
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            rec["val_mse"] = validate(ema.model, val_loader, args)
        if is_main:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            if wb is not None:
                wb.log({k: v for k, v in rec.items() if k != "step"}, step=gstep)
            if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
                torch.save({"epoch": epoch, "args": vars(args),
                            "d_feat": train_ds.d_feat,   # eval rebuilds from this
                            "model": raw.state_dict(),
                            "ema_model": ema.model.state_dict()},
                           out / "checkpoints" / f"{epoch + 1}.ckpt")

    if wb is not None:
        wb.finish()
    if ddp:
        torch.distributed.destroy_process_group()


@torch.no_grad()
def validate(model, val_loader, args):
    """Teacher-forced denoising MSE on the held-out episode, fixed windows.

    On MoveCube this metric is memory-SENSITIVE: identical covered
    observations demand different actions depending on the video, so a
    memoryless model has an irreducible floor here.
    """
    model.eval()
    tot, n = 0.0, 0
    g = torch.Generator().manual_seed(0)
    for batch in val_loader:
        feat, action = batch["feat"], batch["action"]
        mask, lengths = batch["mask"], batch["length"]
        m, _ = model.rollout_memory(feat, mask)
        starts = sample_windows(lengths, batch["exec_start"], 64, generator=g)
        gc, mem, act = gather_windows(feat, action, m, starts, args.horizon,
                                      args.n_obs_steps, lengths)
        tot += float(model.compute_loss(gc, mem, act)) * gc.shape[0]
        n += gc.shape[0]
    model.train()
    return tot / max(n, 1)


if __name__ == "__main__":
    main()
