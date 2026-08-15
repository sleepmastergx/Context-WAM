"""Multi-GPU DDP training entrypoint for RoboMME's DP (train_dp_unet_obs2).
Current launch config: 2 GPUs (per-rank batch 64, global 128); any world size
that divides 128 works identically.

Wraps the RoboMME/DP repo's own components (RoboMMEDataset, DiffusionPolicyUnet,
InfiniteRandomSamplerDDP) rather than reimplementing them — the reproduction
depends on their exact model/sampling code. Their Trainer is single-GPU only,
so the loop lives here with the DDP mechanics added.

Faithfulness contract (identical optimization to their single-GPU run):
  * GLOBAL batch stays 128 -> per-rank batch = 128 // world_size.
  * num_training_steps forced to 200,000 (paper Table 10). This drives BOTH
    the stop condition and the cosine LR curve; base.yaml's 1e6 is a paper/code
    discrepancy we resolve in the paper's favor.
  * fp32 end to end by default. --bf16 exists but defaults OFF: it changes
    the numerics vs their run. TF32 conv/matmul enabled (H200; measurable
    speedup, negligible numeric drift for ResNet18+UNet — disable with
    --no-tf32 for a bit-faithful check).
  * loss goes through DDP forward() via a shim module: their compute_loss is
    a plain method, and calling methods other than forward() on a DDP wrapper
    skips gradient bucketing entirely (silent no-sync bug).

GPU-throughput choices (do not change math):
  * InfiniteRandomSamplerDDP shards one global permutation per epoch —
    identical index stream to their InfiniteRandomSampler, split over ranks.
  * DataLoader: num_workers default 8/rank, pin_memory, persistent_workers,
    prefetch_factor 4 — PNG decode of 2x 256x256 per sample is the actual
    bottleneck at batch 32/rank, not the model.
  * H2D copies non_blocking on a dedicated CUDA stream via pinned buffers
    (comes free with pin_memory + non_blocking).
  * cudnn.benchmark on (fixed 230x230 shapes after crop).

VERIFY-ON-FIRST-RUN (written without executing; check when env exists):
  * import paths assume RoboMME_DP repo root on sys.path (set REPO below).
  * RoboMMEDataset __init__ kwargs, EMAModel API (diffusers), and the hydra
    config tree names match the repo as fetched 2026-08-06; re-check on clone.

Launch (inside the sbatch, never by hand on a login node):
  deploy -m torch.distributed.run --nproc_per_node=2 \
      --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29513 \
      exp/robomme_dp/train_ddp.py --config-name train_dp_unet_obs2 \
      dataset_root=<converted-parquet-root> run_dir=<out>
"""
import argparse, json, os, shutil, sys, time

# BEFORE importing their modules: robomme_dataset.py setdefaults
# HF_DATASETS_CACHE to the AUTHORS' cluster path — claim standard locations
# first (or export your own before launching).
_HF = os.path.expanduser("~/.cache/huggingface")
os.environ.setdefault("HF_HOME", _HF)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_HF, "datasets"))

REPO = os.environ.get("DP_REPO")
assert REPO, "set DP_REPO to your clone of github.com/RoboMME/DP (pin f333cd6)"
sys.path.insert(0, REPO)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

GLOBAL_BATCH = 128
TOTAL_STEPS = 200_000
CKPT_EVERY = 10_000
LOG_EVERY = 100


def snapshot(sd):
    """Detached CPU copy of a state_dict.

    state_dict() hands back live references to the parameters. The EMA
    copy_to/restore pair below mutates those same tensors in place, so without
    a real copy both checkpoint entries end up aliasing one storage and
    serialize the restored RAW weights -- torch.save dedupes the shared
    storage, so the file size looks correct and nothing errors. to(copy=True)
    (not .cpu(), which is a no-op when already on CPU) also keeps the second
    copy off the GPU.
    """
    return {k: v.detach().to("cpu", copy=True) for k, v in sd.items()}


class LossShim(torch.nn.Module):
    """DDP gradient hooks only fire on forward(); route compute_loss through it."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model.compute_loss(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="train_dp_unet_obs2")
    ap.add_argument("dataset_root_kv", nargs="?", default=None,
                    help="dataset_root=<path> (hydra-style kv, optional)")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-cached-data", action="store_true",
                    help="skip the decode-once RAM frame cache and use the "
                         "original per-access PNG-decode path")
    ap.add_argument("--bf16", action="store_true",
                    help="autocast bf16 (OFF by default: not their numerics)")
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU sanity check: 3 steps, batch 4, no CLIP/text "
                         "(dataset's ClipTextEmbedder is CUDA-hardcoded), "
                         "no DDP, no wandb, ckpt to a throwaway dir")
    # ---- DEVIATION knobs: defaults reproduce their recipe exactly. Changing
    # them makes the result non-comparable to the paper's DP numbers. ----
    ap.add_argument("--global-batch", type=int, default=GLOBAL_BATCH)
    ap.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--lr", type=float, default=None,
                    help="override cfg.optimizer.lr (e.g. when batch-scaling)")
    args = ap.parse_args()
    dataset_root = args.dataset_root or (
        args.dataset_root_kv.split("=", 1)[1] if args.dataset_root_kv else None)
    assert dataset_root, "pass --dataset-root"
    run_dir = args.run_dir or os.path.join("runs", "dp_stage1")

    # ---- DDP init (same pattern as exp/ttt/train_stage2.py) ----
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp = world > 1 and not args.smoke
    if ddp:
        dist.init_process_group("nccl")
    if args.smoke and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    is_main = rank == 0
    # per-rank seed offset: without it every rank samples identical windows
    # and the effective batch never actually grows past 32
    torch.manual_seed(args.seed + rank)

    if not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    per_rank_batch = 4 if args.smoke else args.global_batch // world
    assert args.smoke or per_rank_batch * world == args.global_batch, \
        f"world={world} must divide global batch {args.global_batch}"

    # ---- config via their hydra tree, with the paper-vs-code fixes ----
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    total_steps = 3 if args.smoke else args.total_steps
    overrides = [
        f"training.lr_scheduler.num_training_steps={total_steps}",
        f"seed={args.seed}",
    ]
    if args.lr is not None:
        overrides.append(f"optimizer.lr={args.lr}")
    if not args.smoke and (args.global_batch != GLOBAL_BATCH
                           or args.total_steps != TOTAL_STEPS
                           or args.lr is not None):
        print(f"*** DEVIATION from the paper recipe: global_batch="
              f"{args.global_batch} total_steps={args.total_steps} "
              f"lr={args.lr} — result NOT comparable to DP 8.67+-1.78 ***",
              flush=True)
    if args.smoke:
        overrides += ["dataloader.batch_size=4",
                      "task.dataset.embed_text=false",
                      "model.include_text=false"]
    with initialize_config_dir(config_dir=os.path.join(REPO, "eval_envs", "config"),
                               version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        # their serve/eval path reloads config.yaml from the ckpt dir
        OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))

    # ---- dataset: theirs, pointed at our converted parquet ----
    ds_cfg = OmegaConf.to_container(cfg.task.dataset, resolve=True)
    ds_cfg.pop("_target_", None)
    ds_cfg["dataset_root"] = dataset_root
    ds_cfg["stats_path"] = run_dir       # stats.json cached next to ckpts
    if args.no_cached_data:
        from eval_envs.dataset.robomme_dataset import RoboMMEDataset as DSClass
    else:
        # decode-once RAM cache; bit-exact vs the original path (gated by
        # check_cache_equiv.py), so the run stays comparable to DP 8.67+-1.78
        from cached_dataset import CachedRoboMMEDataset as DSClass
        # SLURM_CPUS_PER_TASK, not os.cpu_count(): the latter sees the whole
        # node (240 cores) and would oversubscribe our allocation
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 16)
        ds_cfg.update(
            cache_workers=4 if args.smoke else max(4, cpus // (2 * world)),
            cache_share=args.num_workers > 0,
            max_episodes=10 if args.smoke else None,
        )
    # rank 0 builds first (computes + writes stats.json once); others then
    # load it — otherwise 4 ranks race on the same stats file
    if ddp and rank != 0:
        dist.barrier()
    dataset = DSClass(**ds_cfg)
    if ddp and rank == 0:
        dist.barrier()
    if is_main:
        print(f"dataset: {len(dataset)} windows (execution frames only)",
              flush=True)

    from eval_envs.utils.dataloader_util import InfiniteRandomSamplerDDP
    sampler = InfiniteRandomSamplerDDP(dataset, rank=rank, world_size=world,
                                       seed=args.seed)
    loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=per_rank_batch,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, prefetch_factor=4,
        drop_last=True,
    )

    # ---- model / optim / sched / EMA: all theirs ----
    model = instantiate(cfg.model).to(device)
    shim = LossShim(model)
    if ddp:
        shim = DDP(shim, device_ids=[local_rank],
                   gradient_as_bucket_view=True)
    raw = shim.module.model if ddp else shim.model   # unwrapped, for EMA/ckpt

    optim = torch.optim.AdamW(
        raw.parameters(), lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.betas), eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay)
    from diffusers.optimization import get_scheduler
    sched = get_scheduler(
        "cosine", optimizer=optim,
        num_warmup_steps=cfg.training.lr_scheduler.num_warmup_steps,
        num_training_steps=total_steps)

    ema = None
    if is_main and cfg.training.get("use_ema", True):
        from diffusers.training_utils import EMAModel
        ema = EMAModel(raw.parameters(), decay=cfg.ema.decay,
                       min_decay=cfg.ema.get("min_decay", 0.0),
                       update_after_step=cfg.ema.get("update_after_step", 0),
                       inv_gamma=cfg.ema.get("inv_gamma", 1.0),
                       power=cfg.ema.get("power", 0.75))

    use_wandb = is_main and not args.smoke and not cfg.wandb.get("disabled", False)
    if use_wandb:
        import wandb
        wandb.init(project="wam-ttt", name=os.path.basename(run_dir),
                   config=OmegaConf.to_container(cfg, resolve=True))

    def to_device(batch):
        return {k: (v.to(device, non_blocking=True)
                    if torch.is_tensor(v) else v) for k, v in batch.items()}

    # ---- loop ----
    shim.train()
    ckpt_every = 2 if args.smoke else CKPT_EVERY
    log_every = 1 if args.smoke else LOG_EVERY
    step, t0, data_t, it = 0, time.time(), 0.0, iter(loader)
    autocast = torch.autocast("cuda", torch.bfloat16, enabled=args.bf16)
    while step < total_steps:
        td = time.time()
        batch = to_device(next(it))
        data_t += time.time() - td

        with autocast:
            out = shim(batch)
        loss = out[0] if isinstance(out, tuple) else out
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        sched.step()
        if ema is not None:
            ema.step(raw.parameters())
        step += 1

        if is_main and step % log_every == 0:
            dt = time.time() - t0
            msg = dict(step=step, loss=float(loss.detach()),
                       lr=sched.get_last_lr()[0], gnorm=float(gnorm),
                       steps_per_s=log_every / dt, data_frac=data_t / dt)
            print(json.dumps(msg), flush=True)
            if use_wandb:
                import wandb
                wandb.log(msg, step=step)
            t0, data_t = time.time(), 0.0

        if is_main and (step % ckpt_every == 0 or step == total_steps):
            ck = {"model": snapshot(raw.state_dict()), "global_step": step}
            if ema is not None:
                # store EMA weights in their ckpt["model"] convention too:
                # their serve loads ckpt["model"], and eval should use EMA
                ema.store(raw.parameters())
                ema.copy_to(raw.parameters())
                ck["model_ema"] = snapshot(raw.state_dict())
                ema.restore(raw.parameters())
                assert any(not torch.equal(ck["model"][k], ck["model_ema"][k])
                           for k in ck["model"]), \
                    "model_ema is identical to model -- the snapshot aliased"
            tmp = os.path.join(run_dir, f".tmp_ckpt_{step}.pth")
            torch.save(ck, tmp)
            shutil.move(tmp, os.path.join(run_dir, f"ckpt_{step}.pth"))
            print(f"saved ckpt_{step}.pth", flush=True)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
