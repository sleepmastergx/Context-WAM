# Running the VideoUnmask eval on your own machine

Everything below assumes a fresh `git clone` and nothing else. The repo carries
code only — datasets, checkpoints and the two pinned upstream repos are fetched
by the steps here.

## 0. What you need

- Linux, an NVIDIA GPU, and a driver new enough for CUDA 12.8 (torch 2.9.1+cu128)
- ~30 GB disk for checkpoints + dataset
- A Vulkan driver. **Check this first** — it is the single most common reason
  eval refuses to start:

```bash
python checks/check_vulkan.py     # exit 0 = GPU, 1 = CPU-only, 2 = no Vulkan
```

Exit 1 means your GPU's DRM render node isn't usable (in containers, usually
because `/dev/dri/renderD*` is bind-mounted with an owner outside the user
namespace, so even root gets EACCES). Eval still runs on CPU rendering —
correctly, but roughly an order of magnitude slower — via
`--render-backend cpu`, which the runner scripts pass automatically. Exit 2
means install a driver: `apt-get install -y mesa-vulkan-drivers` gets you the
CPU fallback at minimum.

## 1. Environment

```bash
bash dp/install_env.sh     # .venv-dp + pinned clones into external/:
                           #   RoboMME/robomme_benchmark @ f9015a5
                           #   RoboMME/DP               @ f333cd6
source scripts/env-dp.sh   # PATH, caches, DATASET_ROOT, Vulkan ICD selection
```

`scripts/env-dp.sh` derives everything from its own location, so the clone can
live anywhere. It prints a warning if it had to fall back to CPU rendering.

## 2. Data and checkpoints

None of this is in git (`data/`, `runs/`, `cache/` are ignored — they are
gigabytes).

```bash
# LeRobot dataset, already converted (~2.6 GB) -- this is all eval needs
python -c "from huggingface_hub import snapshot_download; snapshot_download(
  'SleepMastger/robomme-videounmask-raw', repo_type='dataset', local_dir='data',
  allow_patterns='robomme_data_lerobot_videounmask/*')"

# only if you need to re-derive it: the 14 GB source h5 lives in the same repo
#   python dp/convert_h5_to_lerobot.py --h5 data/record_dataset_VideoUnmask.h5 \
#       --out data/robomme_data_lerobot_videounmask --verify

# stage-1 DP checkpoints + config.yaml + stats.json -> runs/dp_stage1/
python -c "from huggingface_hub import snapshot_download; snapshot_download(
  'SleepMastger/context-wam-videounmask-checkpoints', local_dir='runs/dp_stage1')"
```

Both HF repos are **private** — your token needs read access
(`huggingface-cli login`, or `HF_TOKEN`).

```bash
# stage-2 arms: every .ckpt for both arms, the training logs, the frozen-feature
# cache, and whatever eval results existed at upload time
python -c "from huggingface_hub import snapshot_download; snapshot_download(
  'SleepMastger/context-wam-videounmask-stage2', local_dir='stage2_dl')"
# then place them where the runners expect:
#   stage2_dl/arm2_control/*.ckpt -> runs/arm2_control/checkpoints/
#   stage2_dl/arm4a_concat/*.ckpt -> runs/arm4a_concat/checkpoints/
#   stage2_dl/cache/feats.npz     -> cache/feats.npz
```

For Fast-WAM (not needed for DP eval) the precomputed latent window cache is
`SleepMastger/context-wam-videounmask-fastwam-cache` — 100 `ep*.npz` shards plus
`text_context.pt`, which `train.py --cache` reads directly.

## 3. Eval

```bash
# stage-1, one checkpoint, sharded + merged
bash scripts/eval_videounmask.sh ckpt_100000.pth val 50 auto

# stage-1, sweep several checkpoints
EPISODES=20 CONCURRENCY=1 bash scripts/eval_sweep_videounmask.sh \
    ckpt_10000.pth ckpt_20000.pth ckpt_30000.pth

# stage-2 arms (needs the arm checkpoints from step 2)
bash scripts/eval_stage2_videounmask.sh runs/arm2_control 870.ckpt val 50 auto
bash scripts/eval_stage2_videounmask.sh runs/arm4a_concat 870.ckpt val 50 auto
DP_TASK=VideoUnmask python dp/stage2/sweep_table.py val   # paired McNemar
```

`SHARDS=auto` sizes the shard count to free VRAM (~1.3 GiB per stage-1 shard,
~1.5 GiB per stage-2 shard, measured). Oversubscribing does not degrade
gracefully: shards die inside `ClipTextEmbedder` with `torch.OutOfMemoryError`
and the run merges to nothing. The runners preflight for this and refuse rather
than start, and they refuse to write a **partial** rate if a shard dies —
a success rate over "however many shards survived" reads exactly like a real
result, which is worse than no result.

`dp/stage2/sweep_table.py` only pairs arms that share a checkpoint *name*; it
silently skips a comparison like arm2@460 vs arm4a@550. Pair those by hand.

## 4. Watching a rollout

```bash
python dp/record_episode.py --run-dir runs/dp_stage1 --ckpt ckpt_100000.pth \
    --episode 32 --split val --want success --attempts 8 --seed 100
```

Writes an mp4 of front|wrist side by side, prefixed with the episode's
conditioning video. Because the diffusion sampling is unseeded during normal
eval, a scored outcome can only be *searched* for, never replayed exactly —
hence `--want` / `--attempts`.

## Known gotchas

- **`eval_dp.py` does not seed torch.** The same checkpoint on the same episode
  can score `success (95 steps)` one run and `timeout (1301 steps)` the next.
  Env init is deterministic from the episode metadata; the policy's denoising
  noise is not. Budget for this when comparing small-n results.
- **`--stage1-ckpt` defaults to `ckpt_200000.pth`,** which does not exist for
  this run — stage-1 stopped at 100k. The stage-2 runner pins
  `ckpt_100000.pth`; override with `STAGE1_CKPT=`.
- **`--with-text` is mandatory for VideoUnmask** at stage 2: the query colour
  lives in the goal string, so an arm without it is blind to the question.
  Trained arms record `d_feat: 648`; 136 means text is missing.
- **A timeout costs the full 1300-step budget** (~13 min on CPU rendering),
  a failure often ends in ~100. Runtime therefore depends heavily on how good
  the policy is.
