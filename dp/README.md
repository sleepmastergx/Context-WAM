# DP pipeline — stage-1 diffusion policy + stage-2 TTT arms

The predecessor study to the Fast-WAM arms, ported for a new task. Three runs
per task, in order:

1. **stage-1**: DP-UNet (RoboMME's own baseline recipe, 200k steps) — trains
   the vision encoder and sets the reference number,
2. **stage-2 arm 2**: frozen stage-1 encoder, fresh UNet head, **no memory**,
3. **stage-2 arm 4a**: same + a TTT memory (concat-inject, chunk=16) whose
   rollout covers the whole episode including the conditioning video.

MoveCube reference results (50 val episodes, this exact pipeline): stage-1 DP
24% @200k (val sweep 0/10/16/22/24 at 20k..200k); arm 2 = 8.0%, arm 4a =
18.0% — +10 pts, McNemar 7W/2L, p=0.18 (suggestive at n=50; plan 150+
episodes or 3 seeds to resolve a 10-pt effect).

Target task here: **VideoUnmask** (DP-UNet leaderboard 10%, MemER 80%,
leak-probe CLEAN; ~240-frame episodes; per-episode goals name the QUERY —
"...the container hiding the **blue** cube").

## Environment

```bash
bash dp/install_env.sh          # venv + pinned clones of the two PUBLIC deps:
                                #   RoboMME/robomme_benchmark @ f9015a5  (sim env)
                                #   RoboMME/DP               @ f333cd6  (DP framework)
export DP_REPO=$PWD/external/DP
export PATH=$PWD/.venv-dp/bin:$PATH
```

This env is separate from the Fast-WAM one (torch 2.9.1 vs 2.7.1). Sim eval
needs a GPU node with Vulkan.

**Gate the renderer before spending time on a new pod:**

```bash
python checks/check_vulkan.py     # 0 = GPU, 1 = CPU-only (lavapipe), 2 = none
```

`NVIDIA_DRIVER_CAPABILITIES=all` is necessary but not sufficient. The NVIDIA
Vulkan ICD must also open a DRM render node, and some RunPod pods bind-mount
`/dev/dri/renderD*` from the host owned by a uid outside the container's user
namespace — root then gets EACCES (`cap_dac_override` does not cross a userns
boundary) and sapien reports the misleading "failed to find a rendering
device". Nothing inside the pod fixes it: `mknod` is barred in a userns, the
nodes are busy bind-mounts, and `unshare -Urm` is denied. Get a pod whose
render node is openable.

Failing that, CPU rendering works — correctly, but ~an order of magnitude
slower:

```bash
apt-get update && apt-get install -y mesa-vulkan-drivers   # lavapipe; ephemeral
# env-dp.sh then auto-selects the lavapipe ICD and says so.
python dp/eval_dp.py ... --render-backend cpu   # ManiSkill still defaults to gpu
```

Measured on an L4 pod sharing the box with two trainings: 8.5 sim steps/s plus
~45 ms/step of policy. A *failed* episode burns the full 1300-step budget
(~13 min); a success ends in ~100 steps (~1 min).

## Data

Raw episodes (pre-precompute, by design):
<https://huggingface.co/datasets/SleepMastger/robomme-videounmask-raw>
(private — token needs read access).

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(
  'SleepMastger/robomme-videounmask-raw', 'record_dataset_VideoUnmask.h5',
  repo_type='dataset', local_dir='data')"
```

## Pipeline

```bash
# 1) PRECOMPUTE: h5 -> LeRobot parquet (CPU-parallel; --verify re-diffs
#    pixels/actions/state against the h5)
python dp/convert_h5_to_lerobot.py --h5 data/record_dataset_VideoUnmask.h5 \
    --out data/robomme_data_lerobot_videounmask --verify

# 2) stage-1 DP (GPU; their exact recipe — deviations print a warning and
#    break comparability). First gate the RAM cache:
python dp/check_cache_equiv.py \
    --dataset-root data/robomme_data_lerobot_videounmask --stats-dir runs/dp_stage1
torchrun --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29513 \
    dp/train_ddp.py --config-name train_dp_unet_obs2 \
    --dataset-root data/robomme_data_lerobot_videounmask --run-dir runs/dp_stage1

# 3) val checkpoint sweep -> pick the winner (repeat per ckpt, shardable)
python dp/eval_dp.py --run-dir runs/dp_stage1 --ckpt ckpt_200000.pth \
    --task VideoUnmask --split val --episodes 50
# or, sharded + merged for you (adds --render-backend cpu when on lavapipe):
bash scripts/eval_videounmask.sh ckpt_200000.pth val 50 8

# 4) stage-2 feature cache. --with-text is REQUIRED for VideoUnmask: the
#    query color lives in the goal string; without it the arms are blind to
#    the query and the comparison is meaningless. (MoveCube: omit the flag.)
python dp/stage2/cache_features.py --run-dir runs/dp_stage1 \
    --ckpt <winner>.pth \
    --data-dir data/robomme_data_lerobot_videounmask/data/chunk-000 \
    --out cache/feats.npz --with-text

# 5) gate the streaming TTT operator, then train BOTH arms (same defaults)
python dp/stage2/check_chunk_equiv.py --features cache/feats.npz
python dp/stage2/train_stage2.py --features cache/feats.npz \
    --stats-path runs/dp_stage1 --out runs/arm2_control --no-ttt
python dp/stage2/train_stage2.py --features cache/feats.npz \
    --stats-path runs/dp_stage1 --out runs/arm4a_concat --ttt

# 6) paired closed-loop eval (sim, GPU+Vulkan; shard with --shard/--num-shards)
python dp/stage2/eval_stage2.py --run-dir runs/arm2_control \
    --stage1-dir runs/dp_stage1 --stage1-ckpt <winner>.pth \
    --task VideoUnmask --split val --episodes 50
python dp/stage2/eval_stage2.py --run-dir runs/arm4a_concat \
    --stage1-dir runs/dp_stage1 --stage1-ckpt <winner>.pth \
    --task VideoUnmask --split val --episodes 50
DP_TASK=VideoUnmask python dp/stage2/sweep_table.py val   # McNemar table
```

## How the query-color conditioning works

- **stage-1**: native — the DP framework CLIP-embeds each episode's goal
  string (`include_text: true` in their config); the converter writes the
  per-episode goals into `meta/episodes.jsonl` / `tasks.jsonl`, and
  `eval_dp.py` re-embeds the env's goal **per episode**.
- **stage-2**: the frozen-feature cache is vision+state (136-d) and carries no
  text, so `cache_features.py --with-text` stores the per-episode CLIP
  embedding and the dataset appends it per frame — layout
  `[vis 128 | state 8 | text 512]`, d_feat 648. The checkpoint records
  `d_feat`; `eval_stage2.py` auto-detects it and rebuilds the same layout at
  reset from the env's goal. Both the head AND the TTT memory see the query.

## Gates (run before GPU time, after any change)

- `check_cache_equiv.py` — RAM-cached windows bit-exact vs the original
  dataset path (boundary clamps included).
- `check_chunk_equiv.py` — streaming eval-time TTT ≡ training rollout
  (chunk=16). This gate caught a real read-ordering bug once; per-frame
  stepping is a different operator that diverged 92% on a prior benchmark.

## Notes

- Episode-sharded eval: one process per GPU via `--shard/--num-shards`; never
  set `CUDA_VISIBLE_DEVICES` to raw indices inside a batch script.
- `train_ddp.py` deviations from the paper recipe (batch/steps/lr) print a
  loud warning — results are then not comparable to the leaderboard DP.
- Never read `info["simple_subgoal*"]/[grounded_subgoal*]` in eval code —
  those fields are the benchmark's oracle plan (a leak).
- Evaluate EMA weights (`ema_model` in stage-2 ckpts; `model_ema` in stage-1).
