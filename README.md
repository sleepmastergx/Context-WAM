# Context-WAM

TTT fast weights as **episodic memory** in a world-action model, evaluated on
RoboMME **MoveCube** (100 episodes: a conditioning video demonstrates one of
three manners — hook / pick-place / push — then the robot must execute it).

Two arms, differing in exactly one config key (`memory.enabled` — gated by
`checks/check_arms_match.py`):

| arm | model | what it is |
|---|---|---|
| `control` | `configs/model/fastwam_ttt_m5_control.yaml` | Fast-WAM, 5-layer action expert (random init), `fused_mlp`, both experts train — no memory |
| `ttt` | `configs/model/fastwam_ttt_m5.yaml` | same + 5 per-layer TTT fast-weight states (2.25M, +1.9%), gate `tanh(α)` α=1e-3 → **identical function at step 0** |

The TTT memory is written from the **video stream only** (never actions), one
write per `w=8` raw steps over the whole episode, chained **in-graph** from a
learned init (`context_wam/sliding_chain.py`). A training window starting at
raw step `t` reads the state after the last completed write strictly before
`t` (`j_max(t) = (t-33)//8 + 1`). Both arms sample the SAME uniform-random
exec windows, so the memory is the only difference.

## Setup (target cluster)

```bash
git clone git@github.com:sleepmastergx/Context-WAM.git && cd Context-WAM
# python 3.10 env. Known-good pins (the env this was developed against):
#   torch 2.7.1+cu128, accelerate 1.12.0, deepspeed 0.18.5,
#   transformers 4.49.x, huggingface-hub 0.36.x (NOT >=1.0), diffusers,
#   omegaconf + hydra-core, numpy, pyyaml
pip install -r requirements.txt
pip install -e third_party/fastwam          # vendored, MIT, pinned @7ca5e2f
export PYTHONNOUSERSITE=1                    # user-site hub versions break transformers
huggingface-cli login                        # data repo + Wan weights are gated/private
python scripts/download_data.py --out data/movecube_fastwam   # 3.7 GiB
```

**Model weights**: on first build, Fast-WAM's loader downloads
Wan-AI/Wan2.2-TI2V-5B (+ the converted VAE
`DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors`)
into `./checkpoints/`, resolved **relative to the cwd** — always launch from
the repo root (the launch scripts do). If the cluster has no egress, copy a
populated `checkpoints/` dir to the repo root instead.

## Run

```bash
# 0) gates — CPU, seconds; run before ANY GPU time
python checks/check_sliding_chain.py     # chain: convention/stream-equiv/grads/ckpt
python checks/check_arms_match.py        # arms differ ONLY in memory.enabled
python checks/check_write_once.py        # write-once vs 20-step-denoise invariance

# 1) loop smoke — CPU, no weights (verifies mechanics, NOT the model)
python train.py --arm control --synthetic --steps 3
python train.py --arm ttt     --synthetic --steps 3

# 2) GPU smoke, then full runs (both arms, SAME seed — it's in the config)
CACHE_DIR=data/movecube_fastwam bash scripts/launch_control.sh --steps 200
CACHE_DIR=data/movecube_fastwam bash scripts/launch_ttt.sh     --steps 200
CACHE_DIR=data/movecube_fastwam bash scripts/launch_control.sh
CACHE_DIR=data/movecube_fastwam bash scripts/launch_ttt.sh
# Slurm: sbatch --export=ALL,ARM=ttt,CACHE_DIR=... scripts/slurm_example.sbatch
```

~5B params train (their recipe: both experts + proprio encoder), so the
launchers use **DeepSpeed ZeRO-1** via accelerate — replicated AdamW state
alone is 56 GiB and will not fit without it. The TTT memory stays **outside**
the engine in fp32 (bf16 quantizes the inner TTT update to noise); `train.py`
optimizes it separately and all-reduces its grads. Checkpoints
(`runs/*/checkpoints/step_*.pt`) contain model + memory + **EMA shadows**
(~10 GiB each; evaluate the EMA weights). VRAM note: EMA shadows are fp32
on-device, ~20 GiB for the 5B — sized for 140-GiB-class GPUs; on ~80 GiB
cards set `ema.enabled: false` (and evaluate raw weights) or shrink the batch.

Watch in `runs/*/log.jsonl` for the ttt arm: `gate` (|tanh α| — must grow off
1e-3 for the memory to matter), `surprise`, `chain_J` (writes per chain, ≤~62),
`mem_gnorm` (chain BPTT health — if it misbehaves, set `sliding_w: 16` or
`chain_checkpoint_every: 8`; **never** detach the chain — that severs the
outer loop and the memory can no longer learn what to store).

## Repo map

```
context_wam/sliding_chain.py   the in-graph w-cadence chain (the new mechanism)
context_wam/per_layer_memory.py  per-layer TTT states + the MoT read seam patch
context_wam/ttt_cell.py, memory.py  TTT cell (Titans gating) + functional rollout
context_wam/build_model.py     assembles Fast-WAM + memory (FASTWAM_SRC env override)
context_wam/gpu_cache.py       VRAM-resident window cache (no dataloader)
context_wam/convert_movecube.py  provenance: raw RoboMME h5 -> this cache
train.py                       both arms; one sampler; accelerate + ZeRO-1
checks/                        run-before-GPU gates
third_party/fastwam/           vendored Fast-WAM (MIT, pinned — PROVENANCE.md)
```

## Not included (next steps)

- **Eval**: closed-loop RoboMME sim eval is not in this repo yet. The deploy
  loop must reproduce the write cadence exactly (one write per 8 env steps
  from the trailing 33-frame window, reads pure during denoising) — port the
  streaming-equivalence gate pattern from `checks/check_sliding_chain.py`
  test 2 before trusting any number. Protocol: 50 val episodes (eps 90-99
  seeds), paired McNemar across arms; plan 150+ episodes or 3 seeds/arm to
  resolve a ~10-pt effect.
- Ablation arms: frozen-M* (writes stop at the video→exec boundary) and
  write-everywhere — both are small flags on the sliding design.

## Design record

Decisions and their reasons (write-gating, the strictly-before convention,
w=8 realization on the Wan VAE grid, why the chain must stay in-graph) live in
the project's design-review artifacts; the short versions are in the module
docstrings of `sliding_chain.py` and `train.py`.
