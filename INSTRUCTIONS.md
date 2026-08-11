# Instructions — running the MoveCube study end to end

Two training runs (control and ttt), identical in everything except the
memory. Follow the steps in order; the CPU gates exist so that no GPU hour is
spent on a silently-broken setup. Design rationale: `docs/DESIGN.md`.

## 1. Prerequisites

- Linux node(s) with NVIDIA GPUs. Sized for **140-GiB-class** cards (H200/H100
  NVL): params (bf16) + ZeRO-1 optimizer shard + fp32 EMA shadows + 3.7 GiB
  cache + activations. On ~80 GiB cards: set `ema.enabled: false` in
  `configs/train_movecube.yaml` and reduce `batch_size`.
- Python 3.10. CUDA build matching torch 2.7.1 (developed on cu128).
- A HuggingFace token that can read the private dataset
  `SleepMastger/movecube-fastwam-cache` (and download Wan-AI weights).
- Outbound network on the login/compute node for the first weight download —
  or see the offline note in step 4.

## 2. Environment

```bash
git clone git@github.com:sleepmastergx/Context-WAM.git && cd Context-WAM
conda create -n context_wam python=3.10 -y && conda activate context_wam
# torch FIRST, matched to the cluster's CUDA (see pytorch.org):
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e third_party/fastwam        # vendored, MIT, pinned @7ca5e2f
export PYTHONNOUSERSITE=1                 # ALWAYS. A user-site huggingface-hub
                                          # >=1.0 breaks transformers 4.49
                                          # ("requires >=0.26,<1.0") and the
                                          # failure is an unimportable VAE
                                          # loader, not a clear version error.
python -c "import fastwam, torch, accelerate, deepspeed; print('imports OK')"
```

## 3. Data (3.7 GiB)

Dataset repo: <https://huggingface.co/datasets/SleepMastger/movecube-fastwam-cache>
(**private** — your HF token must have read access; the code repo is public
but the data stays gated).

```bash
huggingface-cli login                      # token with read access
python scripts/download_data.py --out data/movecube_fastwam
```

Expect ~107 files: `ep0000.npz … ep0099.npz` (per-window Wan2.2 VAE latents
`[48,3,16,32]` bf16, actions, states, window starts at stride 1, exec_start),
`meta_shard*.json`, `text_context.pt` (precomputed T5 goal embedding — the
model never builds T5), plus the dataset card. Episodes 0–89 train, 90–99 are
held out (`train_episodes` in the config).

## 4. Model weights

On the first real build, Fast-WAM's loader downloads Wan-AI/Wan2.2-TI2V-5B and
the converted VAE (`DiffSynth-Studio/Wan-Series-Converted-Safetensors/
Wan2.2_VAE.safetensors`) into **`./checkpoints/` resolved from the cwd**.
Therefore: always launch from the repo root (every script in `scripts/` does
this via `cd`). Offline clusters: pre-populate `checkpoints/` by copying it
from a machine that has run once, keeping the directory layout.

## 5. Gates — CPU, seconds, run before ANY GPU time

```bash
python checks/check_sliding_chain.py
python checks/check_arms_match.py
python checks/check_write_once.py
```

Expected (abridged):

```
1/4 bookkeeping: strictly-before convention holds (43->2, 47->2, 49->3)
2/4 streaming == training chain (..., max |diff| < 1e-5)
3/4 gradient reaches cell params AND the learned init; ...
4/4 checkpoint_every=3 chain == plain chain (readouts and grads)
PASS  arms differ only in: memory.enabled
PASS  write-once verified: 20 reads == 0 reads == 1 read; ...
```

Gate 2 matters most: a streaming/training mismatch is the silent train/deploy
operator bug that once cost a 92% discrepancy. If ANY gate fails, stop.

## 6. Loop smoke — CPU, no weights, ~30 s

```bash
python train.py --arm control --synthetic --steps 3
python train.py --arm ttt     --synthetic --steps 3
```

The ttt smoke must print `gate`, `mem_gnorm`, `chain_J`, `chain_E`,
`surprise` in its log line. This verifies the loop mechanics only — it says
nothing about the real model.

## 7. GPU smoke, then the real runs

```bash
# smoke: ~200 steps each; confirms weights, VRAM headroom, throughput
CACHE_DIR=data/movecube_fastwam bash scripts/launch_control.sh --steps 200
CACHE_DIR=data/movecube_fastwam bash scripts/launch_ttt.sh     --steps 200

# full runs — SAME config both arms (the seed lives in the config; do not
# vary it between arms). Submit as INDEPENDENT jobs; do not chain with
# --dependency.
CACHE_DIR=data/movecube_fastwam bash scripts/launch_control.sh
CACHE_DIR=data/movecube_fastwam bash scripts/launch_ttt.sh

# Slurm:
sbatch --export=ALL,ARM=control,CACHE_DIR=$PWD/data/movecube_fastwam scripts/slurm_example.sbatch
sbatch --export=ALL,ARM=ttt,CACHE_DIR=$PWD/data/movecube_fastwam scripts/slurm_example.sbatch
```

Full run length: ~12k optimizer steps (≈13k exec windows × 30 epochs ÷ batch
32). Use the logged `steps_per_s` from the smoke to budget wall-clock.

**Never set `CUDA_VISIBLE_DEVICES` to raw indices inside a batch script** —
the scheduler exports the allocation; raw indices can land on another user's
GPUs.

## 8. What to watch (`runs/*/log.jsonl`)

| field | meaning | healthy |
|---|---|---|
| `loss` | joint video+action flow-matching loss | decreasing |
| `gate` | mean \|tanh α\| across the 5 layers | starts 0.001; must GROW for the memory to matter — flat-at-zero to the end means the memory never became useful |
| `surprise` | TTT inner-loss magnitude at the writes | finite, not exploding |
| `chain_J` | writes in the longest chain this step | ≤ ~62 |
| `mem_gnorm` | grad norm through the chain (BPTT health) | finite, no upward drift |
| `vram_gib` | peak allocated | headroom vs the card |

If `mem_gnorm` spikes or NaNs: first `sliding_w: 16` (halves BPTT depth to the
validated regime, costs 8 frames of read lag), else `chain_checkpoint_every:
8` (VRAM only, exact gradients). **Never detach the chain** — that silently
changes the objective and the memory stops learning what to store
(`docs/DESIGN.md` §in-graph).

## 9. Outputs

`runs/<arm>/checkpoints/step_*.pt` (~10 GiB each, `save_every: 500`):

```python
{"step": int, "cfg": dict,
 "model":  state_dict,          # bf16 5B
 "memory": state_dict | None,   # fp32 2.25M (ttt arm)
 "ema":    {name: fp32 tensor}} # evaluate THESE weights, not the raw ones
```

## 10. Troubleshooting

- `TypeError ... unexpected keyword 'memory'` at build → the model config
  reached their factory unfiltered; use `train.py` (it filters), don't call
  `create_fastwam_decoupled` directly.
- VAE loader unimportable / hub version errors → user site leaked in; check
  `PYTHONNOUSERSITE=1`.
- `vae_config.path is None` → launched from the wrong cwd; run from repo root.
- OOM → in order: `chain_checkpoint_every: 8` (ttt), smaller `batch_size`,
  `ema.enabled: false`.
- DeepSpeed complains about `auto` batch fields → launched without
  `configs/accelerate_zero1_ds.yaml`; use the launch scripts.

## 11. After training (not in this repo yet)

Closed-loop RoboMME eval is the next build: 50 val episodes (eps 90–99),
paired McNemar between arms, EMA weights, and the deploy loop must reproduce
the write cadence exactly (one write per 8 env steps from the trailing
33-step window; reads pure during denoising) — port the streaming-equivalence
gate before trusting any number. To resolve a ~10-pt effect, plan 150+
episodes or 3 seeds per arm.
