# RunPod: persistent volume + prebuilt image

The setup this describes splits the world in two:

| lives in the **Docker image** | lives on the **Network Volume** (`/workspace`) |
|---|---|
| OS, CUDA toolchain, Python 3.12 | source (`/workspace/projects/Context-WAM`) |
| torch 2.7.1+cu128, DeepSpeed, all pip deps | dataset (`/workspace/data`) |
| the env-var layout below | model weights (`/workspace/checkpoints`) |
| — nothing else | runs/logs (`/workspace/outputs`), HF cache (`/workspace/hf_cache`) |

Nothing is duplicated between the two, so a new pod never reinstalls a
dependency and never re-downloads a byte of data.

## Starting a new pod

1. Attach the **same** Network Volume at `/workspace`.
2. Launch with image `ghcr.io/sleepmastergx/context-wam:cu128-torch271`.
   The GHCR package is **private**, so add a RunPod container-registry
   credential once (Settings -> Container Registry Auth): username
   `sleepmastergx`, password = a GitHub PAT with `read:packages`. RunPod stores
   it per-account, so later pods just pick the same credential.
3. Source refresh:
   ```bash
   cd /workspace/projects/Context-WAM && git pull
   # first time on a brand-new volume:
   # git clone https://github.com/sleepmastergx/Context-WAM.git /workspace/projects/Context-WAM
   ```
4. Train:
   ```bash
   python train.py --arm control        # or --arm ttt
   # multi-GPU: bash scripts/launch_control.sh
   ```

No `setup.sh`, no `pip install`, no `source env.sh` — every path below is
already exported by the image, and `train.py` fills the same values in itself
if the environment is missing them (a pod template's env vars override the
image's `ENV`, and a bare ssh shell may carry neither). It only does that when
the directory actually exists, so a non-RunPod machine behaves exactly as
before. An explicitly exported value always wins.

## The paths (image `ENV`, so cwd-independent)

| variable | value | what it holds |
|---|---|---|
| `HF_HOME` | `/workspace/hf_cache` | HuggingFace cache **and** the auth token |
| `MODELSCOPE_CACHE` | `/workspace/hf_cache/modelscope` | ModelScope cache |
| `DIFFSYNTH_MODEL_BASE_PATH` | `/workspace/checkpoints/` | Wan weights |
| `CACHE_DIR` | `/workspace/data/movecube_fastwam` | window cache (`train.py --cache` default) |
| `OUT_ROOT` | `/workspace/outputs` | run dirs (`train.py --out` default) |
| `PIP_CACHE_DIR`, `TORCH_HOME`, `TRITON_CACHE_DIR` | `/workspace/.cache/*` | keeps caches off the container disk |

`DIFFSYNTH_MODEL_BASE_PATH` is the important one: Fast-WAM's loader otherwise
resolves weights to `./checkpoints` **relative to the cwd**, which is the
`vae_config.path is None` failure in INSTRUCTIONS.md §10. With it set, the
weights resolve the same from any directory and the repo-root `checkpoints`
symlink is no longer load-bearing.

## What is already on the volume

```
/workspace/
├── projects/Context-WAM/          # the clone
├── data/movecube_fastwam/         # 3.7 GiB, 106 files (ep0000-0099.npz,
│                                  #   meta_shard0-3.json, text_context.pt)
├── checkpoints/
│   └── DiffSynth-Studio/Wan-Series-Converted-Safetensors/
│       └── Wan2.2_VAE.safetensors # 1.4 GiB
├── hf_cache/                      # HF_HOME, incl. `token`
├── outputs/                       # run dirs + setup/smoke logs
├── configs/                       # single-GPU smoke configs (see below)
└── bin/                           # image build tooling
```

### Why only the VAE

`configs/model/fastwam_ttt_m5*.yaml` sets `skip_dit_load_from_pretrain: true`
and `load_text_encoder: false`, so `load_wan22_ti2v_5b_components` downloads
**only** the VAE. The video DiT is constructed from config and randomly
initialised; T5 never loads (the goal embedding is precomputed in
`text_context.pt`). The 10 GiB `Wan-AI/Wan2.2-TI2V-5B` DiT weights are
therefore *not* needed and are not on the volume. Flip
`skip_dit_load_from_pretrain: false` and the loader will fetch them into
`/workspace/checkpoints` on first run.

Note the VAE comes from **ModelScope**, not HuggingFace —
`DiffSynth-Studio/Wan-Series-Converted-Safetensors` does not exist on the Hub
(404), which is why `ModelConfig.parse_download_source()` defaults to
`modelscope`. Do not set `DIFFSYNTH_DOWNLOAD_SOURCE=huggingface`.

## Single-GPU / small-card runs

The repo is sized for 140-GiB cards. On smaller cards
(`/workspace/configs/`, used for the smoke test on a 32 GiB RTX 5090):

- `train_movecube_smoke_1gpu.yaml` — `ema.enabled: false` (fp32 shadows are
  ~20 GiB), `batch_size: 1`, `train_episodes: 8` (shrinks the VRAM-resident
  window cache).
- `ds_zero1_offload.json` — ZeRO-1 with `offload_optimizer: cpu`. With one
  GPU there is nothing to shard the AdamW state across, and 5B params of fp32
  master+moments is ~60 GiB; offloading it to host RAM is what makes a
  single-card step possible at all.
- `accelerate_zero1_offload.yaml` — points accelerate at that JSON.

```bash
accelerate launch --config_file /workspace/configs/accelerate_zero1_offload.yaml \
    --num_processes 1 train.py --arm ttt \
    --config /workspace/configs/train_movecube_smoke_1gpu.yaml --steps 2
```

These are **smoke** settings, not a research configuration: `train_episodes: 8`
and `batch_size: 1` change the sampling distribution, and with `ema.enabled:
false` there are no EMA weights to evaluate. Real runs use
`configs/train_movecube.yaml` unchanged, on 140-GiB-class cards.

## Findings from the first real-model run (2026-08-13)

Three things only appear once a **real** model is built — `setup.sh`'s gates and
the `--synthetic` smoke pass without them, because the synthetic path uses a
stub model and never touches Fast-WAM.

**1. Missing dependency: `boto3`** (fixed, now in `requirements.txt`).
`fastwam/utils/misc.py` imports it at module level, so `fastwam.runtime` — and
therefore every real build — was unimportable. Also added: `ftfy` (the text
encoder silently degrades `fix_text` to a no-op without it) and `hf_transfer`
(RunPod exports `HF_HUB_ENABLE_HF_TRANSFER=1`, and `huggingface_hub` hard-errors
at download time when the package is absent).

**2. `proprio` shape** (fixed in `context_wam/gpu_cache.py`).
`batch()` returned `[B, P]`; `fastwam.build_inputs` requires `[B, T, d]` and
then takes `proprio[:, 0, :]`. One state per window is `T=1`, so `batch()` now
returns `[B, 1, P]` — a reshape, not a change of meaning. `SlidingChain`
indexes `cache.states` directly and is unaffected.
`SyntheticWindowCache.batch()` in `train.py` was aligned to match, so the two
duck-typed caches cannot drift again.

**3. The ttt read seam is dimensionally wrong — NOT fixed, needs your call.**

```
RuntimeError: The size of tensor a (3072) must match the size of tensor b (1024)
  at non-singleton dimension 2      # per_layer_memory.py:149
```

`patch_mot_action_expert` adds `read_layer(...)` to `mixed_attn_out`, but
`MoT._apply_expert_post_block` computes
`block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))` — so
`mixed_attn_out` is **pre-projection**, width `num_heads * attn_head_dim` =
24×128 = **3072**. The memory emits `hidden_dim` = **1024**, which is what
`read_layer`'s docstring and the config's `hidden_dim: 1024  # MUST equal
action_dit_config.hidden_dim` both call for. For the video expert these
coincide (3072 = 24×128), which is why the mismatch is invisible until the
5-layer action expert runs.

So the width is right and the **insertion point** is wrong. Three ways out,
in increasing distance from the stated intent:

- **(C, recommended)** add after the `o` projection but still *inside* the
  `gate_msa` residual — i.e. wrap the action block's `self_attn.o` so the
  memory lands on its 1024-wide output. This is literally "a gated per-layer
  contribution to the action expert's attention output", at `hidden_dim`.
- **(B)** add to the block's final output (after gate, cross-attn and FFN).
  Simplest, but the contribution then bypasses `gate_msa` and the FFN.
- **(A)** widen the memory to `num_heads*attn_head_dim` (3072) and keep adding
  to `mixed_attn_out`. Contradicts the `hidden_dim` invariant in the config.

Verified with option (B) applied as a **runtime monkeypatch only**
(`/workspace/bin/diag_ttt_seam.py` — the repo is untouched) that this is the
*only* blocker: the ttt arm then builds 2.2495M memory params, forward gives
`loss=1.0524` with `chain_J=36, surprise=2.99`, backward gives
`memory grad norm = 0.0197` (gradients do reach the memory through the chain)
and `gate |tanh(alpha)| = 0.001` as designed. Pick a seam and the arm runs.

Note `checks/check_sliding_chain.py` and `check_write_once.py` still pass —
they exercise the chain and the write-once discipline against a stub, never
against the real MoT block. A gate that builds the real action expert once and
asserts the seam's shapes would have caught this.

## Rebuilding the image

`Dockerfile` at the repo root is the source of truth:

```bash
docker build -t ghcr.io/sleepmastergx/context-wam:cu128-torch271 .
docker push  ghcr.io/sleepmastergx/context-wam:cu128-torch271
```

The image deliberately contains no source. `third_party/fastwam` is vendored
in the repo, so instead of `pip install -e`, the image drops a `.pth` naming
`/workspace/projects/Context-WAM/third_party/fastwam/src`. Clone somewhere
else and `import fastwam` stops resolving — export
`FASTWAM_SRC=<clone>/third_party/fastwam/src` (`context_wam/build_model.py`
reads it) or run `pip install -e third_party/fastwam --no-deps` once.
