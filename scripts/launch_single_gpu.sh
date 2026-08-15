#!/bin/bash
# Single-GPU launcher — NO DeepSpeed. Use this on a one-GPU box; use
# launch_control.sh / launch_ttt.sh (ZeRO-1) only for multi-GPU.
#
#   ARM        control | ttt          (required)
#   CACHE_DIR  window cache           (required)
#   CONFIG     train config           (default configs/train_videounmask.yaml)
#   OUT_DIR    default $OUT_ROOT/fwam_$ARM
#
# WHY NOT DEEPSPEED HERE (measured 2026-08-14, B200, VideoUnmask):
#   ZeRO-1 shards optimizer state ACROSS RANKS. With num_processes=1 it shards
#   against nobody, yet still pays fp32 master + fp32 grad copies (~64 GiB) on
#   top of the bf16 model — which forced batch 32 down to 8 to fit.
#   Worse, it is WRONG here: on the step-3 gradient spike this task produces,
#   DeepSpeed reported global grad norm 87.6 and let
#   video_expert.patch_embedding.weight overflow to inf DESPITE
#   gradient_clipping: 1.0, NaN-ing every subsequent step. Plain torch clips the
#   same spike (norm 66.5) and recovers: loss 2.09 -> 0.83 over 20 steps, zero
#   non-finite params. train.py already supports this path — with no deepspeed
#   plugin it clips grads itself (the `if dsp is None` branch).
set -euo pipefail
cd "$(dirname "$0")/.."     # repo root: ./checkpoints resolves from cwd
: "${ARM:?set ARM to control or ttt}"
: "${CACHE_DIR:?set CACHE_DIR to the window cache}"
export PYTHONNOUSERSITE=1
# fragmentation headroom; the fused_mlp activations are large and bursty
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

exec python train.py \
    --arm "$ARM" \
    --cache "$CACHE_DIR" \
    --config "${CONFIG:-configs/train_videounmask.yaml}" \
    --out "${OUT_DIR:-${OUT_ROOT:-runs}/fwam_$ARM}" "$@"
