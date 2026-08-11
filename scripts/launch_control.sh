#!/bin/bash
# Baseline fastwam_m5 (no memory). Env knobs:
#   CACHE_DIR  (required)  downloaded window cache (scripts/download_data.py)
#   OUT_DIR    (default runs/fwam_control)
#   NGPU       (default: all visible GPUs)
#   PORT       (default 29541)
# NEVER set CUDA_VISIBLE_DEVICES to raw indices inside a batch job — let the
# scheduler export the allocation.
set -euo pipefail
cd "$(dirname "$0")/.."     # repo root: DS json + ./checkpoints resolve from cwd
: "${CACHE_DIR:?set CACHE_DIR to the downloaded movecube cache}"
NGPU=${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}
export PYTHONNOUSERSITE=1
accelerate launch --config_file configs/accelerate_zero1_ds.yaml \
    --num_processes "$NGPU" --main_process_port "${PORT:-29541}" \
    train.py --arm control --cache "$CACHE_DIR" \
    --out "${OUT_DIR:-runs/fwam_control}" "$@"
