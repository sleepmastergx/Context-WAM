#!/bin/bash
# M5 Fast-WAM (ttt or control arm) on the 490-episode MoveCube cache.
# Usage: bash scripts/launch_m5_490ep.sh {ttt|control} [extra train.py args]
# Same environment/recipe as launch_original_30x30.sh (native DDP + ZeRO).
set -euo pipefail
cd "$(dirname "$0")/.."

ARM=${1:-}; shift || true
case "$ARM" in
  ttt|control) ;;
  *) echo "usage: launch_m5_490ep.sh {ttt|control} [extra train.py args]" >&2; exit 2 ;;
esac

PYTHON=${PYTHON:-/workspace/venv/bin/python}
CACHE_DIR=${CACHE_DIR:-/workspace/datasets/movecube-fastwam-cache}
OUT_DIR=${OUT_DIR:-/workspace/outputs/fastwam_m5_${ARM}_490ep}
CONFIG=${CONFIG:-configs/train_movecube_m5_490ep.yaml}
NGPU=${NGPU:-2}

test -x "$PYTHON" || { echo "missing training environment: $PYTHON"; exit 1; }
test -f "$CACHE_DIR/text_context.pt" || { echo "incomplete cache: $CACHE_DIR"; exit 1; }
"$PYTHON" checks/check_arms_match.py || exit 1

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/workspace/checkpoints/}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/workspace/hf_cache/modelscope}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export OUT_ROOT=${OUT_ROOT:-/workspace/outputs}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
[ -f /workspace/.wandb_key ] && export WANDB_API_KEY=${WANDB_API_KEY:-$(cat /workspace/.wandb_key)}

"$PYTHON" -m accelerate.commands.launch \
    --config_file configs/accelerate_ddp_h100.yaml \
    --num_processes "$NGPU" --main_process_port "${PORT:-29544}" \
    train.py --arm "$ARM" \
    --config "$CONFIG" \
    --cache "$CACHE_DIR" \
    --out "$OUT_DIR" "$@"
