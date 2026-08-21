#!/bin/bash
# Original equal-depth Fast-WAM on the 500-episode MoveCube cache.
# Tuned for two NVLink-connected H100 80GB GPUs with native PyTorch DDP and
# ZeroRedundancyOptimizer (optimizer-state sharding without DeepSpeed).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/workspace/venv/bin/python}
CACHE_DIR=${CACHE_DIR:-/workspace/datasets/movecube-fastwam-cache}
OUT_DIR=${OUT_DIR:-/workspace/outputs/fastwam_original_30x30_ddp}
CONFIG=${CONFIG:-configs/train_movecube_original_30x30.yaml}
NGPU=${NGPU:-2}

test -x "$PYTHON" || {
    echo "missing training environment: $PYTHON (run SKIP_DATA=1 bash setup.sh)"
    exit 1
}
test -f "$CACHE_DIR/text_context.pt" || {
    echo "incomplete cache: $CACHE_DIR"
    exit 1
}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/workspace/checkpoints/}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/workspace/hf_cache/modelscope}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export OUT_ROOT=${OUT_ROOT:-/workspace/outputs}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/workspace/.cache/pip}
export TORCH_HOME=${TORCH_HOME:-/workspace/.cache/torch}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/workspace/.cache/triton}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

set +e
"$PYTHON" -m accelerate.commands.launch \
    --config_file configs/accelerate_ddp_h100.yaml \
    --num_processes "$NGPU" --main_process_port "${PORT:-29543}" \
    train.py --arm original \
    --config "$CONFIG" \
    --cache "$CACHE_DIR" \
    --out "$OUT_DIR" "$@"
train_status=$?
set -e

if [ "$train_status" -ne 0 ]; then
    echo "training failed with exit code $train_status; skipping HF upload"
    exit "$train_status"
fi

if [ "${HF_UPLOAD_ENABLED:-0}" = "1" ]; then
    HF_UPLOAD_REPO=${HF_UPLOAD_REPO:-SleepMastger/movecube-fastwam-original-30x30}
    for attempt in 1 2 3; do
        echo "HF upload attempt $attempt/3 -> $HF_UPLOAD_REPO"
        if "$PYTHON" scripts/upload_training_run.py \
            --repo "$HF_UPLOAD_REPO" --run-dir "$OUT_DIR"; then
            exit 0
        fi
        if [ "$attempt" -lt 3 ]; then sleep 300; fi
    done
    echo "training completed, but HF upload failed after 3 attempts"
    exit 1
fi
