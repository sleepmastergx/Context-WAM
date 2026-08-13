#!/bin/bash
# context-wam TTT arm (sliding w=8 memory). Same knobs as launch_control.sh.
# Run the CPU gates first: checks/check_sliding_chain.py, check_arms_match.py,
# check_write_once.py.
set -euo pipefail
cd "$(dirname "$0")/.."     # repo root: DS json + ./checkpoints resolve from cwd
: "${CACHE_DIR:?set CACHE_DIR to the downloaded movecube cache}"
NGPU=${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}
export PYTHONNOUSERSITE=1
accelerate launch --config_file configs/accelerate_zero1_ds.yaml \
    --num_processes "$NGPU" --main_process_port "${PORT:-29542}" \
    train.py --arm ttt --cache "$CACHE_DIR" \
    --out "${OUT_DIR:-${OUT_ROOT:-runs}/fwam_ttt}" "$@"
