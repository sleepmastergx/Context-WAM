#!/usr/bin/env bash
# Checkpoint-selection sweep: run several checkpoints through
# scripts/eval_videounmask.sh, at most $CONCURRENCY of them at once.
#
#   EPISODES=20 SHARDS=8 CONCURRENCY=2 \
#     bash scripts/eval_sweep_videounmask.sh ckpt_10000.pth ckpt_20000.pth ...
#
# Concurrency is bounded by two things, neither of them the GPU's compute:
#   * VRAM  -- each shard holds a DP-UNet + CLIP, ~510 MiB measured, so
#              CONCURRENCY*SHARDS*0.5 GiB must fit alongside anything else
#              training on the card.
#   * cores -- on the lavapipe (CPU rendering) fallback each shard is
#              essentially one busy core.
# Defaults here assume the ~24-shard ceiling that leaves an L4 and a 48-core box
# usable by a concurrent training run.
set -uo pipefail

EPISODES=${EPISODES:-20}
SHARDS=${SHARDS:-auto}      # auto = size to free VRAM at each checkpoint's turn
CONCURRENCY=${CONCURRENCY:-1}
SPLIT=${SPLIT:-val}

[ $# -gt 0 ] || { echo "usage: eval_sweep_videounmask.sh <ckpt.pth> ..." >&2; exit 1; }

cd "$(dirname "$0")/.."
LOGDIR=runs/dp_stage1/eval_logs
mkdir -p "$LOGDIR"

echo "sweep: $# checkpoints | $EPISODES episodes | $SHARDS shards each"
echo "       up to $CONCURRENCY checkpoints at a time"

for ckpt in "$@"; do
    while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do wait -n; done
    echo "[$(date -u +%H:%M:%S)] start $ckpt"
    # Never let one bad checkpoint take the sweep down silently: the runner
    # exits non-zero on OOM'd or missing shards and writes no result file, so
    # just say so and carry on to the next one.
    { bash scripts/eval_videounmask.sh "$ckpt" "$SPLIT" "$EPISODES" "$SHARDS" \
        > "$LOGDIR/sweep_${ckpt%.pth}_${SPLIT}.log" 2>&1 \
      || echo "[$(date -u +%H:%M:%S)] FAILED $ckpt -- see $LOGDIR/sweep_${ckpt%.pth}_${SPLIT}.log"; } &
done
wait

echo
echo "=== sweep results ($SPLIT, $EPISODES episodes) ==="
python - "$SPLIT" <<'EOF'
import glob, json, os, re, sys
split = sys.argv[1]
rows = []
for p in glob.glob(f"runs/dp_stage1/eval_VideoUnmask_{split}_ckpt_*.json"):
    if re.search(r"_shard\d+\.json$", p):
        continue
    d = json.load(open(p))
    step = int(re.search(r"ckpt_(\d+)", d["ckpt"]).group(1))
    rows.append((step, d["n"], d["success"], d.get("rate")))
for step, n, succ, rate in sorted(rows):
    print(f"  ckpt_{step:<7d} {succ:3d}/{n:<3d} = {rate:.1%}" if rate is not None
          else f"  ckpt_{step:<7d} no episodes")
if rows:
    best = max(rows, key=lambda r: (r[3] or 0, r[0]))
    print(f"\nbest on {split}: ckpt_{best[0]} at {best[3]:.1%} "
          f"({best[2]}/{best[1]})")
    print("Re-run the winner at the full 50 episodes before quoting it.")
EOF
