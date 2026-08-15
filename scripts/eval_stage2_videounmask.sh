#!/usr/bin/env bash
# Sharded stage-2 arm eval for VideoUnmask (arm 2 control / arm 4a concat-TTT).
#
#   bash scripts/eval_stage2_videounmask.sh <run-dir> <arm-ckpt> [split] [episodes] [shards]
#   e.g. bash scripts/eval_stage2_videounmask.sh runs/arm4a_concat 870.ckpt val 50 auto
#
# Sibling of scripts/eval_videounmask.sh (stage-1); same sharding, same VRAM
# preflight, same refusal to report a partial rate. The stage-1 backbone is
# pinned to ckpt_100000.pth because eval_stage2.py's default (ckpt_200000.pth)
# does not exist for this run -- override with STAGE1_CKPT if that changes.
set -euo pipefail

RUN_DIR=${1:?usage: eval_stage2_videounmask.sh <run-dir> <arm-ckpt> [split] [episodes] [shards]}
CKPT=${2:?missing arm checkpoint, e.g. 870.ckpt}
SPLIT=${3:-val}
EPISODES=${4:-50}
SHARDS=${5:-auto}

STAGE1_DIR=${STAGE1_DIR:-runs/dp_stage1}
STAGE1_CKPT=${STAGE1_CKPT:-ckpt_100000.pth}

# Repo-relative so a clone runs anywhere; the pod overlay (network-volume
# caches, wandb key) is applied on top when present.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env-dp.sh"
[ -f /workspace/env-dp.sh ] && source /workspace/env-dp.sh
cd "$REPO_ROOT"

[ -f "$RUN_DIR/checkpoints/$CKPT" ] || { echo "no such arm ckpt: $RUN_DIR/checkpoints/$CKPT" >&2; exit 1; }
[ -f "$STAGE1_DIR/$STAGE1_CKPT" ]  || { echo "no such stage-1 ckpt: $STAGE1_DIR/$STAGE1_CKPT" >&2; exit 1; }

LOGDIR=$RUN_DIR/eval_logs
mkdir -p "$LOGDIR"

RENDER_ARGS=()
case "${VK_ICD_FILENAMES:-}" in
    *lvp_icd.json) RENDER_ARGS=(--render-backend cpu) ;;
esac

# A stage-2 shard carries the frozen stage-1 encoder AND the stage-2 policy AND
# CLIP, so it is heavier than a stage-1 shard (~1.25 GiB there). Budget 1.9 GiB
# and let the preflight refuse rather than discover it inside ClipTextEmbedder.
PER_SHARD_MIB=1900
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$SHARDS" = "auto" ]; then
    SHARDS=$((FREE_MIB / PER_SHARD_MIB))
    [ "$SHARDS" -gt 12 ] && SHARDS=12
    [ "$SHARDS" -lt 1 ] && SHARDS=1
fi
NEED_MIB=$((SHARDS * PER_SHARD_MIB))

echo "==> $RUN_DIR @ $CKPT | stage1=$STAGE1_CKPT | $SPLIT | $EPISODES ep | $SHARDS shards"
echo "    renderer: ${VK_ICD_FILENAMES:-<none>} ${RENDER_ARGS[*]:-(gpu)}"
echo "    vram: need ~${NEED_MIB} MiB, ${FREE_MIB} MiB free"
if [ "$NEED_MIB" -gt "$FREE_MIB" ]; then
    echo "    REFUSING: use at most $((FREE_MIB / PER_SHARD_MIB)) shards." >&2
    exit 1
fi

STEM=${CKPT%.ckpt}
rm -f "$RUN_DIR/eval_${SPLIT}"*"_${STEM}_shard"*.json \
      "$RUN_DIR/eval_VideoUnmask_${SPLIT}_${STEM}_shard"*.json

pids=()
for s in $(seq 0 $((SHARDS - 1))); do
    python dp/stage2/eval_stage2.py \
        --run-dir "$RUN_DIR" --ckpt "$CKPT" \
        --stage1-dir "$STAGE1_DIR" --stage1-ckpt "$STAGE1_CKPT" \
        --task VideoUnmask --split "$SPLIT" --episodes "$EPISODES" \
        --shard "$s" --num-shards "$SHARDS" "${RENDER_ARGS[@]}" \
        > "$LOGDIR/${STEM}_${SPLIT}_shard${s}.log" 2>&1 &
    pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || echo "WARNING: a shard exited non-zero; see $LOGDIR" >&2

python - "$RUN_DIR" "$CKPT" "$SPLIT" "$SHARDS" <<'EOF'
import json, os, sys
run_dir, ckpt, split, shards = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
stem = ckpt.replace(".ckpt", "")
merged, missing = {}, []
for s in range(shards):
    p = os.path.join(run_dir, f"eval_VideoUnmask_{split}_{stem}_shard{s}.json")
    if not os.path.exists(p):
        missing.append(s); continue
    merged.update(json.load(open(p))["results"])
n = len(merged)
if not n:
    print(f"FAILED: every shard of {ckpt} produced nothing — check the logs.",
          file=sys.stderr)
    sys.exit(1)
if missing:
    print(f"FAILED: shards {missing} of {shards} produced nothing; refusing to "
          f"write a partial rate over {n} episodes.", file=sys.stderr)
    sys.exit(1)
succ = sum(v == "success" for v in merged.values())
by_outcome = {}
for v in merged.values():
    by_outcome[v] = by_outcome.get(v, 0) + 1
out = os.path.join(run_dir, f"eval_VideoUnmask_{split}_{stem}.json")
json.dump({"arm": os.path.basename(run_dir.rstrip("/")), "ckpt": ckpt,
           "split": split, "n": n, "success": succ, "rate": succ / n,
           "by_outcome": by_outcome, "results": merged},
          open(out, "w"), indent=1)
print(f"{os.path.basename(run_dir.rstrip('/'))} {ckpt} {split}: "
      f"{succ}/{n} = {succ/n:.1%}")
print("outcomes:", by_outcome)
print("wrote", out)
EOF
