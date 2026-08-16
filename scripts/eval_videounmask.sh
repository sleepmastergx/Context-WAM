#!/usr/bin/env bash
# Sharded stage-1 eval for VideoUnmask.
#
#   bash scripts/eval_videounmask.sh ckpt_100000.pth [split] [episodes] [shards]
#
# Splits the episodes across N processes, waits, then merges the per-shard JSON
# into one success rate. Defaults: val / 50 episodes / 8 shards.
#
# Shard count is about the RENDERER, not the GPU: with the NVIDIA ICD the GPU
# serialises the work and ~4 shards is plenty, but on the lavapipe (CPU)
# fallback each shard is a CPU rasteriser, so more shards is strictly better up
# to nproc. env-dp.sh reports which one is in use when you source it.
set -euo pipefail

CKPT=${1:?usage: eval_videounmask.sh <ckpt.pth> [split] [episodes] [shards]}
SPLIT=${2:-val}
EPISODES=${3:-50}
SHARDS=${4:-8}

# Repo-relative so a clone runs anywhere; the pod overlay (network-volume
# caches, wandb key) is applied on top when present.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env-dp.sh"
[ -f /workspace/env-dp.sh ] && source /workspace/env-dp.sh
cd "$REPO_ROOT"

RUN_DIR=${RUN_DIR:-runs/dp_stage1}
LOGDIR=$RUN_DIR/eval_logs
mkdir -p "$LOGDIR"

[ -f "$RUN_DIR/$CKPT" ] || { echo "no such checkpoint: $RUN_DIR/$CKPT" >&2; exit 1; }

# ManiSkill defaults to render_backend="gpu" and will ask sapien for cuda:0
# regardless of which ICD is loaded, so on the lavapipe fallback it must be told
# explicitly. Key off the ICD env-dp.sh picked rather than making the caller
# remember.
RENDER_ARGS=()
case "${VK_ICD_FILENAMES:-}" in
    *lvp_icd.json) RENDER_ARGS=(--render-backend cpu) ;;
esac

echo "==> $CKPT | split=$SPLIT | episodes=$EPISODES | shards=$SHARDS"
echo "    renderer: ${VK_ICD_FILENAMES:-<none>} ${RENDER_ARGS[*]:-(gpu)}"

# VRAM preflight. Each shard holds a DP-UNet + a CLIP text tower + its own CUDA
# context: ~1.25 GiB measured, NOT the ~0.5 GiB you see if you look before CLIP
# has loaded. Oversubscribing does not degrade gracefully -- shards die at
# ClipTextEmbedder with torch.OutOfMemoryError and the run merges to n=0.
PER_SHARD_MIB=1300
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)

# SHARDS=auto sizes to whatever VRAM is free right now, so a sweep queued behind
# a training run widens by itself once that run exits. Capped at 16: past that
# the lavapipe renderer, not the GPU, is the bottleneck.
if [ "$SHARDS" = "auto" ]; then
    SHARDS=$((FREE_MIB / PER_SHARD_MIB))
    [ "$SHARDS" -gt 16 ] && SHARDS=16
    [ "$SHARDS" -lt 1 ] && SHARDS=1
    echo "    shards: auto -> $SHARDS (${FREE_MIB} MiB free)"
fi

NEED_MIB=$((SHARDS * PER_SHARD_MIB))
echo "    vram: need ~${NEED_MIB} MiB for $SHARDS shards, ${FREE_MIB} MiB free"
if [ "$NEED_MIB" -gt "$FREE_MIB" ]; then
    MAX=$((FREE_MIB / PER_SHARD_MIB))
    echo "    REFUSING: not enough free VRAM. Use at most $MAX shards," >&2
    echo "    or wait for whatever else is on the GPU to finish." >&2
    exit 1
fi

# Drop per-shard JSON from any earlier run of this ckpt+split. Without this, a
# re-run at a different shard count leaves orphan shard files behind and the
# merge below silently folds those stale episodes into the new number.
rm -f "$RUN_DIR/eval_VideoUnmask_${SPLIT}_${CKPT%.pth}_shard"*.json

pids=()
for s in $(seq 0 $((SHARDS - 1))); do
    python dp/eval_dp.py \
        --run-dir "$RUN_DIR" --ckpt "$CKPT" \
        --task VideoUnmask --split "$SPLIT" --episodes "$EPISODES" \
        --shard "$s" --num-shards "$SHARDS" "${RENDER_ARGS[@]}" \
        > "$LOGDIR/${CKPT%.pth}_${SPLIT}_shard${s}.log" 2>&1 &
    pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || echo "WARNING: at least one shard exited non-zero; see $LOGDIR" >&2

python - "$RUN_DIR" "$CKPT" "$SPLIT" "$SHARDS" <<'EOF'
import json, os, sys
run_dir, ckpt, split, shards = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
stem = ckpt.replace(".pth", "")
merged, missing = {}, []
for s in range(shards):
    p = os.path.join(run_dir, f"eval_VideoUnmask_{split}_{stem}_shard{s}.json")
    if not os.path.exists(p):
        missing.append(s); continue
    merged.update(json.load(open(p))["results"])
n = len(merged)
if not n:
    print(f"FAILED: every shard of {ckpt} produced nothing — check the logs "
          f"(a CUDA OOM at ClipTextEmbedder looks exactly like this). "
          f"Writing no result file.", file=sys.stderr)
    sys.exit(1)
if missing:
    # A partial number is worse than none: it reads like a real success rate.
    print(f"FAILED: shards {missing} of {shards} produced nothing for {ckpt}; "
          f"refusing to write a partial rate over {n} episodes.", file=sys.stderr)
    sys.exit(1)
succ = sum(v == "success" for v in merged.values())
by_outcome = {}
for v in merged.values():
    by_outcome[v] = by_outcome.get(v, 0) + 1
out = os.path.join(run_dir, f"eval_VideoUnmask_{split}_{stem}.json")
json.dump({"ckpt": ckpt, "split": split, "n": n, "success": succ,
           "rate": succ / n if n else None, "by_outcome": by_outcome,
           "results": merged}, open(out, "w"), indent=1)
print(f"{ckpt} {split}: {succ}/{n} = {succ/n:.1%}" if n else "no episodes")
print("outcomes:", by_outcome)
print("wrote", out)
EOF
