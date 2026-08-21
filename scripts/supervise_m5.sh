#!/bin/bash
# Keep the Fast-WAM M5 490-episode MoveCube run alive until it finishes.
# Generalisation of supervise_original_30x30.sh for the m5 arms:
#   * if training is not running, resume it from the newest complete checkpoint
#     (local first, otherwise the newest one on the HF repo is downloaded);
#   * keep the upload-then-delete checkpoint watcher running (started only once
#     training is actively stepping, i.e. past loading its resume checkpoint);
#   * after the final checkpoint is uploaded, push the run metadata/logs.
# Meant to run inside tmux:
#   ARM=ttt FINAL_STEP=64260 tmux new -d -s wam-ttt bash scripts/supervise_m5.sh
set -u
cd "$(dirname "$0")/.."

ARM=${ARM:-ttt}
case "$ARM" in ttt|control) ;; *) echo "ARM must be ttt or control" >&2; exit 2 ;; esac

PYTHON=${PYTHON:-/workspace/venv/bin/python}
OUT_DIR=${OUT_DIR:-/workspace/outputs/fastwam_m5_${ARM}_490ep}
HF_REPO=${HF_REPO:-SleepMastger/movecube-fastwam-m5-${ARM}-490ep}
FINAL_STEP=${FINAL_STEP:?set FINAL_STEP to the run total step count}
MAX_RESTARTS=${MAX_RESTARTS:-12}
CKPT_DIR="$OUT_DIR/checkpoints"
SUP_LOG="$OUT_DIR/supervisor.log"
TRAIN_LOG="$OUT_DIR/train_supervised_stdout.log"
WATCH_LOG="$OUT_DIR/checkpoint_upload.log"

mkdir -p "$CKPT_DIR"

export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
if [ -f /workspace/.hf_token ]; then export HF_TOKEN; HF_TOKEN=$(cat /workspace/.hf_token); fi
export HF_UPLOAD_ENABLED=0      # final metadata upload is done here, not by the launch script
export PYTHONUNBUFFERED=1

log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$SUP_LOG"; }

train_running()   { pgrep -f "train.py --arm $ARM .*--out $OUT_DIR" >/dev/null; }
watcher_running() { pgrep -f "watch_and_upload_checkpoints.py --repo $HF_REPO --run-dir $OUT_DIR" >/dev/null; }

last_logged_step() {
    [ -f "$OUT_DIR/log.jsonl" ] || { echo 0; return; }
    tail -n 1 "$OUT_DIR/log.jsonl" | "$PYTHON" -c 'import sys,json; print(json.loads(sys.stdin.read() or "{}").get("step",0))' 2>/dev/null || echo 0
}

training_done() {
    grep -qs "done: $FINAL_STEP steps" "$TRAIN_LOG" 2>/dev/null && return 0
    [ "$(last_logged_step)" -ge "$FINAL_STEP" ]
}

final_uploaded() {
    [ -f "$OUT_DIR/.uploaded_checkpoints.json" ] && grep -q "step_${FINAL_STEP}.pt" "$OUT_DIR/.uploaded_checkpoints.json"
}

# Newest local checkpoint that loads (complete zip, not mid-write). Prints path or nothing.
latest_local_ckpt() {
    "$PYTHON" - "$CKPT_DIR" <<'PY'
import re, sys, time, torch
from pathlib import Path
d = Path(sys.argv[1])
cands = sorted(d.glob("step_*.pt"), key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)), reverse=True)
for p in cands:
    if time.time() - p.stat().st_mtime < 120:   # possibly still being written
        continue
    try:
        payload = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
        if isinstance(payload, dict) and "step" in payload and "model" in payload:
            print(p); break
    except Exception as exc:
        print(f"unreadable checkpoint {p}: {exc}", file=sys.stderr)
PY
}

# Download the newest checkpoint from HF into CKPT_DIR. Prints path or nothing.
download_latest_hf_ckpt() {
    "$PYTHON" - "$HF_REPO" "$CKPT_DIR" <<'PY'
import re, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
repo, ckpt_dir = sys.argv[1], sys.argv[2]
api = HfApi()
try:
    names = [f for f in api.list_repo_files(repo) if re.fullmatch(r"checkpoints/step_\d+\.pt", f)]
except Exception as exc:
    print(f"cannot list {repo}: {exc}", file=sys.stderr); sys.exit(0)
if not names:
    sys.exit(0)
newest = max(names, key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
print(f"downloading {newest} from {repo}", file=sys.stderr)
out_dir = Path(ckpt_dir).parent           # file lands at <out_dir>/checkpoints/step_N.pt
print(hf_hub_download(repo, newest, local_dir=str(out_dir)))
PY
}

launch_training() {
    local ckpt="$1"
    local args=()
    [ -n "$ckpt" ] && args=(--resume "$ckpt")
    log "launching training ${args[*]:-from scratch}"
    OUT_DIR="$OUT_DIR" setsid nohup bash scripts/launch_m5_490ep.sh "$ARM" "${args[@]}" >> "$TRAIN_LOG" 2>&1 < /dev/null &
    disown
}

launch_watcher() {
    log "launching checkpoint watcher (upload then delete local)"
    setsid nohup "$PYTHON" scripts/watch_and_upload_checkpoints.py \
        --repo "$HF_REPO" --run-dir "$OUT_DIR" --final-step "$FINAL_STEP" \
        --poll-seconds 30 --delete-local >> "$WATCH_LOG" 2>&1 < /dev/null &
    disown
}

upload_metadata() {
    log "uploading run metadata to $HF_REPO"
    "$PYTHON" - "$HF_REPO" "$OUT_DIR" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import HfApi
repo, out = sys.argv[1], Path(sys.argv[2])
api = HfApi()
for p in sorted(out.iterdir()):
    if p.is_file() and (p.suffix in {".yaml", ".jsonl", ".log", ".json", ".md"}):
        api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=repo,
                        repo_type="model", commit_message=f"Upload {p.name}")
        print("uploaded", p.name, flush=True)
PY
}

restarts=0
last_launch=0
log "supervisor started (pid $$) arm=$ARM final_step=$FINAL_STEP out=$OUT_DIR repo=$HF_REPO"
while true; do
    if training_done; then
        log "training reached step $FINAL_STEP"
        for _ in $(seq 1 240); do
            final_uploaded && break
            watcher_running || launch_watcher
            sleep 30
        done
        final_uploaded && log "final checkpoint uploaded" || log "WARNING: final checkpoint not confirmed uploaded"
        upload_metadata && log "metadata uploaded" || log "WARNING: metadata upload failed"
        log "supervisor exiting"
        exit 0
    fi

    if ! train_running; then
        now=$(date +%s)
        if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
            log "ERROR: exceeded $MAX_RESTARTS restarts; giving up. Inspect $TRAIN_LOG"
            exit 1
        fi
        if [ "$last_launch" -ne 0 ] && [ $((now - last_launch)) -lt 900 ]; then
            log "training died within 15 min of launch; backing off 10 min before retrying"
            sleep 600
        fi
        ckpt=$(latest_local_ckpt 2>>"$SUP_LOG")
        if [ -z "$ckpt" ]; then
            log "no complete local checkpoint; checking HF"
            ckpt=$(download_latest_hf_ckpt 2>>"$SUP_LOG")
        fi
        [ -n "$ckpt" ] && log "resume checkpoint: $ckpt"
        launch_training "$ckpt"
        restarts=$((restarts + 1)); last_launch=$(date +%s)
        sleep 120
        continue
    fi

    # Start / restart the watcher only while training is actively stepping (log.jsonl fresh),
    # which guarantees the trainer has finished mmap-loading its resume checkpoint.
    if ! watcher_running; then
        if [ -f "$OUT_DIR/log.jsonl" ] && [ $(( $(date +%s) - $(stat -c %Y "$OUT_DIR/log.jsonl") )) -lt 300 ]; then
            launch_watcher
        fi
    fi
    sleep 60
done
