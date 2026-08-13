#!/usr/bin/env python3
"""Mirror training checkpoints from the network volume to a HuggingFace repo.

Checkpoints are WRITTEN to /workspace (fast, local to the pod) and mirrored to
HF in the background so the volume does not grow without bound. With --keep N
the local copy of an already-uploaded checkpoint is deleted, newest N retained.

Two writers to handle, and they differ:
  dp/train_ddp.py       runs/dp_stage1/ckpt_<step>.pth   torch.save -> .tmp_ckpt_*
                        then shutil.move, so the final name appears atomically.
  dp/stage2/train_stage2.py  <out>/checkpoints/<epoch>.ckpt  plain torch.save,
                        so the file is visible WHILE being written.
Hence the stability check: a candidate must have the same size and mtime across
two consecutive polls before it is uploaded. .tmp_* is skipped outright.

Uploaded files are recorded in <watch>/.hf_sync_state.json keyed by
name+size+mtime, so a restart does not re-upload and a rewritten checkpoint of
the same name does.

    # one-shot
    python /workspace/tools/hf_sync.py --watch runs/dp_stage1 --once
    # daemon alongside training, prune all but the newest 2 local copies
    python /workspace/tools/hf_sync.py --watch runs/dp_stage1 --keep 2 &
"""
import argparse, json, os, sys, time
from pathlib import Path

DEFAULT_REPO = "SleepMastger/context-wam-videounmask-checkpoints"
PATTERNS = ("*.pth", "*.ckpt", "*.pt")
STATE = ".hf_sync_state.json"


def log(msg):
    print(f"[hf_sync {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state(watch):
    p = Path(watch) / STATE
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            log(f"WARNING: {p} unreadable, starting fresh")
    return {}


def save_state(watch, state):
    p = Path(watch) / STATE
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(p)


def candidates(watch):
    """Checkpoint files under watch, excluding in-progress temporaries."""
    out = []
    for pat in PATTERNS:
        for f in Path(watch).rglob(pat):
            if f.name.startswith(".tmp") or f.name.startswith("."):
                continue
            out.append(f)
    return sorted(out)


def fingerprint(f):
    st = f.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def sync_once(api, watch, repo_id, repo_type, keep, seen, stable):
    """One pass. Returns number uploaded. Mutates seen/stable in place."""
    uploaded = 0
    for f in candidates(watch):
        rel = str(f.relative_to(watch))
        fp = fingerprint(f)
        if seen.get(rel) == fp:
            continue
        # require two identical observations before trusting a non-atomic write
        if stable.get(rel) != fp:
            stable[rel] = fp
            log(f"pending (awaiting stable size): {rel}")
            continue
        size_gb = f.stat().st_size / 2**30
        log(f"uploading {rel} ({size_gb:.2f} GiB) -> {repo_id}")
        t0 = time.time()
        try:
            api.upload_file(path_or_fileobj=str(f), path_in_repo=rel,
                            repo_id=repo_id, repo_type=repo_type)
        except Exception as e:                     # noqa: BLE001 - keep looping
            log(f"FAILED {rel}: {type(e).__name__}: {e}")
            continue
        dt = time.time() - t0
        log(f"done {rel} in {dt:.0f}s ({size_gb / max(dt, 1e-9) * 1024:.0f} MiB/s)")
        seen[rel] = fp
        save_state(watch, seen)
        uploaded += 1

    if keep is not None:
        prune(watch, keep, seen)
    return uploaded


def prune(watch, keep, seen):
    """Delete local copies of uploaded checkpoints, keeping the newest `keep`."""
    done = [f for f in candidates(watch)
            if seen.get(str(f.relative_to(watch))) == fingerprint(f)]
    done.sort(key=lambda f: f.stat().st_mtime)
    for f in done[:max(0, len(done) - keep)]:
        size_gb = f.stat().st_size / 2**30
        f.unlink()
        log(f"pruned local {f.relative_to(watch)} ({size_gb:.2f} GiB freed; "
            f"copy is on HF)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", required=True, help="directory to mirror")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--keep", type=int, default=None,
                    help="keep only the newest N uploaded ckpts on the volume "
                         "(default: keep everything)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--private", action="store_true", default=True)
    args = ap.parse_args()

    # tqdm bars are unreadable in a nohup log; the per-file done lines suffice
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import HfApi, get_token
    watch = Path(args.watch).resolve()
    if not watch.is_dir():
        sys.exit(f"--watch {watch} is not a directory")

    # get_token() resolves HF_TOKEN then $HF_HOME/token; HfApi().token does
    # NOT -- it only reports what was passed to the constructor.
    token = get_token()
    if token is None:
        sys.exit("no HF token; set HF_TOKEN or write $HF_HOME/token")
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type=args.repo_type,
                    private=args.private, exist_ok=True)
    log(f"mirroring {watch} -> {args.repo} ({args.repo_type}), "
        f"keep={args.keep}, interval={args.interval}s")

    seen, stable = load_state(watch), {}
    if args.once:
        n = sync_once(api, watch, args.repo, args.repo_type, args.keep,
                      seen, stable)
        # second pass so files that were merely "pending" this round can settle
        if n == 0:
            time.sleep(min(5.0, args.interval))
            sync_once(api, watch, args.repo, args.repo_type, args.keep,
                      seen, stable)
        return
    while True:
        try:
            sync_once(api, watch, args.repo, args.repo_type, args.keep,
                      seen, stable)
        except KeyboardInterrupt:
            log("stopped")
            return
        except Exception as e:                     # noqa: BLE001
            log(f"pass failed: {type(e).__name__}: {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
