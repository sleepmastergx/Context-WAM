"""Upload each completed training checkpoint while training continues."""

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import HfApi


def checkpoint_is_complete(path: Path) -> bool:
    """A torch zip checkpoint is only readable once its central directory is
    written, so a successful lazy (mmap) load proves the save finished."""
    try:
        import torch
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        return isinstance(payload, dict) and "step" in payload and "model" in payload
    except Exception as exc:
        print(f"CHECKPOINT_INCOMPLETE checkpoint={path.name} error={exc}", flush=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--final-step", required=True, type=int)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--delete-local", action="store_true",
                        help="delete each local checkpoint after verified upload")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"
    state_path = run_dir / ".uploaded_checkpoints.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    uploaded = set(state.get("uploaded", []))
    observed: dict[str, tuple[int, int, int]] = {}

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    print(f"CHECKPOINT_WATCHER_READY repo={args.repo}", flush=True)
    if args.delete_local:
        for checkpoint_name in sorted(uploaded):
            uploaded_path = checkpoint_dir / checkpoint_name
            if uploaded_path.exists():
                uploaded_path.unlink()
                print(
                    f"CHECKPOINT_CLEARED checkpoint={checkpoint_name}",
                    flush=True,
                )

    while True:
        for path in sorted(checkpoint_dir.glob("step_*.pt")):
            if path.name in uploaded:
                continue
            stat = path.stat()
            prior = observed.get(path.name)
            unchanged = (
                prior is not None
                and prior[0] == stat.st_size
                and prior[1] == stat.st_mtime_ns
            )
            stable_scans = (prior[2] + 1) if unchanged else 0
            observed[path.name] = (stat.st_size, stat.st_mtime_ns, stable_scans)
            if stable_scans < 1 or time.time() - stat.st_mtime < args.poll_seconds:
                continue
            if not checkpoint_is_complete(path):
                continue

            try:
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=f"checkpoints/{path.name}",
                    repo_id=args.repo,
                    repo_type="model",
                    commit_message=f"Upload {path.stem}",
                )
                for metadata_name in ("run_config.yaml", "log.jsonl"):
                    metadata_path = run_dir / metadata_name
                    if metadata_path.is_file():
                        api.upload_file(
                            path_or_fileobj=str(metadata_path),
                            path_in_repo=metadata_name,
                            repo_id=args.repo,
                            repo_type="model",
                            commit_message=f"Update metadata for {path.stem}",
                        )
            except Exception as exc:  # retry on the next scan
                print(
                    f"CHECKPOINT_UPLOAD_RETRY checkpoint={path.name} error={exc}",
                    flush=True,
                )
                continue

            uploaded.add(path.name)
            state_path.write_text(
                json.dumps({"uploaded": sorted(uploaded)}, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"CHECKPOINT_UPLOADED checkpoint={path.name} "
                f"repo=https://huggingface.co/{args.repo}",
                flush=True,
            )
            if args.delete_local:
                path.unlink()
                print(
                    f"CHECKPOINT_CLEARED checkpoint={path.name}",
                    flush=True,
                )

            if path.name == f"step_{args.final_step}.pt":
                print("CHECKPOINT_WATCHER_DONE", flush=True)
                return

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
