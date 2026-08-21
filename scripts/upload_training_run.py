"""Create a private Hugging Face model repo and upload a completed run."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Hugging Face model repo id")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    checkpoints = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    if not checkpoints:
        raise SystemExit(f"no completed checkpoints under {run_dir / 'checkpoints'}")
    if not (run_dir / "log.jsonl").is_file():
        raise SystemExit(f"missing training log: {run_dir / 'log.jsonl'}")

    card_path = run_dir / "README.md"
    if not card_path.exists():
        card_path.write_text(
            "---\n"
            "library_name: fastwam\n"
            "tags:\n"
            "- fastwam\n"
            "- robotics\n"
            "- video-generation\n"
            "- movecube\n"
            "---\n\n"
            "# Fast-WAM MoveCube 30x30\n\n"
            "Original equal-depth Fast-WAM with 30 video layers and 30 action "
            "layers, trained for 30 epochs on MoveCube episodes 0-89 and "
            "100-499. Official episodes 90-99 were held out.\n\n"
            f"This run contains {len(checkpoints)} checkpoints. The final "
            f"checkpoint is `{checkpoints[-1].name}`. See `run_config.yaml` "
            "and `log.jsonl` for the exact configuration and training log.\n",
            encoding="utf-8",
        )

    api = HfApi()
    identity = api.whoami()
    print(f"authenticated as {identity['name']}", flush=True)
    repo_url = api.create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    print(
        f"uploading {len(checkpoints)} checkpoints from {run_dir} to {repo_url}",
        flush=True,
    )
    api.upload_large_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=run_dir,
        private=True,
        num_workers=args.workers,
        print_report=True,
        print_report_every=60,
    )
    print(f"upload complete: {repo_url}", flush=True)


if __name__ == "__main__":
    main()
