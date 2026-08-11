"""Download the MoveCube Fast-WAM window cache from HuggingFace.

The repo is PRIVATE: run `huggingface-cli login` first (or set HF_TOKEN).

    python scripts/download_data.py --out data/movecube_fastwam
"""
import argparse

from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="SleepMastger/movecube-fastwam-cache")
    ap.add_argument("--out", default="data/movecube_fastwam")
    args = ap.parse_args()
    path = snapshot_download(repo_id=args.repo, repo_type="dataset",
                             local_dir=args.out)
    print(f"cache at: {path}")
    print("train with:  CACHE_DIR=" + args.out + " scripts/launch_control.sh")


if __name__ == "__main__":
    main()
