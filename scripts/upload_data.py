"""One-time upload of the converted MoveCube cache to HuggingFace (private).

    python scripts/upload_data.py --cache /path/to/movecube_fastwam
"""
import argparse
import io

from huggingface_hub import HfApi

CARD = """---
license: other
license_name: robomme-derived
license_link: https://robomme.github.io
---

# MoveCube — Fast-WAM window cache (context-wam study)

Derived training cache for the context-wam experiments
(github.com/sleepmastergx/Context-WAM). NOT the raw benchmark.

- Source: RoboMME benchmark, MoveCube task, all 100 episodes
  (`record_dataset_MoveCube.h5`); splits: episodes 0-89 train, 90-99 val.
- Contents: per-episode `epNNNN.npz` with per-window Wan2.2 VAE latents
  ([48, 3, 16, 32] bf16; source video 9 frames @ stride 4 spanning 33 raw
  steps, 256x512), `actions` (T, 14->8), `states`, `starts` (stride 1),
  `exec_start`; `meta*.json` (window index); `text_context.pt` (precomputed
  T5 embedding of the single MoveCube goal string).
- Produced by `context_wam/convert_movecube.py` at 2026-08-09 settings
  (action_horizon 32).
- License/terms follow the RoboMME benchmark; this repo is private and for
  the study only.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo", default="SleepMastger/movecube-fastwam-cache")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=io.BytesIO(CARD.encode()),
                    path_in_repo="README.md", repo_id=args.repo,
                    repo_type="dataset")
    api.upload_folder(folder_path=args.cache, repo_id=args.repo,
                      repo_type="dataset",
                      commit_message="MoveCube Fast-WAM window cache (100 eps)")
    print(f"uploaded {args.cache} -> {args.repo}")


if __name__ == "__main__":
    main()
