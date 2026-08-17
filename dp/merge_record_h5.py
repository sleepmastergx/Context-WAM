"""Merge the official record_dataset h5 with extra generated episodes.

Copies every `episode_*` group from --base, then every `episode_*` group from
each per-episode h5 under --extra-dir (the output of
dp/generate_oracle_episodes.py). Group names must be disjoint (official file is
episode_0..99; the generator starts at 100) -- a duplicate name aborts the
merge rather than silently overwriting.

    python dp/merge_record_h5.py \
        --base data/record_dataset_VideoUnmask.h5 \
        --extra-dir data/oracle_extra/hdf5_files \
        --out data/record_dataset_VideoUnmask_500.h5
"""
import argparse
import glob
import os

import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None,
                    help="official h5 to copy first; omit for a standalone "
                         "extension file (generated episodes only)")
    ap.add_argument("--extra-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert not os.path.exists(args.out), f"refusing to overwrite {args.out}"
    extra_files = sorted(glob.glob(os.path.join(args.extra_dir, "*.h5")))
    assert extra_files, f"no h5 files under {args.extra_dir}"

    seen = set()
    with h5py.File(args.out, "w") as out:
        if args.base:
            with h5py.File(args.base, "r") as base:
                for name in sorted(base.keys(), key=lambda s: int(s.split("_")[1])):
                    base.copy(name, out)
                    seen.add(name)
            print(f"copied {len(seen)} episodes from {args.base}")

        n_extra = 0
        for path in extra_files:
            with h5py.File(path, "r") as f:
                for name in f.keys():
                    assert name.startswith("episode_"), (path, name)
                    assert name not in seen, f"duplicate group {name} from {path}"
                    f.copy(name, out)
                    seen.add(name)
                    n_extra += 1
        print(f"copied {n_extra} extra episodes from {len(extra_files)} files")
    print(f"wrote {args.out}: {len(seen)} episodes total")


if __name__ == "__main__":
    main()
