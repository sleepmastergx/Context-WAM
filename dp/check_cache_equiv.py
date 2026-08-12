"""Gate for cached_dataset.py: cached windows must be BIT-EXACT vs the
original RoboMMEDataset path, including every episode-boundary case
(obs window clamped at exec/episode start, action window clamped at end).

Runs on CPU with a 10-episode subset (login-node polite: ~1.6 GiB heap,
no shared memory). Also reports the per-sample speedup.
"""
import argparse, os, random, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DP_REPO = os.environ.get("DP_REPO")
assert DP_REPO, "set DP_REPO to your clone of github.com/RoboMME/DP"
sys.path.insert(0, DP_REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--stats-dir", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--random-samples", type=int, default=100)
    args = ap.parse_args()

    from eval_envs.dataset.robomme_dataset import RoboMMEDataset
    from cached_dataset import CachedRoboMMEDataset

    common = dict(dataset_root=args.dataset_root, obs_horizon=2,
                  action_exec_horizon=8, action_pred_horizon=16,
                  stats_path=args.stats_dir, embed_text=False)
    ref = RoboMMEDataset(**common)
    cached = CachedRoboMMEDataset(**common, cache_workers=8,
                                  cache_share=False, max_episodes=args.episodes)
    n = len(cached)
    assert cached._valid_indices == ref._valid_indices[:n], \
        "cached valid-index prefix diverges from reference"

    # boundary indices: first 3 and last 17 windows of every cached episode
    picks = set()
    starts = [i for i in range(n)
              if i == 0 or cached._row_ep[cached._valid_indices[i]]
              != cached._row_ep[cached._valid_indices[i - 1]]]
    for s_i, s in enumerate(starts):
        e = starts[s_i + 1] if s_i + 1 < len(starts) else n
        picks.update(range(s, min(s + 3, e)))
        picks.update(range(max(s, e - 17), e))
    rng = random.Random(0)
    picks.update(rng.randrange(n) for _ in range(args.random_samples))
    picks = sorted(picks)

    for k, i in enumerate(picks):
        a, b = ref[i], cached[i]
        assert set(a.keys()) == set(b.keys()), (i, a.keys(), b.keys())
        for key in a:
            av, bv = np.asarray(a[key]), np.asarray(b[key])
            assert av.shape == bv.shape and av.dtype == bv.dtype, \
                (i, key, av.shape, bv.shape, av.dtype, bv.dtype)
            assert np.array_equal(av, bv), \
                f"MISMATCH idx={i} key={key} maxdiff={np.abs(av-bv).max()}"
    print(f"EQUIVALENCE OK: {len(picks)} windows bit-exact "
          f"({len(starts)} episode-boundary regions + {args.random_samples} random)")

    for name, ds in (("original", ref), ("cached", cached)):
        t0 = time.time()
        for i in picks[:60]:
            ds[i]
        print(f"  {name:<9}: {(time.time()-t0)/60*1000:7.1f} ms/sample")


if __name__ == "__main__":
    main()
