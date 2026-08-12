"""Render the stage-2 arm comparison across checkpoints.

Both arms run the SAME val seeds, so per checkpoint we report the paired
(McNemar) test on discordant episodes, not just the marginal rates —
50 episodes gives +-8-10 pts on each rate, which alone cannot resolve a
10-pt effect, but the pairing removes episode difficulty from the noise.
"""
import glob, json, os, sys
from math import comb

RUNS = os.environ.get("DP_RUNS", "runs")
TASK = os.environ.get("DP_TASK", "MoveCube")
ARMS = [("arm2_control", "arm2 (no mem)"), ("arm4a_concat", "arm4a (TTT)")]
split = sys.argv[1] if len(sys.argv) > 1 else "val"


def load(arm, ck):
    r = {}
    for fp in glob.glob(f"{RUNS}/{arm}/eval_{TASK}_{split}_{ck}_shard*.json"):
        r.update(json.load(open(fp))["results"])
    return {int(k): v for k, v in r.items()}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    p = 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(p, 1.0)


cks = sorted({fp.split("_shard")[0].split(f"_{split}_")[-1]
              for arm, _ in ARMS
              for fp in glob.glob(f"{RUNS}/{arm}/eval_{TASK}_{split}_*_shard*.json")},
             key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

print(f"split={split}   (n=50 episodes per arm per checkpoint)")
print(f"{'ckpt':>8} {'arm2':>10} {'arm4a':>10} {'delta':>8} {'4a-only':>8} "
      f"{'2-only':>7} {'McNemar p':>10}")
for ck in cks:
    a2, a4 = load(ARMS[0][0], ck), load(ARMS[1][0], ck)
    if not a2 or not a4:
        continue
    common = sorted(set(a2) & set(a4))
    s2 = {e for e in common if a2[e] == "success"}
    s4 = {e for e in common if a4[e] == "success"}
    b, c = len(s2 - s4), len(s4 - s2)
    n = len(common)
    print(f"{ck:>8} {len(s2)/n:>9.1%} {len(s4)/n:>10.1%} "
          f"{(len(s4)-len(s2))/n:>+8.1%} {c:>8} {b:>7} {mcnemar(b, c):>10.3f}")
