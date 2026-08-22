"""Gate: action/proprio normalization round-trips exactly and the eval server's
inverse reproduces the raw joint targets the env expects.

    python checks/check_action_norm.py [--cache data/movecube_fastwam_ref]
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from context_wam.gpu_cache import GPUWindowCache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/movecube_fastwam_ref")
    args = ap.parse_args()
    raw = GPUWindowCache(args.cache, "cpu", split_episodes=[0], dtype=torch.float32,
                         storage_device="cpu", action_mode="raw")
    A_raw, S_raw = raw.actions.numpy(), raw.states.numpy()
    for mode in ("norm", "delta_norm"):
        c = GPUWindowCache(args.cache, "cpu", split_episodes=[0], dtype=torch.float32,
                           storage_device="cpu", action_mode=mode)
        st = c.action_stats
        A, S = c.actions.numpy(), c.states.numpy()
        assert A.min() >= -1 - 1e-5 and A.max() <= 1 + 1e-5, "actions not in [-1,1]"
        assert S.min() >= -1 - 1e-5 and S.max() <= 1 + 1e-5, "states not in [-1,1]"
        a_min, a_max = np.array(st["action_min"]), np.array(st["action_max"])
        s_min, s_max = np.array(st["state_min"]), np.array(st["state_max"])
        eps = st["eps"]
        # server-side inverse (same formula as dp/eval_fastwam_server.py)
        a = (A + 1) / 2 * (a_max - a_min + eps) + a_min
        if mode == "delta_norm":
            a[:, :, :7] += S_raw[:, None, :7]
        s_back = (S + 1) / 2 * (s_max - s_min + eps) + s_min
        da, ds = np.abs(a - A_raw).max(), np.abs(s_back - S_raw).max()
        assert da < 1e-4 and ds < 1e-4, f"{mode}: round-trip error a={da:.2e} s={ds:.2e}"
        motion = np.abs(np.diff(A, axis=1)).mean()
        print(f"PASS {mode:10s}: round-trip max err a={da:.1e} s={ds:.1e} | "
              f"per-step motion in normalized units {motion:.4f} "
              f"(raw: {np.abs(np.diff(A_raw, axis=1)).mean():.4f} rad)")
    print("ALL PASS -- normalization inverts exactly; deploy formula reproduces raw targets")


if __name__ == "__main__":
    main()
