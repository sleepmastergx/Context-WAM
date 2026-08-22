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
    ap.add_argument("--eef-cache", default=None, help="cache dir with eef_*_pq keys")
    ap.add_argument("--eef-episodes", type=int, nargs="*", default=[100, 101])
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
    if args.eef_cache:
        from context_wam.se3 import compose
        c = GPUWindowCache(args.eef_cache, "cpu", split_episodes=args.eef_episodes, dtype=torch.float32,
                           storage_device="cpu", action_mode="eef_delta_norm")
        st = c.action_stats; A = c.actions.numpy()
        a_min, a_max, eps = np.array(st["action_min"]), np.array(st["action_max"]), st["eps"]
        d = (A + 1) / 2 * (a_max - a_min + eps) + a_min                     # [N,H,7] deltas
        raw = GPUWindowCache(args.eef_cache, "cpu", split_episodes=args.eef_episodes, dtype=torch.float32,
                             storage_device="cpu", action_mode="raw")
        # rebuild absolute poses from anchors and compare to the stored eef_action_pq
        import glob, os
        per_ep = {}
        for f in sorted(glob.glob(os.path.join(args.eef_cache, "ep*.npz"))):
            e = int(os.path.basename(f)[2:6])
            if e in set(c.ep.tolist()):          # only episodes the cache actually loaded
                z = np.load(f); per_ep[e] = (z["eef_action_pq"], z["eef_state_pq"], z["starts"])
        eps_sorted = sorted(per_ep); n0 = 0; worst_p = worst_q = 0.0
        for e in eps_sorted:
            EA, ES, starts = per_ep[e]; n = len(starts)
            dd = d[n0:n0+n]; anc = ES[starts]
            H = dd.shape[1]
            idx = np.clip(starts[:, None] + np.arange(H)[None, :], 0, len(EA) - 1)
            p, q = compose(anc[:, None, :3], anc[:, None, 3:7], dd[:, :, :3], dd[:, :, 3:6])
            worst_p = max(worst_p, np.abs(p - EA[idx][:, :, :3]).max())
            # angle of the relative rotation via atan2 -- arccos(|dot|) is
            # ill-conditioned near identity (float roundoff -> ~0.03 deg)
            from context_wam.se3 import quat_mul, quat_conj
            qr = quat_mul(quat_conj(q), EA[idx][:, :, 3:7])
            ang = np.degrees(2 * np.arctan2(np.linalg.norm(qr[..., 1:], axis=-1), np.abs(qr[..., 0])))
            worst_q = max(worst_q, float(ang.max()))
            assert np.allclose(dd[:, :, 6], raw.actions.numpy()[n0:n0+n, :, 7], atol=1e-5), "gripper not absolute"
            n0 += n
        assert worst_p < 1e-4 and worst_q < 0.01, f"eef compose error p={worst_p:.2e} q={worst_q:.4f}deg"
        print(f"PASS eef_delta_norm: compose(anchor, deltas) == recorded EEF pose "
              f"(max pos err {worst_p:.1e} m, rot err {worst_q:.4f} deg); action_dim={c.action_dim}")
    print("ALL PASS -- normalization inverts exactly; deploy formula reproduces raw targets")


if __name__ == "__main__":
    main()
