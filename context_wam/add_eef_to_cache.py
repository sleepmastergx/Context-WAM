"""Add end-effector poses to a Fast-WAM window cache (in place, latents untouched).

For every ep*.npz: FK each frame's joint ACTION and joint STATE through the
benchmark's own recorder path (RobommeRecordWrapper._joint_action_to_ee_pose_dict
-- verified bit-exact against the h5 `eef_action` field) and store

    eef_action_pq [T, 7]  x y z  qw qx qy qz   (world frame, TCP link)
    eef_state_pq  [T, 7]  same, from the joint state (the deploy anchor)

Quaternions, not RPY: gpu_cache builds window-relative deltas as rotation
vectors (R_0^T R_k), which RPY subtraction cannot do (roll wraps at +-pi here).
Run in the DP env (needs sapien/mplib):

    source scripts/env-dp.sh && python context_wam/add_eef_to_cache.py --cache data/movecube_fastwam_500
"""
import argparse, glob, os, sys, tempfile
import numpy as np

_DP_REPO = os.environ.get("DP_REPO"); assert _DP_REPO, "source scripts/env-dp.sh"
sys.path.insert(0, _DP_REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--task", default="MoveCube")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    import gymnasium as gym
    from robomme.env_record_wrapper import RobommeRecordWrapper
    import importlib; importlib.import_module("robomme.robomme_env")   # registers env ids
    env = gym.make(args.task, obs_mode="rgb+depth+segmentation", control_mode="pd_joint_pos",
                   render_mode="rgb_array", reward_mode="dense", seed=0, difficulty="easy")
    w = RobommeRecordWrapper(env, dataset=tempfile.mkdtemp(), env_id=args.task,
                             episode=0, seed=0, save_video=False)
    w.reset()
    assert w._fk_available, "recorder FK unavailable"

    def fk(q8):
        """[T, 8] joint vectors (7 joints + gripper cmd/width) -> [T, 7] (p, quat wxyz)."""
        w._prev_action_ee_quat_wxyz = None; w._prev_action_ee_rpy_xyz = None
        out = np.zeros((len(q8), 7), np.float32)
        for i, q in enumerate(q8):
            d = w._joint_action_to_ee_pose_dict(np.asarray(q, np.float64))
            out[i, :3] = np.asarray(d["pose"]).ravel()[:3]
            out[i, 3:] = np.asarray(d["quat"]).ravel()
        return out

    files = sorted(glob.glob(os.path.join(args.cache, "ep*.npz")))[: args.limit]
    for p in files:
        z = dict(np.load(p))
        if "eef_action_pq" in z:
            continue
        z["eef_action_pq"] = fk(z["actions"])
        z["eef_state_pq"] = fk(z["states"])
        tmp = p + ".tmp.npz"
        np.savez_compressed(tmp, **z); os.replace(tmp, p)
        print("eef added:", os.path.basename(p), z["eef_action_pq"].shape, flush=True)
    env.close()


if __name__ == "__main__":
    main()
