"""Replay recorded h5 actions through the benchmark env -- the controller-
fidelity test for an action space. Generated episodes carry their own seeds
(setup/seed, setup/difficulty), so the env is built from those, not from the
split metadata.

    python dp/replay_h5_actions.py --h5 data/record_dataset_MoveCube_extra400.h5 \
        --action-space ee_pose --episodes 10
"""
import argparse, json, os, sys
import h5py, numpy as np
_DP_REPO = os.environ.get("DP_REPO"); assert _DP_REPO
sys.path.insert(0, _DP_REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--task", default="MoveCube")
    ap.add_argument("--action-space", default="ee_pose", choices=["ee_pose", "joint_angle"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    builder = BenchmarkEnvBuilder(env_id=args.task, dataset="train",
                                  action_space=args.action_space, max_steps=1300)
    key = "eef_action" if args.action_space == "ee_pose" else "joint_action"
    results = {}
    with h5py.File(args.h5, "r") as f:
        names = sorted(f.keys(), key=lambda n: int(n.split("_")[1]))[:args.episodes]
        for n in names:
            e = f[n]
            seed = int(e["setup/seed"][()]); diff = e["setup/difficulty"][()]
            diff = diff.decode() if isinstance(diff, bytes) else str(diff)
            ts = sorted([k for k in e if k.startswith("timestep")], key=lambda s: int(s.split("_")[1]))
            acts = [np.asarray(e[t][f"action/{key}"][()], np.float32) for t in ts
                    if not bool(e[t]["info/is_video_demo"][()])]
            builder.resolve_episode = lambda ep_idx, s=seed, d=diff: (s, d)
            env = builder.make_env_for_episode(0)
            obs, info = env.reset()
            outcome, steps = "unknown", 0
            for a in acts:
                obs, _, term, trunc, info = env.step(a)
                steps += 1
                if term or trunc:
                    outcome = info.get("status", "unknown"); break
            env.close()
            results[n] = outcome
            print(f"{n} seed={seed} {diff}: {outcome} after {steps}/{len(acts)} replayed actions", flush=True)
    succ = sum(v == "success" for v in results.values())
    print(f"{args.action_space} replay: {succ}/{len(results)} success")
    if args.out:
        json.dump({"action_space": args.action_space, "results": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
