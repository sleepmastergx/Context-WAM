"""Record one Fast-WAM closed-loop episode to an mp4 (via the action server).

Same loop as dp/eval_fastwam_client.py -- current frame + state to the server,
execute the first --exec-horizon of the returned chunk -- with every rendered
frame captured (conditioning demo first, then the rollout, front|wrist side by
side, outcome card at the end).

    python dp/record_episode_fastwam.py --episode 0 --split val \
        --out runs/eval_videos --tag step25000
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_fastwam_client import send_msg, recv_msg  # noqa: E402

_DP_REPO = os.environ.get("DP_REPO")
assert _DP_REPO, "set DP_REPO (source scripts/env-dp.sh)"
sys.path.insert(0, _DP_REPO)


def label(img, text, color=(255, 255, 255)):
    import cv2
    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                color, 1, cv2.LINE_AA)
    return img


def main():
    import socket
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/tmp/fastwam_eval.sock")
    ap.add_argument("--task", default="MoveCube")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=1300)
    ap.add_argument("--exec-horizon", type=int, default=8)
    ap.add_argument("--abort-after", type=int, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--tag", default="wam")
    ap.add_argument("--out", default="runs/eval_videos")
    args = ap.parse_args()

    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(args.socket)
    send_msg(conn, {"cmd": "ping"})
    assert recv_msg(conn).get("ok")

    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    builder = BenchmarkEnvBuilder(env_id=args.task, dataset=args.split,
                                  action_space="joint_angle",
                                  max_steps=args.max_steps)
    env = builder.make_env_for_episode(args.episode)
    obs, info = env.reset()
    goal = info["task_goal"][0]

    def pair(front, wrist, caption):
        return label(np.concatenate([np.asarray(front, np.uint8),
                                     np.asarray(wrist, np.uint8)], axis=1),
                     caption)

    frames = []
    for i, (f, w) in enumerate(zip(obs["front_rgb_list"], obs["wrist_rgb_list"])):
        frames.append(pair(f, w, f"conditioning {i+1}/{len(obs['front_rgb_list'])}"))

    outcome, steps, done, replans = "unknown", 0, False, 0
    t0 = time.time()
    while not done:
        js = np.asarray(obs["joint_state_list"][-1], np.float32).reshape(-1)
        gr = np.asarray(obs["gripper_state_list"][-1], np.float32).reshape(-1)
        state = np.concatenate([js, [gr.mean()]]).astype(np.float32)
        send_msg(conn, {"front": np.asarray(obs["front_rgb_list"][-1], np.uint8),
                        "wrist": np.asarray(obs["wrist_rgb_list"][-1], np.uint8),
                        "state": state, "seed": args.episode * 100000 + replans})
        acts = recv_msg(conn)["action"]
        replans += 1
        for a in acts[:args.exec_horizon]:
            obs, _, terminated, truncated, info = env.step(np.asarray(a, np.float32))
            steps += 1
            frames.append(pair(obs["front_rgb_list"][-1], obs["wrist_rgb_list"][-1],
                               f"step {steps}  grip={a[7]:+.2f}"))
            if info is not None and info.get("status") == "error":
                outcome, done = "error", True
                break
            if terminated or truncated:
                outcome, done = info.get("status", "unknown"), True
                break
        if not done and args.abort_after and steps >= args.abort_after:
            outcome, done = "aborted", True
    env.close()
    print(f"episode {args.episode}: {outcome} ({steps} steps, "
          f"{time.time()-t0:.0f}s) | goal: {goal}", flush=True)

    for _ in range(args.fps):
        frames.append(label(np.zeros_like(frames[-1]),
                            f"{outcome.upper()} after {steps} steps"))
    import imageio
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"fastwam_{args.tag}_{args.task}_"
                                  f"{args.split}_ep{args.episode}_{outcome}.mp4")
    imageio.mimsave(path, frames, fps=args.fps)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
