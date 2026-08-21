"""Closed-loop MoveCube eval client for the Fast-WAM action server.

Runs in the DP env (.venv-dp) next to a running dp/eval_fastwam_server.py
(.venv-wam). Mirrors dp/eval_dp.py's episode loop: the env plays the
conditioning demo itself at reset; per replan we send the CURRENT front/wrist
frames + state to the server and execute the first --exec-horizon of the
returned 32-action chunk. Actions are raw joint values (no normalization --
the WAM trains on raw cached actions).

Never reads info["simple_subgoal*"] (benchmark oracle plan -- a leak).

    python dp/eval_fastwam_client.py --task MoveCube --split val \
        --episodes 50 --out runs/wam30/eval_MoveCube_val_step5000.json
"""
import argparse
import json
import os
import pickle
import socket
import struct
import sys
import time

import numpy as np

_DP_REPO = os.environ.get("DP_REPO")
assert _DP_REPO, "set DP_REPO (source scripts/env-dp.sh)"
sys.path.insert(0, _DP_REPO)


def send_msg(conn, obj):
    b = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack("<Q", len(b)) + b)


def recv_msg(conn):
    hdr = b""
    while len(hdr) < 8:
        c = conn.recv(8 - len(hdr))
        if not c:
            raise ConnectionError("server closed")
        hdr += c
    (n,) = struct.unpack("<Q", hdr)
    buf = b""
    while len(buf) < n:
        c = conn.recv(min(1 << 20, n - len(buf)))
        if not c:
            raise ConnectionError("server closed")
        buf += c
    return pickle.loads(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/tmp/fastwam_eval.sock")
    ap.add_argument("--task", default="MoveCube")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=1300)
    ap.add_argument("--exec-horizon", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--render-backend", default=None)
    ap.add_argument("--noise-mode", default="fresh", choices=["fresh", "fixed"],
                    help="fresh: new denoising seed every replan (plan may "
                         "switch modes). fixed: one seed per episode, so "
                         "consecutive replans stay in the same mode -- the "
                         "candidate for small R (TTT cadence needs R=8).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.render_backend:
        import gymnasium as gym
        _orig_make = gym.make

        def _make(env_id, **kw):
            kw.setdefault("render_backend", args.render_backend)
            return _orig_make(env_id, **kw)
        gym.make = _make

    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(args.socket)
    send_msg(conn, {"cmd": "ping"})
    assert recv_msg(conn).get("ok"), "server ping failed"
    print("server connected", flush=True)

    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    builder = BenchmarkEnvBuilder(env_id=args.task, dataset=args.split,
                                  action_space="joint_angle",
                                  max_steps=args.max_steps)
    n = min(args.episodes, builder.get_episode_num())
    my_eps = list(range(n))[args.shard::args.num_shards]

    results, steps_per_ep = {}, {}
    for ep_i in my_eps:
        env = builder.make_env_for_episode(ep_i)
        obs, info = env.reset()
        outcome, steps, done, replans = "unknown", 0, False, 0
        t_ep = time.time()
        # frames not yet shipped to the server (memory arms consume EVERY
        # frame; control arms ignore these fields)
        new_f, new_w, new_s = [], [], []

        def collect(o):
            fl, wl = o["front_rgb_list"], o["wrist_rgb_list"]
            js_l, gr_l = o["joint_state_list"], o["gripper_state_list"]
            for i in range(len(fl)):
                j = min(i, len(js_l) - 1)
                js = np.asarray(js_l[j], np.float32).reshape(-1)
                gr = np.asarray(gr_l[j], np.float32).reshape(-1)
                new_f.append(np.asarray(fl[i], np.uint8))
                new_w.append(np.asarray(wl[i], np.uint8))
                new_s.append(np.concatenate([js, [gr.mean()]]).astype(np.float32))

        collect(obs)   # reset() carries the whole conditioning demo
        first = True
        while not done:
            js = np.asarray(obs["joint_state_list"][-1], np.float32).reshape(-1)
            gr = np.asarray(obs["gripper_state_list"][-1], np.float32).reshape(-1)
            state = np.concatenate([js, [gr.mean()]]).astype(np.float32)
            send_msg(conn, {"front": np.asarray(obs["front_rgb_list"][-1], np.uint8),
                            "wrist": np.asarray(obs["wrist_rgb_list"][-1], np.uint8),
                            "state": state,
                            "ep_start": first,
                            "frames_front": new_f, "frames_wrist": new_w,
                            "frames_states": new_s,
                            "seed": ep_i * 100000 + (0 if args.noise_mode == "fixed" else replans)})
            first = False
            new_f, new_w, new_s = [], [], []
            acts = recv_msg(conn)["action"]
            replans += 1
            for a in acts[:args.exec_horizon]:
                obs, _, terminated, truncated, info = env.step(
                    np.asarray(a, np.float32))
                steps += 1
                if info is not None and info.get("status") == "error":
                    outcome, done = "error", True
                    break
                if terminated or truncated:
                    outcome, done = info.get("status", "unknown"), True
                    break
                collect(obs)
        env.close()
        results[ep_i] = outcome
        steps_per_ep[ep_i] = steps
        print(f"episode {ep_i}: {outcome} ({steps} steps, {replans} replans, "
              f"{time.time()-t_ep:.0f}s)", flush=True)

    succ = sum(v == "success" for v in results.values())
    print(f"shard {args.shard}/{args.num_shards}: {succ}/{len(results)} "
          f"({succ/max(len(results),1):.1%})")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"task": args.task, "split": args.split,
                   "exec_horizon": args.exec_horizon,
                   "noise_mode": args.noise_mode,
                   "results": results, "steps": steps_per_ep,
                   "success": succ, "n": len(results),
                   "rate": succ / max(len(results), 1)}, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
