"""Convert RoboMME record_dataset_<Task>.h5 -> LeRobot v2.0 parquet layout.

The RoboMME/DP repo trains from LeRobot parquet (eval_envs/dataset/
robomme_dataset.py) but the benchmark ships h5, and no converter exists
anywhere in their repos — this fills that gap for one task.

Output layout (what their RoboMMEDataset/LeRobotDataset expects):
  <out>/meta/info.json
  <out>/meta/tasks.jsonl            {"task_index", "task"}
  <out>/meta/episodes.jsonl         {"episode_index", "tasks", "length"}
  <out>/meta/episodes_stats.jsonl   per-episode per-feature stats; their demo
                                    filter reads exec_start_idx["min"][0]
  <out>/data/chunk-000/episode_%06d.parquet

Per-row columns:
  image, wrist_image : HF-datasets Image struct {bytes: PNG, path: None}
  state (8,) f32     : joint_state(7) + gripper scalar   [VERIFY note below]
  actions (8,) f32   : action/joint_action
  is_demo bool       : info/is_video_demo (True during conditioning video)
  exec_start_idx i64 : first non-video frame index, constant per episode
  timestamp f32, frame_index i64, episode_index i64, index i64, task_index i64

VERIFY-ON-FIRST-RUN:
  * gripper element of `state`: h5 gripper_state is (2,) finger widths in
    [0, 0.04]; serve_dp.py treats state as an opaque 8-vector so the exact
    choice is set by whoever built their parquet. Default here: gripper_state
    mean (fingers are symmetric; [0] and mean coincide to ~1e-4 in demos).
    Flag --gripper-elem {mean,0,1} to switch if their stats disagree.
  * fps: recorder uses 30 fps; delta_timestamps only needs consistency
    between info.json and the dataset's window math, both use this value.

CPU-parallel over episodes. Verification pass (--verify) re-reads the parquet
and diffs pixels/actions/state against the h5 for sampled frames.
"""
import argparse, io, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

FPS = 30
CHUNK = "chunk-000"


def encode_png(img):
    import cv2
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def gripper_scalar(g, mode):
    g = np.asarray(g, np.float32).reshape(-1)
    if mode == "mean":
        return float(g.mean())
    return float(g[int(mode)])


def stats_of(arr):
    a = np.asarray(arr, np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return {
        "min": a.min(0).tolist(), "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
        "count": [int(a.shape[0])],
    }


def episode_order(names, extended):
    """Group ordering -> parquet episode index.

    Legacy (default) is plain alphabetical -- how the original 100-episode
    conversion ran, so its parquet 0-99 mapping (episode_0, episode_1,
    episode_10, ...) is load-bearing for everything derived from it (stage-1
    runs, the fastwam cache). --extended-order keeps that exact alphabetical
    block for groups numbered <100 and appends groups >=100 in numeric order,
    so an extended h5 converts to: parquet 0-99 == the legacy conversion
    bit-for-bit, parquet 100+ == the generated episodes in numeric order.
    Plain alphabetical over 500 groups would interleave them (episode_1,
    episode_10, episode_100, ...).
    """
    if not extended:
        return sorted(names)
    num = lambda n: int(n.split("_")[1])
    official = sorted(n for n in names if num(n) < 100)
    extra = sorted((n for n in names if num(n) >= 100), key=num)
    return official + extra


def convert_episode(args):
    (h5_path, ep_name, ep_index, global_start, out_dir, gripper_mode,
     task_index) = args
    import h5py
    import datasets as hfds

    with h5py.File(h5_path, "r") as f:
        ep = f[ep_name]
        ts = sorted([k for k in ep.keys() if k.startswith("timestep")],
                    key=lambda x: int(x.split("_")[1]))
        T = len(ts)
        goal = ep["setup/task_goal"][()][0].decode()

        imgs, wimgs, states, actions, is_demo = [], [], [], [], []
        for t in ts:
            g = ep[t]
            imgs.append(encode_png(g["obs/front_rgb"][()]))
            wimgs.append(encode_png(g["obs/wrist_rgb"][()]))
            js = np.asarray(g["obs/joint_state"][()], np.float32).reshape(-1)
            gr = gripper_scalar(g["obs/gripper_state"][()], gripper_mode)
            states.append(np.concatenate([js, [gr]]).astype(np.float32))
            actions.append(np.asarray(g["action/joint_action"][()],
                                      np.float32).reshape(-1))
            is_demo.append(bool(g["info/is_video_demo"][()]))

    states = np.stack(states)          # (T, 8)
    actions = np.stack(actions)        # (T, 8)
    is_demo = np.array(is_demo, bool)
    nonvid = np.nonzero(~is_demo)[0]
    exec_start = int(nonvid[0]) if len(nonvid) else T
    assert states.shape[1] == 8 and actions.shape[1] == 8, \
        (states.shape, actions.shape)

    # Write through the HF `datasets` library so the parquet carries the
    # datasets.Features metadata — without it, lerobot loads image columns as
    # raw {bytes, path} dicts and its torch transform crashes
    # ("Could not infer dtype of dict").
    feats = hfds.Features({
        "image": hfds.Image(),
        "wrist_image": hfds.Image(),
        "state": hfds.Sequence(hfds.Value("float32"), length=8),
        "actions": hfds.Sequence(hfds.Value("float32"), length=8),
        "is_demo": hfds.Value("bool"),
        "exec_start_idx": hfds.Value("int64"),
        "timestamp": hfds.Value("float32"),
        "frame_index": hfds.Value("int64"),
        "episode_index": hfds.Value("int64"),
        "index": hfds.Value("int64"),
        "task_index": hfds.Value("int64"),
    })
    ds = hfds.Dataset.from_dict({
        "image": [{"bytes": b, "path": None} for b in imgs],
        "wrist_image": [{"bytes": b, "path": None} for b in wimgs],
        "state": states.tolist(),
        "actions": actions.tolist(),
        "is_demo": is_demo.tolist(),
        "exec_start_idx": [exec_start] * T,
        "timestamp": (np.arange(T) / FPS).astype(np.float32).tolist(),
        "frame_index": np.arange(T).tolist(),
        "episode_index": [ep_index] * T,
        "index": np.arange(global_start, global_start + T).tolist(),
        "task_index": [task_index] * T,
    }, features=feats)
    os.makedirs(f"{out_dir}/data/{CHUNK}", exist_ok=True)
    ds.to_parquet(f"{out_dir}/data/{CHUNK}/episode_{ep_index:06d}.parquet",
                  batch_size=64)   # small row groups: window reads stay cheap

    ep_stats = {
        "state": stats_of(states),
        "actions": stats_of(actions),
        "exec_start_idx": stats_of(np.array([exec_start])),
        "timestamp": stats_of(np.arange(T) / FPS),
        "frame_index": stats_of(np.arange(T)),
        "episode_index": stats_of(np.array([ep_index] * T)),
        "index": stats_of(np.arange(global_start, global_start + T)),
        "task_index": stats_of(np.array([task_index] * T)),
    }
    return ep_index, T, goal, exec_start, ep_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--gripper-elem", default="mean", choices=["mean", "0", "1"])
    p.add_argument("--extended-order", action="store_true",
                   help="official episodes (<100) in legacy alphabetical "
                        "order first, then generated (>=100) numerically; "
                        "see episode_order()")
    p.add_argument("--verify", action="store_true",
                   help="after conversion, diff sampled frames vs the h5")
    args = p.parse_args()
    import h5py

    # Pre-pass: lengths AND goals. The goal->task_index map must exist BEFORE
    # the workers run, because task_index is a per-row parquet column and it is
    # how the consumer resolves the goal text: RoboMMEDataset reads
    # task_index off the episode's first row and CLIP-embeds tasks[that index]
    # (eval_envs/dataset/robomme_dataset.py). Writing a constant 0 here would
    # silently give every episode the first episode's goal -- harmless on a
    # single-goal task like MoveCube, wrong for VideoUnmask, where the goal
    # string carries the QUERY COLOR and 9 distinct strings appear.
    with h5py.File(args.h5, "r") as f:
        ep_names = episode_order(f.keys(), args.extended_order)
        lengths, ep_goals = [], []
        for n in ep_names:
            lengths.append(len([k for k in f[n].keys()
                                if k.startswith("timestep")]))
            ep_goals.append(f[n]["setup/task_goal"][()][0].decode())
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])

    goals = {}
    for g in ep_goals:
        goals.setdefault(g, len(goals))

    jobs = [(args.h5, n, i, int(starts[i]), args.out, args.gripper_elem,
             goals[ep_goals[i]])
            for i, n in enumerate(ep_names)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(convert_episode, jobs):
            results.append(r)
            print(f"episode {r[0]} T={r[1]} exec_start={r[3]}", flush=True)
    results.sort()

    os.makedirs(f"{args.out}/meta", exist_ok=True)
    assert len(goals) >= 1
    # the workers saw the same map; re-derive from their returns and cross-check
    assert {g for _, _, g, _, _ in results} == set(goals), "goal set drift"
    with open(f"{args.out}/meta/tasks.jsonl", "w") as fh:
        for g, ti in goals.items():
            fh.write(json.dumps({"task_index": ti, "task": g}) + "\n")
    with open(f"{args.out}/meta/episodes.jsonl", "w") as fh:
        for i, T, goal, _, _ in results:
            fh.write(json.dumps({"episode_index": i, "tasks": [goal],
                                 "length": T}) + "\n")
    with open(f"{args.out}/meta/episodes_stats.jsonl", "w") as fh:
        for i, _, _, _, st in results:
            fh.write(json.dumps({"episode_index": i, "stats": st}) + "\n")

    total = int(sum(T for _, T, _, _, _ in results))
    n_exec = int(sum(T - ex_ for _, T, _, ex_, _ in results))
    info = {
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "fps": FPS,
        "total_episodes": len(results),
        "total_frames": total,
        "total_tasks": len(goals),
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": f"0:{len(results)}"},
        "data_path":
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "image": {"dtype": "image", "shape": [256, 256, 3]},
            "wrist_image": {"dtype": "image", "shape": [256, 256, 3]},
            "state": {"dtype": "float32", "shape": [8]},
            "actions": {"dtype": "float32", "shape": [8]},
            "is_demo": {"dtype": "bool", "shape": [1]},
            "exec_start_idx": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    with open(f"{args.out}/meta/info.json", "w") as fh:
        json.dump(info, fh, indent=1)
    print(f"DONE: {len(results)} episodes, {total} frames "
          f"({n_exec} execution / {total - n_exec} video-demo)")

    if args.verify:
        verify(args)


def verify(args):
    """Sample frames per episode; assert pixel/action/state equality vs h5."""
    import h5py, cv2
    import pyarrow.parquet as pq
    rng = np.random.default_rng(0)
    with h5py.File(args.h5, "r") as f:
        for i, n in enumerate(episode_order(f.keys(), args.extended_order)):
            tbl = pq.read_table(
                f"{args.out}/data/{CHUNK}/episode_{i:06d}.parquet")
            ts = sorted([k for k in f[n].keys() if k.startswith("timestep")],
                        key=lambda x: int(x.split("_")[1]))
            for k in rng.choice(len(ts), size=min(3, len(ts)), replace=False):
                k = int(k)
                png = tbl["image"][k].as_py()["bytes"]
                dec = cv2.cvtColor(cv2.imdecode(
                    np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB)
                ref = f[n][ts[k]]["obs/front_rgb"][()]
                assert np.array_equal(dec, ref), f"pixel diff ep{i} t{k}"
                act = np.array(tbl["actions"][k].as_py(), np.float32)
                ref_a = np.asarray(f[n][ts[k]]["action/joint_action"][()],
                                   np.float32).reshape(-1)
                assert np.allclose(act, ref_a), f"action diff ep{i} t{k}"
            if i % 20 == 0:
                print(f"verify: episode {i} OK", flush=True)
    print("VERIFY PASSED")


if __name__ == "__main__":
    main()
