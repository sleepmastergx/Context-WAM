"""Generate EXTRA VideoUnmask oracle demonstrations beyond the official 100.

Modeled 1:1 on the benchmark's own generation harness
(external/robomme_benchmark/tests/_shared/dataset_generation.py): plain
gym.make (no DemonstrationWrapper -- the RecordWrapper path records the video
phase as explicit timesteps), RobommeRecordWrapper for the official h5 layout,
FailAware motion planner with screw->RRT* retry, and the env's own
task_list[].solve(env, planner) as the expert. Only SUCCESSFUL episodes are
written (RecordWrapper gates write() on episode_success), so every recorded
episode is a valid demonstration; failures retry at seed+1 (up to
--max-attempts).

Seeds: episode i uses --base-seed + i*1000 (+attempt). Default 2,000,000 --
far above the official train (6,000-15,900), test (560,000+) and val
(1,060,000+) ranges, so no split leakage.

Difficulty matches the official train mix 2:1:1 (easy:medium:hard) via
episode-index round robin (i%4 -> easy,easy,medium,hard).

Output: <out>/hdf5_files/VideoUnmask_ep<N>_seed<S>.h5, one per episode, each
holding a single `episode_<N>` group in the record_dataset format; merge them
(plus the official file) with dp/merge_record_h5.py, then convert with
dp/convert_h5_to_lerobot.py.

Shardable: --shard/--num-shards splits the episode list round-robin, one
process per shard (same pattern as the eval runners).

    python dp/generate_oracle_episodes.py --out data/oracle_extra \
        --start 100 --count 400 --shard 0 --num-shards 6
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
import gymnasium as gym

from robomme.env_record_wrapper import RobommeRecordWrapper, FailsafeTimeout
from robomme.robomme_env import *  # noqa: F401,F403  (registers env ids)
from robomme.robomme_env.utils.SceneGenerationError import SceneGenerationError
from robomme.robomme_env.utils.planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,
    ScrewPlanFailure,
)

SCREW_MAX_ATTEMPTS = 3
RRT_MAX_ATTEMPTS = 3
DIFF_MIX = ["easy", "easy", "medium", "hard"]  # official train mix 2:1:1


def _tensor_to_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().any().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def _patch_planner_screw_to_rrt(planner) -> None:
    original_screw = planner.move_to_pose_with_screw
    original_rrt = planner.move_to_pose_with_RRTStar

    def _move_screw_then_rrt(*args, **kwargs):
        for _ in range(SCREW_MAX_ATTEMPTS):
            try:
                result = original_screw(*args, **kwargs)
            except ScrewPlanFailure:
                continue
            if isinstance(result, int) and result == -1:
                continue
            return result
        for _ in range(RRT_MAX_ATTEMPTS):
            try:
                result = original_rrt(*args, **kwargs)
            except Exception:
                continue
            if isinstance(result, int) and result == -1:
                continue
            return result
        return -1

    planner.move_to_pose_with_screw = _move_screw_then_rrt


def run_one(env_id, episode, seed, difficulty, out_dir) -> bool:
    env = gym.make(
        env_id,
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        seed=seed,
        difficulty=difficulty,
    )
    env = RobommeRecordWrapper(
        env,
        dataset=str(out_dir),
        env_id=env_id,
        episode=episode,
        seed=seed,
        # save_video=True is REQUIRED for h5 recording: RecordWrapper nests the
        # entire record_data/buffer path (h5 included) inside its
        # _video_should_record() gate, so False silently records nothing.
        save_video=True,
    )
    episode_successful = False
    try:
        env.reset()
        planner = FailAwarePandaArmMotionPlanningSolver(
            env,
            debug=False,
            vis=False,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )
        _patch_planner_screw_to_rrt(planner)

        tasks = list(getattr(env.unwrapped, "task_list", []) or [])
        for task_entry in tasks:
            solve_callable = task_entry.get("solve")
            if not callable(solve_callable):
                continue
            env.unwrapped.evaluate(solve_complete_eval=True)
            screw_failed = False
            try:
                solve_result = solve_callable(env, planner)
                if isinstance(solve_result, int) and solve_result == -1:
                    screw_failed = True
                    env.unwrapped.failureflag = torch.tensor([True])
                    env.unwrapped.successflag = torch.tensor([False])
                    env.unwrapped.current_task_failure = True
            except ScrewPlanFailure:
                screw_failed = True
                env.unwrapped.failureflag = torch.tensor([True])
                env.unwrapped.successflag = torch.tensor([False])
                env.unwrapped.current_task_failure = True
            except FailsafeTimeout:
                break

            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            fail_flag = evaluation.get("fail", False)
            success_flag = evaluation.get("success", False)
            if _tensor_to_bool(success_flag):
                episode_successful = True
                break
            if screw_failed or _tensor_to_bool(fail_flag):
                break
        else:
            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            episode_successful = _tensor_to_bool(evaluation.get("success", False))

        episode_successful = episode_successful or _tensor_to_bool(
            getattr(env, "episode_success", False)
        )
    except SceneGenerationError:
        episode_successful = False
    finally:
        try:
            env.close()  # writes the h5 iff episode_success
        except Exception:
            pass
    return episode_successful


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="VideoUnmask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=100,
                    help="first episode number (official train is 0-99)")
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--base-seed", type=int, default=2_000_000)
    ap.add_argument("--max-attempts", type=int, default=30)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"gen_log_shard{args.shard}.jsonl"

    my_eps = list(range(args.count))[args.shard::args.num_shards]
    for idx in my_eps:
        episode = args.start + idx
        difficulty = DIFF_MIX[idx % len(DIFF_MIX)]
        expect = out_dir / "hdf5_files"
        already = list(expect.glob(f"{args.env_id}_ep{episode}_seed*.h5")) \
            if expect.exists() else []
        if already:
            print(f"episode {episode}: exists ({already[0].name}), skipping",
                  flush=True)
            continue
        t0 = time.time()
        ok, used_seed, attempts = False, None, 0
        for attempt in range(args.max_attempts):
            seed = args.base_seed + idx * 1000 + attempt
            attempts = attempt + 1
            try:
                ok = run_one(args.env_id, episode, seed, difficulty, out_dir)
            except Exception as exc:
                print(f"episode {episode} seed {seed}: exception {exc!r}",
                      flush=True)
                ok = False
            if ok:
                used_seed = seed
                break
        rec = dict(episode=episode, difficulty=difficulty, ok=ok,
                   seed=used_seed, attempts=attempts,
                   sec=round(time.time() - t0, 1))
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"episode {episode}: {'OK' if ok else 'FAILED'} "
              f"(difficulty={difficulty}, attempts={attempts}, "
              f"{rec['sec']}s)", flush=True)


if __name__ == "__main__":
    main()
