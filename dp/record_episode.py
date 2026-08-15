"""Record one eval episode to an mp4, for eyeballing what the policy does.

Drives the policy through EXACTLY the same loop as dp/eval_dp.py -- same obs
construction, same DataTransform round-trip, same exec-horizon chunking -- so
what you see is what eval scored. Frames are front|wrist side by side.

The clip opens with the episode's conditioning video (the frames already sitting
in obs["front_rgb_list"] at reset, which for VideoUnmask is the "watch the
video" phase that names where the cube is hidden), then the rollout.

    python dp/record_episode.py --run-dir runs/dp_stage1 --ckpt ckpt_100000.pth \
        --episode 14 --split val --render-backend cpu

NOTE: the diffusion sampling is unseeded, so a re-run of a scored episode may
end differently. Pass --seed to make it reproducible.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_dp import load_policy, chw01, HistoryBuf  # noqa: E402


def label(img, text, color=(255, 255, 255)):
    """Burn a caption into the top-left of a frame (cv2 is a hard dep here)."""
    import cv2
    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                color, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--task", default="VideoUnmask")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--max-steps", type=int, default=1300)
    ap.add_argument("--render-backend", default=None)
    ap.add_argument("--raw-weights", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed torch so the diffusion sampling is reproducible; "
                         "attempt N uses seed+N")
    ap.add_argument("--want", default=None,
                    choices=["success", "fail", "timeout", "error"],
                    help="retry with new seeds until this outcome occurs. Eval "
                         "runs unseeded, so a scored outcome can only be "
                         "searched for, never replayed exactly.")
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--abort-after", type=int, default=None,
                    help="while searching for --want, give up on an attempt "
                         "after this many steps (ignored for --want timeout)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(
                        os.environ.get("REPO_ROOT", "."), "runs", "eval_videos"))
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.render_backend:
        import gymnasium as gym
        _orig_make = gym.make

        def _make(env_id, **kw):
            kw.setdefault("render_backend", args.render_backend)
            return _orig_make(env_id, **kw)
        gym.make = _make

    t0 = time.time()
    def log(*a):
        print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf, cfg = load_policy(args.run_dir, args.ckpt, device,
                                 use_ema=not args.raw_weights)
    obs_h, exec_h = int(cfg.obs_horizon), int(cfg.action_exec_horizon)
    log("policy loaded")

    include_text = bool(cfg.model.get("include_text", True))
    if include_text:
        from eval_envs.utils.clip_model import ClipTextEmbedder
        embedder = ClipTextEmbedder(
            device=str(device),
            model_name=cfg.task.dataset.get("clip_model_name",
                                            "openai/clip-vit-base-patch32"))

    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    builder = BenchmarkEnvBuilder(env_id=args.task, dataset=args.split,
                                  action_space="joint_angle",
                                  max_steps=args.max_steps)

    def pair(front, wrist, caption):
        f = np.asarray(front, np.uint8)
        w = np.asarray(wrist, np.uint8)
        return label(np.concatenate([f, w], axis=1), caption)

    def rollout(seed, abort_after=None):
        """One full episode. Returns (outcome, steps, frames)."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        env = builder.make_env_for_episode(args.episode)
        obs, info = env.reset()
        goal = info["task_goal"][0]

        frames = []
        # the conditioning video: everything already in the buffer at reset
        for i, (f, w) in enumerate(zip(obs["front_rgb_list"],
                                       obs["wrist_rgb_list"])):
            frames.append(pair(f, w,
                               f"conditioning {i+1}/{len(obs['front_rgb_list'])}"))

        text_emb = None
        if include_text:
            e = embedder.embed_texts([goal])
            if isinstance(e, torch.Tensor):
                e = e.detach().cpu().numpy()
            text_emb = e[0].astype(np.float32)

        imgs, states = HistoryBuf(obs_h), HistoryBuf(obs_h)

        def push_obs(o):
            imgs.push(np.concatenate([chw01(o["front_rgb_list"][-1]),
                                      chw01(o["wrist_rgb_list"][-1])], axis=0))
            js = np.asarray(o["joint_state_list"][-1], np.float32).reshape(-1)
            gr = np.asarray(o["gripper_state_list"][-1], np.float32).reshape(-1)
            states.push(np.concatenate([js, [gr.mean()]]).astype(np.float32))

        push_obs(obs)
        outcome, steps, done = "unknown", 0, False
        while not done:
            din = tf.transform_in({"state": states.stacked()})
            batch = {"image": torch.from_numpy(imgs.stacked()[None]).to(device),
                     "state": torch.from_numpy(
                         din["state"][None].astype(np.float32)).to(device)}
            if include_text:
                te = np.tile(text_emb[None], (obs_h, 1)).astype(np.float32)
                batch["text_emb"] = torch.from_numpy(te[None]).to(device)
            with torch.no_grad():
                out, _ = model.predict_action(batch, None)
            acts = tf.transform_out(
                {"action": out["action"][0].detach().cpu().numpy()})["action"]

            for a in acts[:exec_h]:
                obs, _, terminated, truncated, info = env.step(
                    a.astype(np.float32))
                steps += 1
                frames.append(pair(obs["front_rgb_list"][-1],
                                   obs["wrist_rgb_list"][-1],
                                   f"step {steps}  grip={a[7]:+.2f}"))
                if info is not None and info.get("status") == "error":
                    outcome, done = "error", True
                    break
                if terminated or truncated:
                    outcome, done = info.get("status", "unknown"), True
                    break
                push_obs(obs)
            # search-time escape hatch: a run that has already gone long is not
            # the short success we are hunting for, so stop paying for it.
            if not done and abort_after is not None and steps >= abort_after:
                outcome, done = "aborted", True
            if steps % 200 < exec_h:
                log(f"  step {steps}")
        env.close()
        return outcome, steps, frames, goal

    # The diffusion sampling is unseeded during normal eval, so a scored outcome
    # cannot be replayed exactly -- it can only be searched for. Each attempt
    # uses a different seed until the wanted outcome turns up.
    attempt, outcome, steps, frames, goal = 0, None, 0, None, ""
    while attempt < args.attempts:
        seed = None if args.seed is None else args.seed + attempt
        abort = args.abort_after if (args.want and args.want != "timeout") else None
        outcome, steps, frames, goal = rollout(seed, abort_after=abort)
        log(f"attempt {attempt+1}/{args.attempts} (seed={seed}): "
            f"{outcome} ({steps} steps)")
        attempt += 1
        if not args.want or outcome == args.want:
            break
    log("goal:", goal)
    if args.want and outcome != args.want:
        log(f"WARNING: never produced '{args.want}' in {args.attempts} attempts; "
            f"writing the last one ({outcome}) instead")
    log(f"episode {args.episode}: {outcome} ({steps} steps)")

    # tail card so the outcome is readable without scrubbing to the last frame
    for _ in range(args.fps):
        frames.append(label(np.zeros_like(frames[-1]),
                            f"{outcome.upper()} after {steps} steps"))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(
        args.out,
        f"{args.task}_{args.split}_ep{args.episode}_"
        f"{args.ckpt.replace('.pth','')}_{outcome}.mp4")
    import imageio.v2 as imageio
    imageio.mimwrite(path, frames, fps=args.fps, quality=7,
                     macro_block_size=1)
    log(f"wrote {path}  ({len(frames)} frames, {os.path.getsize(path)/1e6:.1f} MB)")
    print(f"OUTCOME={outcome} STEPS={steps} PATH={path}")


if __name__ == "__main__":
    main()
