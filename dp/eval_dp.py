"""Evaluate a trained DP checkpoint on a RoboMME task (--task, --split).

Loop per their scripts/evaluation.py (status accounting at :103-111); DP
inference per serve_dp.py conventions: rolling obs_horizon buffers padded by
repeating the first entry, 6-channel image stack, DataTransform round-trip
(training=False), EMA weights, CenterCrop applied inside the model (eval()).

APIs verified against the cloned repo 2026-08-06:
  RoboMMEDataset stats saved via eval_envs.utils.normalize.save -> stats.json
  DataTransform(norm_stats, norm_type, mask, use_delta_action, training)
  ClipTextEmbedder(device=..., model_name=...).embed_texts([str]) -> (1,512)
  model.predict_action(obs_dict, policy_info) -> ({"action": (B,exec,8)}, _)

Sharding: --shard/--num-shards splits the 50 test episodes across processes.
NEVER reads info["simple_subgoal_online"]/grounded_subgoal_online — those
always-present info fields are an oracle-plan leak.
"""
import argparse, json, os, sys

_HF = os.path.expanduser("~/.cache/huggingface")
os.environ.setdefault("HF_HOME", _HF)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_HF, "datasets"))

REPO = os.environ.get("DP_REPO")
assert REPO, "set DP_REPO to your clone of github.com/RoboMME/DP (pin f333cd6)"
sys.path.insert(0, REPO)

import numpy as np
import torch


def load_policy(run_dir, ckpt, device, use_ema=True):
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from eval_envs.utils.normalize import load as load_norm_stats
    from eval_envs.utils.transform import DataTransform

    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    model = instantiate(cfg.model)
    ck = torch.load(os.path.join(run_dir, ckpt), map_location="cpu",
                    weights_only=False)
    key = "model_ema" if (use_ema and "model_ema" in ck) else "model"
    model.load_state_dict(ck[key])
    model.to(device).eval()

    stats = load_norm_stats(run_dir, filename="stats.json")
    tf = DataTransform(norm_stats=stats,
                       norm_type=str(cfg.norm_type),
                       mask=None,
                       use_delta_action=bool(cfg.use_delta_action),
                       training=False)
    return model, tf, cfg


def chw01(img):
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[-1] == 3:
        a = a.transpose(2, 0, 1)
    a = a.astype(np.float32)
    return a / 255.0 if a.max() > 1.0 else a


class HistoryBuf:
    def __init__(self, n):
        self.n, self.buf = n, []

    def push(self, x):
        self.buf.append(x)
        self.buf = self.buf[-self.n:]

    def stacked(self):
        pad = self.n - len(self.buf)
        return np.stack([self.buf[0]] * pad + self.buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", required=True, help="e.g. ckpt_200000.pth")
    ap.add_argument("--task", default="MoveCube")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="checkpoint-selection sweeps use val; report test once")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=1300)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--raw-weights", action="store_true")
    ap.add_argument("--render-backend", default=None,
                    help="ManiSkill render backend. Leave unset for GPU. Pass "
                         "'cpu' on a pod whose /dev/dri render node is not "
                         "openable (see checks/check_vulkan.py) -- correct but "
                         "far slower.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.render_backend:
        # BenchmarkEnvBuilder hardcodes its gym.make kwargs, so the only way to
        # reach ManiSkill's render_backend argument without forking the pinned
        # benchmark clone is to wrap gym.make. setdefault, so an explicit
        # kwarg from the builder would still win.
        import gymnasium as gym
        _orig_make = gym.make

        def _make_with_backend(env_id, **kw):
            kw.setdefault("render_backend", args.render_backend)
            return _orig_make(env_id, **kw)

        gym.make = _make_with_backend

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf, cfg = load_policy(args.run_dir, args.ckpt, device,
                                 use_ema=not args.raw_weights)
    obs_h = int(cfg.obs_horizon)
    exec_h = int(cfg.action_exec_horizon)

    text_emb = None
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
    n = min(args.episodes, builder.get_episode_num())
    my_eps = list(range(n))[args.shard::args.num_shards]

    results = {}
    for ep_i in my_eps:
        env = builder.make_env_for_episode(ep_i)
        obs, info = env.reset()
        goal = info["task_goal"][0]
        if include_text:
            # embed PER EPISODE: VideoUnmask-family goals vary (query color);
            # single-goal tasks like MoveCube just recompute a constant
            e = embedder.embed_texts([goal])
            if isinstance(e, torch.Tensor):
                e = e.detach().cpu().numpy()
            text_emb = e[0].astype(np.float32)   # (512,)

        imgs, states = HistoryBuf(obs_h), HistoryBuf(obs_h)

        def push_obs(o):
            fused = np.concatenate([chw01(o["front_rgb_list"][-1]),
                                    chw01(o["wrist_rgb_list"][-1])], axis=0)
            js = np.asarray(o["joint_state_list"][-1], np.float32).reshape(-1)
            gr = np.asarray(o["gripper_state_list"][-1],
                            np.float32).reshape(-1)
            imgs.push(fused)
            # gripper scalar MUST match the converter default (--gripper-elem mean)
            states.push(np.concatenate([js, [gr.mean()]]).astype(np.float32))

        push_obs(obs)
        outcome, steps = "unknown", 0
        while True:
            din = tf.transform_in({"state": states.stacked()})  # normalize state
            batch = {
                "image": torch.from_numpy(imgs.stacked()[None]).to(device),
                "state": torch.from_numpy(
                    din["state"][None].astype(np.float32)).to(device),
            }
            if include_text:
                te = np.tile(text_emb[None], (obs_h, 1)).astype(np.float32)
                batch["text_emb"] = torch.from_numpy(te[None]).to(device)
            with torch.no_grad():
                out, _ = model.predict_action(batch, None)
            acts = out["action"][0].detach().cpu().numpy()
            acts = tf.transform_out({"action": acts})["action"]

            done = False
            for a in acts[:exec_h]:
                obs, _, terminated, truncated, info = env.step(
                    a.astype(np.float32))
                steps += 1
                if info is not None and info.get("status") == "error":
                    outcome, done = "error", True
                    break
                if terminated or truncated:
                    outcome, done = info.get("status", "unknown"), True
                    break
                push_obs(obs)
            if done:
                break
        env.close()
        results[ep_i] = outcome
        print(f"episode {ep_i}: {outcome} ({steps} steps)", flush=True)

    succ = sum(v == "success" for v in results.values())
    print(f"shard {args.shard}/{args.num_shards}: {succ}/{len(results)} "
          f"success ({succ/max(len(results),1):.1%})")
    out = args.out or os.path.join(
        args.run_dir, f"eval_{args.task}_{args.split}_{args.ckpt.replace('.pth','')}"
        f"_shard{args.shard}.json")
    with open(out, "w") as fh:
        json.dump({"task": args.task, "split": args.split, "ckpt": args.ckpt,
                   "results": results, "success": succ, "n": len(results)}, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
