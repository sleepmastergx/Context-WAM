"""Closed-loop eval of the stage-2 arms on RoboMME MoveCube.

Both arms share this script; arm 2 simply has use_ttt=False, so the memory
path is skipped and the head sees only the current 2-frame obs window.

The memory-critical part: the conditioning VIDEO must reach the TTT state.
The env plays the demonstration itself during the first phase of the episode
and returns EVERY rendered frame in obs["front_rgb_list"] / ["wrist_rgb_list"]
(these are lists precisely because of video conditioning). So we push all
frames of every list through the frozen encoder into the TTT, and read m_t at
the latest frame — mirroring training, where the rollout covered the video
prefix and windows were read at execution frames.

CHUNKING (RMBench landmine): the TTT write must be chunked exactly as in
training (chunk=16). Per-frame stepping is a DIFFERENT operator — on RMBench
it diverged by 92%. Frames are therefore buffered and flushed in chunk-sized
groups; check_chunk_equiv.py gates this against the training-time rollout.

Never reads info["simple_subgoal*"] — that is the benchmark's oracle plan.
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_DP_REPO = os.environ.get("DP_REPO")
assert _DP_REPO, "set DP_REPO to your clone of github.com/RoboMME/DP"
sys.path.insert(0, _DP_REPO)
_HF = os.path.expanduser("~/.cache/huggingface")
os.environ.setdefault("HF_HOME", _HF)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_HF, "datasets"))

from model_mc import DPMemoryPolicyMC  # noqa: E402

N_STATE = 8


def chw01(img):
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[-1] == 3:
        a = a.transpose(2, 0, 1)
    a = a.astype(np.float32)
    return a / 255.0 if a.max() > 1.0 else a


class FrozenEncoder:
    """Stage-1 image_encoder + image_pool under the deterministic eval
    transform — the same operator cache_features_mc.py used."""

    def __init__(self, run_dir, ckpt, device):
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
        cfg.device = str(device)
        m = instantiate(cfg.model)
        sd = torch.load(os.path.join(run_dir, ckpt), map_location="cpu",
                        weights_only=False)
        m.load_state_dict(sd.get("model_ema") or sd["model"], strict=True)
        self.m = m.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, fused_chw):
        """fused (6,H,W) float in [0,1] -> (128,) visual feature."""
        x = torch.from_numpy(fused_chw).to(self.device).view(2, 3, 256, 256)
        x = self.m.img_tf_val(x)
        return self.m.image_pool(self.m.image_encoder(x)).view(-1).cpu().numpy()


class ChunkedTTT:
    """Streaming TTT whose write AND read match the training operator exactly.

    Training (ttt_cell.forward): within a chunk every frame is READ at the
    chunk-start weights, and the write (inner grads, momentum, forget) is
    applied ONCE per chunk using all frames in it. So streaming must:
      * read every frame through the CURRENT chunk-start weights — including
        the very first chunk, which reads through the learned W1_0/W2_0
        (returning zeros there was wrong: rel-diff 1.0 vs training);
      * buffer frames and apply the identical write when `chunk` accumulate.
    check_chunk_equiv.py gates both halves against a batched rollout.
    """

    def __init__(self, model, device):
        t = model.ttt
        self.ttt = t
        self.chunk = t.chunk
        self.device = device
        self.buf = []
        # m for the current frame; set on every consume, initialised so a
        # read before any frame arrives still yields the learned-init readout
        self.m_cur = torch.zeros(t.d_out, device=device)
        B = 1
        self.W1 = t.W1_0.unsqueeze(0).expand(B, -1, -1).clone()
        self.W2 = t.W2_0.unsqueeze(0).expand(B, -1, -1).clone()
        self.M1 = torch.zeros_like(self.W1)
        self.M2 = torch.zeros_like(self.W2)

    @torch.no_grad()
    def push(self, feat_vec):
        self.buf.append(feat_vec)
        if len(self.buf) >= self.chunk:
            self._write()

    @torch.no_grad()
    def _x(self, arr):
        return self.ttt.ln_in(torch.from_numpy(np.asarray(arr)).float()
                              .to(self.device).view(1, -1, self.ttt.ln_in.normalized_shape[0]))

    @torch.no_grad()
    def _write(self):
        t = self.ttt
        x = self._x(np.stack(self.buf))                    # (1,C,d_in)
        kc, vc = t.to_k(x), t.to_v(x)
        eta = torch.nn.functional.softplus(t.to_eta(x)).squeeze(-1)
        alpha = torch.sigmoid(t.to_alpha(x)).squeeze(-1)
        beta = torch.sigmoid(t.beta_logit)

        dPre, dY, h, kk, _ = t._inner_grad(self.W1, self.W2, kc, vc)
        C = x.shape[1]
        w = eta.unsqueeze(-1)
        denom = float(C)
        g1 = torch.einsum("bch,bck->bhk", dPre * w, kk) / denom
        g2 = torch.einsum("bcv,bch->bvh", dY * w, h) / denom
        gn = (g1.flatten(1).pow(2).sum(1) + g2.flatten(1).pow(2).sum(1) + 1e-12).sqrt()
        scale = (t.max_write / gn).clamp(max=1.0).view(1, 1, 1)
        g1, g2 = g1 * scale, g2 * scale

        a = alpha.mean(dim=1).view(1, 1, 1)
        self.M1 = beta * self.M1 - g1
        self.M2 = beta * self.M2 - g2
        self.W1 = (1.0 - a) * self.W1 + self.M1
        self.W2 = (1.0 - a) * self.W2 + self.M2
        self.buf.clear()

    @torch.no_grad()
    def read(self, feat_vec):
        """m_t for the CURRENT frame at chunk-start weights (as in training)."""
        t = self.ttt
        q = t.to_q(self._x(feat_vec[None]))                # (1,1,d_k)
        preq = torch.einsum("bhk,bck->bch", self.W1, q)
        hq = torch.nn.functional.gelu(preq)
        mq = torch.einsum("bvh,bch->bcv", self.W2, hq)
        return t.ln(t.readout(mq))[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="stage-2 arm run dir")
    ap.add_argument("--ckpt", default="600.ckpt")
    ap.add_argument("--stage1-dir", required=True,
                    help="stage-1 DP run dir (frozen encoder + stats.json)")
    ap.add_argument("--stage1-ckpt", default="ckpt_200000.pth")
    ap.add_argument("--task", default="MoveCube")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
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
        # Same wrap as dp/eval_dp.py: BenchmarkEnvBuilder hardcodes its
        # gym.make kwargs, so this is the only way to reach ManiSkill's
        # render_backend without forking the pinned benchmark clone.
        import gymnasium as gym
        _orig_make = gym.make

        def _make_with_backend(env_id, **kw):
            kw.setdefault("render_backend", args.render_backend)
            return _orig_make(env_id, **kw)

        gym.make = _make_with_backend

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(args.run_dir, "checkpoints", args.ckpt),
                    map_location="cpu", weights_only=False)
    targs = ck["args"]
    d_feat = int(ck.get("d_feat", 136))     # 648 when trained --with-text
    with_text = d_feat > 136
    model = DPMemoryPolicyMC(
        d_feat=d_feat, action_dim=8, horizon=targs["horizon"],
        n_obs_steps=targs["n_obs_steps"], n_action_steps=targs["n_action_steps"],
        use_ttt=targs["use_ttt"],
        ttt_kwargs=dict(d_k=targs["d_k"], d_v=targs["d_v"],
                        d_hidden=targs["d_hidden"], d_out=targs["d_out"],
                        chunk=targs["chunk"]),
        inject=targs["inject"]).to(device)
    model.load_state_dict(ck["model" if args.raw_weights else "ema_model"])
    model.eval()
    obs_h, exec_h = targs["n_obs_steps"], targs["n_action_steps"]
    print(f"arm: use_ttt={targs['use_ttt']} inject={targs['inject']} "
          f"ckpt={args.ckpt} weights={'raw' if args.raw_weights else 'ema'}",
          flush=True)

    enc = FrozenEncoder(args.stage1_dir, args.stage1_ckpt, device)
    embedder = None
    if with_text:
        from eval_envs.utils.clip_model import ClipTextEmbedder
        embedder = ClipTextEmbedder(device=str(device))
        print("query-text conditioning ON (d_feat=%d)" % d_feat, flush=True)
    from eval_envs.utils.normalize import load as load_norm_stats
    from eval_envs.utils.transform import DataTransform
    stats = load_norm_stats(args.stage1_dir, filename="stats.json")
    tf = DataTransform(norm_stats=stats, norm_type="minmax", mask=None,
                       use_delta_action=False, training=False)

    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    builder = BenchmarkEnvBuilder(env_id=args.task, dataset=args.split,
                                  action_space="joint_angle",
                                  max_steps=args.max_steps)
    n = min(args.episodes, builder.get_episode_num())
    my_eps = list(range(n))[args.shard::args.num_shards]

    results, frame_counts = {}, {}
    for ep_i in my_eps:
        env = builder.make_env_for_episode(ep_i)
        obs, info = env.reset()
        text_emb = None
        if with_text:
            # PER EPISODE: the query color varies across VideoUnmask episodes.
            # Must match training layout [vis 128 | state 8 | text 512].
            e = embedder.embed_texts([info["task_goal"][0]])
            if isinstance(e, torch.Tensor):
                e = e.detach().cpu().numpy()
            text_emb = np.asarray(e[0], np.float32)
        mem = ChunkedTTT(model, device) if model.use_ttt else None
        feat_hist, nframes = [], 0

        def consume(o, all_frames):
            """Push observation frames into the feature history (+TTT)."""
            nonlocal nframes
            fl, wl = o["front_rgb_list"], o["wrist_rgb_list"]
            js_l, gr_l = o["joint_state_list"], o["gripper_state_list"]
            idxs = range(len(fl)) if all_frames else [len(fl) - 1]
            for i in idxs:
                fused = np.concatenate([chw01(fl[i]), chw01(wl[i])], axis=0)
                vis = enc(fused)
                j = min(i, len(js_l) - 1)
                js = np.asarray(js_l[j], np.float32).reshape(-1)
                gr = np.asarray(gr_l[j], np.float32).reshape(-1)
                st = np.concatenate([js, [gr.mean()]]).astype(np.float32)
                st = tf.transform_in({"state": st[None]})["state"][0]
                parts = [vis, st] if text_emb is None else [vis, st, text_emb]
                f = np.concatenate(parts).astype(np.float32)
                feat_hist.append(f)
                if mem is not None:
                    # read BEFORE push: training reads frame t at the weights
                    # in force when t arrives, i.e. before t's own chunk writes
                    m_cur = mem.read(f)
                    mem.push(f)
                    mem.m_cur = m_cur
                nframes += 1

        # reset() may already carry the conditioning video in the lists
        consume(obs, all_frames=True)

        outcome, steps = "unknown", 0
        while True:
            win = [feat_hist[max(0, len(feat_hist) - obs_h + k)]
                   for k in range(obs_h)]
            gc = torch.from_numpy(np.concatenate(win)[None]).float().to(device)
            m = mem.m_cur[None] if mem is not None else None
            with torch.no_grad():
                _, acts = model.predict_action(gc, m)
            acts = acts[0].cpu().numpy()
            acts = tf.transform_out({"action": acts})["action"]

            done = False
            for a in acts[:exec_h]:
                obs, _, terminated, truncated, info = env.step(a.astype(np.float32))
                steps += 1
                if info is not None and info.get("status") == "error":
                    outcome, done = "error", True
                    break
                if terminated or truncated:
                    outcome, done = info.get("status", "unknown"), True
                    break
                consume(obs, all_frames=True)
            if done:
                break
        env.close()
        results[ep_i] = outcome
        frame_counts[ep_i] = nframes
        print(f"episode {ep_i}: {outcome} ({steps} steps, {nframes} frames "
              f"into memory)", flush=True)

    succ = sum(v == "success" for v in results.values())
    print(f"shard {args.shard}/{args.num_shards}: {succ}/{len(results)} "
          f"success ({succ/max(len(results),1):.1%})")
    arm = pathlib.Path(args.run_dir).name
    out = args.out or os.path.join(
        args.run_dir, f"eval_{args.task}_{args.split}_"
        f"{args.ckpt.replace('.ckpt','')}_shard{args.shard}.json")
    with open(out, "w") as fh:
        json.dump({"task": args.task, "split": args.split, "arm": arm,
                   "ckpt": args.ckpt, "results": results,
                   "frames": frame_counts, "success": succ,
                   "n": len(results)}, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
