"""Gate: the eval-time STREAMING TTT must equal the training-time rollout.

RMBench landmine: at deploy the memory write has to be chunked exactly as it
was in training. Per-frame stepping is a different operator — on swap_blocks
it diverged by 92% (exp/ttt/check_deploy_chunked.py). eval_stage2_mc.py feeds
frames one at a time as the env yields them, so this script proves that path
reproduces the batched rollout the head was trained against.

Compares, on real cached MoveCube features:
  reference : ttt(x[:, :t])[0, -1]        one batched call, training operator
  streaming : ChunkedTTT.push(...) x t; .read()
at every t where the streaming buffer has just flushed (t % chunk == 0), and
also reports the WORST-CASE mid-chunk lag (t between flushes), which is the
staleness training also has (reads use chunk-start weights).

PASS = flush-aligned readouts match to <1e-4 relative.
"""
import argparse
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model_mc import DPMemoryPolicyMC  # noqa: E402
from eval_stage2 import ChunkedTTT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="cache/feats.npz")
    ap.add_argument("--ckpt", default=None,
                    help="stage-2 arm ckpt; random init if omitted")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=16)
    args = ap.parse_args()

    dev = torch.device("cpu")
    d = np.load(args.features, allow_pickle=True)
    x_np = d["feats"][d["ep_from"][0]:d["ep_from"][0] + args.frames].astype(np.float32)

    # Take the architecture from the checkpoint rather than assuming defaults --
    # a VideoUnmask arm is trained --with-text (d_feat 648), and the cache holds
    # only the 136-d [vis 128 | state 8] part, with the per-episode CLIP vector
    # in text_embs. Hardcoding 136 here made this gate unrunnable against any
    # text-conditioned arm: load_state_dict just threw a wall of size mismatches.
    ck = None
    d_feat, chunk = 136, args.chunk
    mk = dict(action_dim=8, horizon=16, n_obs_steps=2, n_action_steps=8,
              use_ttt=True, inject="concat",
              ttt=dict(d_k=128, d_v=128, d_hidden=512, d_out=256, chunk=chunk))
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        ta = ck["args"]
        d_feat = int(ck.get("d_feat", 136))
        chunk = ta["chunk"]
        mk = dict(action_dim=8, horizon=ta["horizon"],
                  n_obs_steps=ta["n_obs_steps"],
                  n_action_steps=ta["n_action_steps"],
                  use_ttt=ta["use_ttt"], inject=ta["inject"],
                  ttt=dict(d_k=ta["d_k"], d_v=ta["d_v"], d_hidden=ta["d_hidden"],
                           d_out=ta["d_out"], chunk=chunk))
        if not ta["use_ttt"]:
            sys.exit("this arm was trained --no-ttt; there is no memory to gate")

    if d_feat > x_np.shape[1]:
        # rebuild the eval-time layout [vis 128 | state 8 | text 512]; frames
        # sliced above all belong to episode 0, so its embedding applies to all
        text = d["text_embs"][0].astype(np.float32)
        x_np = np.concatenate(
            [x_np, np.tile(text, (len(x_np), 1))], axis=1).astype(np.float32)
    assert x_np.shape[1] == d_feat, \
        f"feature width {x_np.shape[1]} != checkpoint d_feat {d_feat}"

    model = DPMemoryPolicyMC(d_feat=d_feat, action_dim=mk["action_dim"],
                             horizon=mk["horizon"],
                             n_obs_steps=mk["n_obs_steps"],
                             n_action_steps=mk["n_action_steps"],
                             use_ttt=mk["use_ttt"],
                             ttt_kwargs=mk["ttt"],
                             inject=mk["inject"]).to(dev)
    if ck is not None:
        model.load_state_dict(ck["ema_model"])
    model.eval()
    args.chunk = chunk

    # Reference: the training operator over the whole sequence — m[t] is the
    # readout for frame t at its chunk-start weights. Streaming must match it
    # at EVERY t, not merely at flush boundaries.
    with torch.no_grad():
        ref_all = model.ttt(torch.from_numpy(x_np[None]).to(dev),
                            torch.ones(1, args.frames))[0][0]      # (T, d_out)
        stream = ChunkedTTT(model, dev)
        worst, worst_t = 0.0, -1
        for t in range(args.frames):
            got = stream.read(x_np[t])       # read before push, as in eval
            stream.push(x_np[t])
            rel = float((ref_all[t] - got).norm() / ref_all[t].norm().clamp(min=1e-8))
            if rel > worst:
                worst, worst_t = rel, t

    print(f"frames={args.frames} chunk={args.chunk}")
    print(f"  worst rel-diff over ALL frames : {worst:.2e} (at t={worst_t})")
    ok = worst < 1e-4
    print("PASS — streaming write == training operator" if ok else
          "FAIL — streaming TTT diverges from the trained operator")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
