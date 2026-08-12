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

    model = DPMemoryPolicyMC(d_feat=136, action_dim=8, horizon=16, n_obs_steps=2,
                             n_action_steps=8, use_ttt=True,
                             ttt_kwargs=dict(d_k=128, d_v=128, d_hidden=512,
                                             d_out=256, chunk=args.chunk),
                             inject="concat").to(dev)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["ema_model"])
    model.eval()

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
