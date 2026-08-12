"""Cache frozen stage-1 features over ALL MoveCube frames (video + execution).

Encoder = image_encoder + image_pool from the val-winning ckpt_200000 EMA
weights, applied through the deterministic eval transform (CenterCrop 230 +
Normalize +-0.5) — byte-identical to what the deployed policy computes.

Output feats_mc.npz:
  feats    (N, 136) float32   2 cams x 64 spatial-softmax dims + 8 state dims
  actions  (N, 8)   float32   raw (unnormalized)
  states   (N, 8)   float32   raw
  is_demo  (N,)     bool      True = conditioning-video frame
  ep_from, ep_to (E,) int64   episode row ranges
  seeds    (E,)     int64

The TTT arm consumes the full stream from t=0 (memory sees the video); the
control head only ever reads execution-frame windows. Action loss is masked
to execution frames by the dataset, not here.
"""
import argparse, glob, io, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch

REPO = os.environ.get("DP_REPO")
assert REPO, "set DP_REPO to your clone of github.com/RoboMME/DP"
sys.path.insert(0, REPO)


def load_frozen_encoder(run_dir, ckpt_name, device):
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    cfg.device = str(device)
    model = instantiate(cfg.model)
    ckpt = torch.load(os.path.join(run_dir, ckpt_name), map_location="cpu",
                      weights_only=False)
    sd = ckpt.get("model_ema") or ckpt["model"]
    missing, unexpected = model.load_state_dict(sd, strict=True), None
    model.to(device).eval()
    return model


def decode_episode(path):
    df = pd.read_parquet(path, columns=["image", "wrist_image", "state",
                                        "actions", "is_demo"])
    T = len(df)
    imgs = np.empty((T, 6, 256, 256), np.uint8)
    for i in range(T):
        pair = []
        for col in ("image", "wrist_image"):
            cell = df[col].iloc[i]
            buf = cell["bytes"] if isinstance(cell, dict) else cell
            a = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            pair.append(np.ascontiguousarray(a[:, :, ::-1].transpose(2, 0, 1)))
        imgs[i] = np.concatenate(pair, axis=0)
    return (imgs,
            np.stack(df["state"].to_numpy()).astype(np.float32),
            np.stack(df["actions"].to_numpy()).astype(np.float32),
            df["is_demo"].to_numpy().astype(bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="stage-1 DP run dir")
    ap.add_argument("--ckpt", default="ckpt_200000.pth")
    ap.add_argument("--data-dir", required=True,
                    help="<lerobot-root>/data/chunk-000 of the converted task")
    ap.add_argument("--out", default="cache/feats.npz")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu",
                    help="cpu works but is slow; use a GPU batch job")
    ap.add_argument("--with-text", action="store_true",
                    help="ALSO cache per-episode CLIP text embeddings of the "
                         "goal strings (REQUIRED for VideoUnmask-family tasks "
                         "— the query color lives in the goal; without this "
                         "the stage-2 arms are blind to the query)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model = load_frozen_encoder(args.run_dir, args.ckpt, device)
    print(f"encoder loaded (EMA) on {device}", flush=True)

    paths = sorted(glob.glob(os.path.join(args.data_dir, "episode_*.parquet")))
    assert len(paths) == 100, f"expected 100 episodes, found {len(paths)}"

    text_embs = None
    if args.with_text:
        # goals come from the converter's meta/episodes.jsonl, in episode order
        import json
        root = os.path.dirname(os.path.dirname(os.path.abspath(args.data_dir)))
        goals = {}
        with open(os.path.join(root, "meta", "episodes.jsonl")) as fh:
            for line in fh:
                rec = json.loads(line)
                goals[int(rec["episode_index"])] = rec["tasks"][0]
        assert len(goals) == len(paths), (len(goals), len(paths))
        from eval_envs.utils.clip_model import ClipTextEmbedder
        embedder = ClipTextEmbedder(device=str(device))
        e = embedder.embed_texts([goals[i] for i in range(len(paths))])
        if isinstance(e, torch.Tensor):
            e = e.detach().cpu().numpy()
        text_embs = np.asarray(e, np.float32)               # (E, 512)
        print(f"text embeddings: {text_embs.shape}, "
              f"{len(set(goals.values()))} distinct goals", flush=True)

    feats, states_all, actions_all, demo_all = [], [], [], []
    ep_from, ep_to = [], []
    row = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for ep_i, (imgs, states, actions, is_demo) in enumerate(
                pool.map(decode_episode, paths)):
            T = len(imgs)
            vis = np.empty((T, 128), np.float32)
            with torch.inference_mode():
                for s in range(0, T, args.batch):
                    chunk = torch.from_numpy(imgs[s:s + args.batch]).to(device)
                    b = chunk.shape[0]
                    x = chunk.view(b * 2, 3, 256, 256).float().div_(255.0)
                    x = model.img_tf_val(x)
                    f = model.image_pool(model.image_encoder(x))   # (b*2, 64)
                    vis[s:s + b] = f.view(b, 128).cpu().numpy()
            feats.append(np.concatenate([vis, states], axis=1))    # (T, 136)
            states_all.append(states); actions_all.append(actions)
            demo_all.append(is_demo)
            ep_from.append(row); ep_to.append(row + T); row += T
            if (ep_i + 1) % 10 == 0:
                print(f"  {ep_i+1}/100 episodes, {row} frames, "
                      f"{time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = dict(
        feats=np.concatenate(feats),
        states=np.concatenate(states_all),
        actions=np.concatenate(actions_all),
        is_demo=np.concatenate(demo_all),
        ep_from=np.asarray(ep_from, np.int64),
        ep_to=np.asarray(ep_to, np.int64),
        encoder_ckpt=os.path.join(args.run_dir, args.ckpt),
    )
    if text_embs is not None:
        payload["text_embs"] = text_embs      # dataset appends per-frame
    np.savez_compressed(args.out, **payload)
    d = np.load(args.out, allow_pickle=True)
    print(f"WROTE {args.out}: feats {d['feats'].shape}, "
          f"{int(d['is_demo'].sum())} video / {int((~d['is_demo']).sum())} exec "
          f"frames, {len(d['ep_from'])} episodes, {time.time()-t0:.0f}s total",
          flush=True)


if __name__ == "__main__":
    main()
