"""MoveCube (LeRobot) -> FastWAM window cache.

Produces one record per WINDOW:
    latents   [48, 3, 16, 32]  Wan2.2 VAE over 9 sampled frames of the
                               2-camera mosaic (front | wrist, 256x512)
    action    [32, 8]          the 32-action chunk for this window
    proprio   [8]              state at the window's first frame
    is_exec   bool             window's action chunk starts at/after exec_start
    ep, start                  provenance

Layout facts taken from their loader, not guessed:
  * multi-camera is tiled SPATIALLY into one image
    (robot_video_dataset.py:288-321: "horizontal" concatenates along width;
    their "robotwin" mode mosaics 3 cams into 384x320). Two cams at 256x256
    -> 256x512, both dims multiples of 16 as the model asserts.
  * a window spans num_frames=33 raw frames; every 4th is sampled for video
    (action_video_freq_ratio=4) -> 9 video frames -> VAE temporal /4 ->
    latent_t = (9-1)//4 + 1 = 3.
  * 33 % 4 == 1 and 9 % 4 == 1, satisfying their T % 4 == 1 checks.

STRIDE 1, matching their own `global_sample_stride: 1`: cache EVERY window
start. R (replan_steps) is undecided and the TTT arm's window spacing must
equal it, so a coarse cache would silently restrict both R and the chain
alignment. ~362 windows/episode -> ~36k windows -> ~5.3 GiB, and one pass then
serves every R and every alignment forever.

The CONTROL arm needs nothing more than this: with no memory each window's loss
is independent, so it trains on random windows exactly as FastWAM natively does
and R stays a pure deploy knob. Only the TTT arm needs ordered chains.
"""
import argparse
import glob
import io
import json
import os
import sys

import numpy as np

WINDOW_FRAMES = 33
VIDEO_RATIO = 4          # every 4th frame -> 9 video frames
ACTION_HORIZON = 32


def window_starts(n_frames: int, stride: int):
    return list(range(0, max(0, n_frames - WINDOW_FRAMES) + 1, stride))


def video_frame_indices(start: int):
    """The 9 frames the VAE sees for a window starting at `start`."""
    idx = list(range(start, start + WINDOW_FRAMES, VIDEO_RATIO))
    assert len(idx) == 9 and (len(idx) - 1) % VIDEO_RATIO == 0, idx
    return idx


def mosaic(front_chw, wrist_chw):
    """Two cameras -> one frame, concatenated along WIDTH (their 'horizontal')."""
    return np.concatenate([front_chw, wrist_chw], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot-root", required=True,
                    help="LeRobot-format MoveCube dataset root")
    ap.add_argument("--out", default="data/movecube_fastwam")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-vae", action="store_true",
                    help="enumerate windows and shapes only; no GPU needed")
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--first-episode", type=int, default=0,
                    help="skip episodes below this global index (extend an "
                         "existing cache without re-encoding it; ep numbering "
                         "stays global, so ep0100.npz still means episode 100)")
    ap.add_argument("--episode-offset", type=int, default=0,
                    help="add this to every episode id (npz name + index "
                         "'ep'). For converting a STANDALONE extension "
                         "dataset whose parquet 0..N-1 are global episodes "
                         "offset..offset+N-1 (e.g. a generated-only LeRobot "
                         "set extending an existing 100-episode cache).")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="episode-sharded VAE pass; one process per GPU")
    args = ap.parse_args()
    # the VAE branch chdirs into the fastwam root; keep outputs anchored
    args.out = os.path.abspath(args.out)
    args.lerobot_root = os.path.abspath(args.lerobot_root)

    import pandas as pd
    from PIL import Image

    files = sorted(glob.glob(f"{args.lerobot_root}/data/chunk-000/episode_*.parquet"))
    if args.limit_episodes:
        files = files[:args.limit_episodes]
    all_files = list(files)
    files = files[args.first_episode:]
    files = files[args.shard::args.num_shards]   # episodes are independent
    if not files:
        sys.exit(f"no parquets under {args.lerobot_root}")
    os.makedirs(args.out, exist_ok=True)

    vae = None
    if not args.no_vae:
        if not os.environ.get("SLURM_JOB_ID"):
            sys.exit("VAE encoding needs a GPU: run inside a Slurm job "
                     "(standing rule: no ad-hoc GPU use), or pass --no-vae")
        # Load EXACTLY as their scripts/precompute_video_latents.py:381-392
        # does — resolve the registered config, then _load_registered_model.
        # (WanVideoVAE38 has no from_pretrained; that guess cost job 1391.)
        import pathlib
        fw_root = os.environ.get(
            "FASTWAM_ROOT",
            str(pathlib.Path(__file__).resolve().parents[1] / "third_party" / "fastwam"))
        sys.path.insert(0, os.path.join(fw_root, "src"))
        import torch
        from fastwam.models.wan22.helpers.loader import (_load_registered_model,
                                                         _resolve_configs)
        # their config paths are RELATIVE to the fastwam repo root
        os.chdir(fw_root)
        _, _, vae_config, _ = _resolve_configs(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
            redirect_common_files=True)
        vae_config.download_if_necessary()
        vae = _load_registered_model(vae_config.path, "wan_video_vae",
                                     torch_dtype=torch.bfloat16,
                                     device="cuda").eval()
        z_dim = int(vae.model.z_dim)
        tdf = int(vae.temporal_downsample_factor)
        usf = int(vae.upsampling_factor)
        exp_shape = (z_dim, (9 - 1) // tdf + 1, 256 // usf, 512 // usf)
        print(f"VAE loaded: z_dim {z_dim}, temporal /{tdf}, spatial /{usf} "
              f"-> expected latent {exp_shape}", flush=True)

    index, n_win, n_exec = [], 0, 0
    for fp in files:
        ep_i = all_files.index(fp) + args.episode_offset  # global episode id, shard-invariant
        df = pd.read_parquet(fp, columns=["image", "wrist_image", "state",
                                          "actions", "is_demo"])
        T = len(df)
        is_demo = df["is_demo"].to_numpy().astype(bool)
        exec_start = int(np.argmin(is_demo)) if (~is_demo).any() else T
        states = np.stack(df["state"].to_numpy()).astype(np.float32)
        actions = np.stack(df["actions"].to_numpy()).astype(np.float32)

        starts = window_starts(T, args.stride)
        ep_latents = []
        for s in starts:
            vidx = video_frame_indices(s)
            if not args.no_vae:
                frames = []
                for i in vidx:
                    fr = []
                    for col in ("image", "wrist_image"):
                        cell = df[col].iloc[i]
                        buf = cell["bytes"] if isinstance(cell, dict) else cell
                        a = np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))
                        fr.append(a.transpose(2, 0, 1))
                    frames.append(mosaic(fr[0], fr[1]))
                v = np.stack(frames).astype(np.float32) / 127.5 - 1.0   # [-1,1]
                import torch
                with torch.inference_mode():
                    # their call: encode(videos, device=str(device), tiled=False)
                    # their VAE wants [B, C, T, H, W]; `v` is (T, C, H, W).
                    # Passing [B, T, C, H, W] made the encoder read the 9 frames
                    # as channels (it patchifies 2x2, so 3->12 expected but it
                    # saw 9*4=36) -- the failure in job 1392.
                    vt = torch.from_numpy(v).permute(1, 0, 2, 3)[None].to(
                        device="cuda", dtype=torch.bfloat16)
                    z = vae.encode(vt, device="cuda", tiled=False)
                if tuple(z.shape) != (1, *exp_shape):
                    raise ValueError(
                        f"VAE output {tuple(z.shape)} != expected {(1,*exp_shape)}")
                # numpy has no bfloat16 -> store float16. For VAE latents
                # (|x| well under fp16 range) fp16 has MORE mantissa than bf16,
                # so this loses nothing; gpu_cache casts back to bf16 on load.
                ep_latents.append(z[0].detach().to("cpu", torch.float16).numpy())
            # action chunk: clamp at the episode end (repeat-last padding)
            a_idx = np.clip(np.arange(s, s + ACTION_HORIZON), 0, T - 1)
            is_exec = s >= exec_start
            index.append({"ep": ep_i, "start": s, "is_exec": bool(is_exec),
                          "exec_start": exec_start, "n_frames": T})
            n_win += 1
            n_exec += int(is_exec)
            if args.no_vae and s == starts[0] and ep_i == 0:
                print(f"  window {s}: video frames {vidx} -> mosaic "
                      f"(9, 3, 256, 512) -> latent [48, 3, 16, 32]")
                print(f"  action chunk idx {a_idx[0]}..{a_idx[-1]} "
                      f"shape ({ACTION_HORIZON}, {actions.shape[1]})")
        if not args.no_vae:
            np.savez_compressed(
                f"{args.out}/ep{ep_i:04d}.npz",
                latents=np.stack(ep_latents), starts=np.asarray(starts),
                actions=actions, states=states, exec_start=exec_start)
        if len(index) % 2000 < 420:
            print(f"  shard {args.shard}: ep {ep_i}, {n_win} windows", flush=True)

    if args.num_shards > 1:
        meta_path = f"{args.out}/meta_shard{args.shard}.json"
    else:
        meta_path = f"{args.out}/meta.json"
    meta = {"stride": args.stride, "window_frames": WINDOW_FRAMES,
            "video_ratio": VIDEO_RATIO, "action_horizon": ACTION_HORIZON,
            "n_episodes": len(files), "n_windows": n_win,
            "n_exec_windows": n_exec, "latent_shape": [48, 3, 16, 32],
            "mosaic": "horizontal front|wrist 256x512"}
    with open(meta_path, "w") as f:
        json.dump({"meta": meta, "index": index}, f)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
