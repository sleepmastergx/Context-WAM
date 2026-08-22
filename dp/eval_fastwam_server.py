"""Fast-WAM action server for closed-loop MoveCube eval (control arm).

Runs in the Fast-WAM env (.venv-wam, torch 2.7.1); the sim runs in the DP env
(.venv-dp, torch 2.9.1). The two repos' pins cannot share a process, so the
model serves action chunks over a Unix socket to dp/eval_fastwam_client.py.

Memory arms (m5-ttt): the client streams EVERY frame; the server maintains
the raw frame/state buffer and replicates the training write schedule exactly:
write j consumes the window starting at s_j = w*(j-1) (w=8, 33 raw frames,
every 4th VAE-encoded) and COMPLETES at raw step s_j+32; before denoising a
window starting at t, all writes with j <= j_max(t) = (t-33)//w + 1 are
applied via PerLayerEpisodeMemory.advance (detached carry), the strictly-
before convention -- gated by checks/check_sliding_chain.py check 2. Reads
happen inside denoising through the patched action expert (memory._m).

Deploy operator = the training operator: training_loss jointly denoises the
window's future video (9 frames / 33 raw steps) and its 32-action chunk,
conditioned on the window's FIRST frame, the proprio at that frame, and the
T5 goal context. Deploy therefore calls model.infer_joint(input_image=current
front|wrist mosaic, num_video_frames=9, action_horizon=32, proprio=state,
context=the cache's text_context) and returns the jointly-denoised actions
(test_action_with_infer_action=False -- the action-only KV route is a
DIFFERENT operator). Actions are RAW joint values, exactly as cached.

This arm has memory.enabled=false, so no TTT state and no write cadence --
the streaming-equivalence machinery does not apply.

Protocol: length-prefixed pickles. Request {front,wrist:uint8[256,256,3],
state:float32[8], seed:int} -> response {action: float32[32,8]}.
Request {"cmd":"ping"} -> {"ok":True}; {"cmd":"quit"} shuts the server down.
"""
import argparse
import os
import pickle
import socket
import struct
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def recv_msg(conn):
    hdr = b""
    while len(hdr) < 8:
        c = conn.recv(8 - len(hdr))
        if not c:
            return None
        hdr += c
    (n,) = struct.unpack("<Q", hdr)
    buf = b""
    while len(buf) < n:
        c = conn.recv(min(1 << 20, n - len(buf)))
        if not c:
            return None
        buf += c
    return pickle.loads(buf)


def send_msg(conn, obj):
    b = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack("<Q", len(b)) + b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True, help="run_config.yaml of the training run")
    ap.add_argument("--text-context", required=True,
                    help="text_context.pt from the training cache (single-goal payload)")
    ap.add_argument("--socket", default="/tmp/fastwam_eval.sock")
    ap.add_argument("--num-inference-steps", type=int, default=20)
    ap.add_argument("--action-horizon", type=int, default=32)
    args = ap.parse_args()

    # we chdir into the fastwam root below (weight paths resolve relative to
    # it), so every user-supplied path must be absolute first
    args.ckpt = os.path.abspath(args.ckpt)
    args.config = os.path.abspath(args.config)
    args.text_context = os.path.abspath(args.text_context)

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    # Wan pretrained weights resolve relative to the fastwam repo root (same
    # convention as context_wam/convert_movecube.py); VAE/T5 already live there.
    fw_root = os.environ.get("FASTWAM_ROOT",
                             os.path.join(REPO, "third_party", "fastwam"))
    os.chdir(fw_root)

    from context_wam.build_model import build
    t0 = time.time()
    model, memory = build(cfg, device="cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected, f"unexpected keys in ckpt: {unexpected[:5]}"
    if missing:
        print(f"note: {len(missing)} keys missing from ckpt (buffers?):",
              missing[:5], flush=True)
    model.eval()
    print(f"model class: {type(model).__name__}", flush=True)
    if memory is not None:
        assert ck.get("memory"), "memory arm but ckpt has no memory state"
        memory.load_state_dict(ck["memory"])
        memory.eval()
        # sanity: our deploy write schedule == the chain's convention
        from context_wam.sliding_chain import SlidingChain
        jm = lambda t: 0 if t < 33 else (t - 33) // 8 + 1
        for t, want in ((43, 2), (47, 2), (49, 3)):
            assert jm(t) == want, (t, jm(t), want)
        print(f"memory arm: {memory.n_layers} layers, write cadence w=8, "
              f"gate |tanh a| = {float(memory.alpha.tanh().abs().mean()):.4f}",
              flush=True)
    print(f"model loaded from step {ck.get('step')} in {time.time()-t0:.0f}s",
          flush=True)

    stats = ck.get("action_stats")
    if stats:
        print(f"action_mode={stats['mode']}: normalizing proprio and "
              f"de-normalizing actions with the checkpoint's stats", flush=True)
        a_min = np.asarray(stats["action_min"], np.float32); a_max = np.asarray(stats["action_max"], np.float32)
        s_min = np.asarray(stats["state_min"], np.float32); s_max = np.asarray(stats["state_max"], np.float32)
        eps = float(stats.get("eps", 1e-6))

    def norm_state(raw):
        if not stats:
            return raw
        return (2.0 * (raw - s_min) / (s_max - s_min + eps) - 1.0).astype(np.float32)

    def denorm_action(act_n, raw_state):
        if not stats:
            return act_n
        a = (act_n + 1.0) / 2.0 * (a_max - a_min + eps) + a_min
        if stats["mode"] == "delta_norm":
            a[:, :7] = a[:, :7] + raw_state[None, :7]
        return a.astype(np.float32)

    tc = torch.load(args.text_context, map_location="cpu", weights_only=False)
    assert "context" in tc, "expected the single-goal text_context payload"
    context = tc["context"].unsqueeze(0).to("cuda", model.torch_dtype)
    context_mask = tc["mask"].unsqueeze(0).to("cuda")
    print(f"goal: {tc['goal']!r} | context {tuple(context.shape)}", flush=True)

    if os.path.exists(args.socket):
        os.remove(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    srv.listen(1)
    print(f"listening on {args.socket}", flush=True)

    W, SPAN, NSUB = 8, 33, 9

    @torch.inference_mode()
    def encode_window(frames, s):
        """VAE-encode the 9-frame subsample of the window starting at s.
        Mirrors convert_movecube.py exactly: mosaic [-1,1], [B,C,T,H,W] bf16."""
        idx = list(range(s, s + SPAN, 4))
        v = np.stack([frames[i] for i in idx]).astype(np.float32) / 127.5 - 1.0
        vt = torch.from_numpy(v).permute(3, 0, 1, 2)[None].to(
            "cuda", model.torch_dtype)                    # [1,C,T,H,W]
        return model.vae.encode(vt, device="cuda", tiled=False)

    ep = {"frames": [], "states": [], "writes": 0}

    n_req = 0
    while True:
        conn, _ = srv.accept()
        while True:
            req = recv_msg(conn)
            if req is None:
                break
            if req.get("cmd") == "ping":
                send_msg(conn, {"ok": True})
                continue
            if req.get("cmd") == "quit":
                send_msg(conn, {"ok": True})
                conn.close()
                srv.close()
                os.remove(args.socket)
                print(f"served {n_req} requests; bye", flush=True)
                return
            if memory is not None:
                if req.get("ep_start"):
                    memory.reset()
                    ep["frames"], ep["states"], ep["writes"] = [], [], 0
                for f, w_, st_ in zip(req.get("frames_front", []),
                                      req.get("frames_wrist", []),
                                      req.get("frames_states", [])):
                    ep["frames"].append(np.concatenate(
                        [np.asarray(f, np.uint8), np.asarray(w_, np.uint8)],
                        axis=1))
                    ep["states"].append(np.asarray(st_, np.float32))
                t = len(ep["frames"]) - 1     # current raw step = window start
                due = 0 if t < SPAN else (t - SPAN) // W + 1
                while ep["writes"] < due:
                    s_j = W * ep["writes"]
                    z = encode_window(ep["frames"], s_j)
                    pro = torch.from_numpy(norm_state(ep["states"][s_j])[None]).to(
                        "cuda", torch.float32)
                    with torch.inference_mode():
                        memory.advance(z.float(), pro)
                    ep["writes"] += 1

            front = np.asarray(req["front"], np.uint8)
            wrist = np.asarray(req["wrist"], np.uint8)
            mosaic = np.concatenate([front, wrist], axis=1)      # H, 2W, 3
            img = torch.from_numpy(mosaic).permute(2, 0, 1).float() / 127.5 - 1.0
            img = img.unsqueeze(0).to("cuda", model.torch_dtype)  # [1,3,256,512]
            raw_state = np.asarray(req["state"], np.float32)
            proprio = torch.from_numpy(norm_state(raw_state)).unsqueeze(0).to(
                "cuda", model.torch_dtype)
            t1 = time.time()
            with torch.inference_mode():
                if "Decoupled" not in type(model).__name__:
                    out = model.infer_joint(
                        prompt=None,
                        input_image=img,
                        num_video_frames=9,
                        action_horizon=args.action_horizon,
                        proprio=proprio,
                        context=context.clone(),
                        context_mask=context_mask.clone(),
                        num_inference_steps=args.num_inference_steps,
                        seed=int(req.get("seed", 0)),
                        test_action_with_infer_action=False,
                    )
                else:
                    # FastWAMDecoupled: their eval contract (trainer.evaluate ->
                    # infer) denoises actions from clean-first-frame video KV;
                    # infer_action is that stage without paying for the video
                    # rollout. Memory reads flow through the patched blocks.
                    out = model.infer_action(
                        prompt=None,
                        input_image=img,
                        action_horizon=args.action_horizon,
                        proprio=proprio,
                        context=context.clone(),
                        context_mask=context_mask.clone(),
                        num_inference_steps=args.num_inference_steps,
                        seed=int(req.get("seed", 0)),
                    )
            act = out["action"]
            if isinstance(act, torch.Tensor):
                act = act.detach().float().cpu().numpy()
            act = np.asarray(act, np.float32).reshape(args.action_horizon, -1)
            act = denorm_action(act, raw_state)
            n_req += 1
            if n_req % 50 == 1:
                print(f"req {n_req}: {time.time()-t1:.2f}s/infer", flush=True)
            send_msg(conn, {"action": act})
        conn.close()


if __name__ == "__main__":
    main()
