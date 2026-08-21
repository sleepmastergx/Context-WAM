"""Contiguous latent cache with optional VRAM or host-RAM residency.

The original 100-episode MoveCube cache is ~5.7 GiB bf16 (41.7k windows x
[48,3,16,32]) and can be gathered directly in VRAM. The 500-episode extension
is kept in host RAM for 80 GiB cards and only each micro batch moves to the
training GPU. Both modes avoid PNG/parquet or compressed-NPZ reads in the step
loop.

VRAM accounting per rank (bf16):
    latents      5.7 GiB   resident, read-only
    actions/state <0.1 GiB
    video expert ~10  GiB   Wan2.2-5B frozen: params only, no grads, no optim
    action expert ~1.7 GiB   115M: params + grads + AdamW fp32 states
    activations   the free variable -> batch size
"""
import glob
import json
import os

import numpy as np
import torch


class GPUWindowCache:
    """Also carries the two things their `training_loss` demands beyond latents:

      context / context_mask   T5 embedding of the goal. `load_text_encoder:
                               false` means the model never builds T5, so this
                               is precomputed once (encode_text_context.py).
                               MoveCube has ONE goal, so it is shared by every
                               window and just expanded to the batch.
      source_video_shape       [C, T, H, W] of the SOURCE video behind each
                               cached latent: [3, 9, 256, 512]. Required
                               whenever video_latents is supplied instead of
                               raw video; their checks want T % 4 == 1 (9 ok)
                               and action_horizon % (T-1) == 0 (32 % 8 ok).
    """

    def __init__(
        self,
        root: str,
        device,
        split_episodes=None,
        dtype=torch.bfloat16,
        storage_device=None,
    ):
        metas = sorted(glob.glob(f"{root}/meta*.json"))
        if not metas:
            raise FileNotFoundError(f"no meta*.json under {root} — run the conversion")
        index, meta = [], None
        for m in metas:
            d = json.load(open(m))
            meta = d["meta"]
            index.extend(d["index"])
        self.meta = meta
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device or device)

        eps = sorted({r["ep"] for r in index})
        if split_episodes is not None:
            eps = [e for e in eps if e in set(split_episodes)]
        keep = set(eps)

        lat, act, sta, ep_id, start, is_exec = [], [], [], [], [], []
        for e in eps:
            f = f"{root}/ep{e:04d}.npz"
            if not os.path.exists(f):
                raise FileNotFoundError(f"missing {f}")
            z = np.load(f)
            lat.append(torch.from_numpy(z["latents"]).to(dtype))
            starts = z["starts"]
            A, S = z["actions"], z["states"]
            T, H = len(A), meta["action_horizon"]
            idx = np.clip(starts[:, None] + np.arange(H)[None, :], 0, T - 1)
            act.append(torch.from_numpy(A[idx]).to(dtype))
            sta.append(torch.from_numpy(S[np.clip(starts, 0, T - 1)]).to(dtype))
            ep_id.append(torch.full((len(starts),), e, dtype=torch.int32))
            start.append(torch.from_numpy(starts.astype(np.int32)))
            is_exec.append(torch.from_numpy(
                (starts >= int(z["exec_start"])).astype(np.bool_)))

        self.latents = torch.cat(lat).to(self.storage_device)
        self.actions = torch.cat(act).to(self.storage_device)
        self.states = torch.cat(sta).to(self.storage_device)
        self.ep = torch.cat(ep_id).to(self.storage_device)
        self.start = torch.cat(start).to(self.storage_device)
        self.is_exec = torch.cat(is_exec).to(self.storage_device)
        # Text context. MoveCube: ONE goal shared by every window. VideoUnmask:
        # the goal names the query color, so the context is per EPISODE and is
        # gathered per window — sharing one context there would make the model
        # blind to the query (see encode_text_context.py).
        tpath = os.path.join(root, "text_context.pt")
        if not os.path.exists(tpath):
            raise FileNotFoundError(
                f"missing {tpath} — run encode_text_context.py "
                "(their training_loss requires sample['context'])")
        tp = torch.load(tpath, map_location="cpu")
        self.per_episode_text = "contexts" in tp
        if self.per_episode_text:
            self.contexts = tp["contexts"].to(
                self.storage_device, dtype)  # [K, L, 4096]
            self.context_masks = tp["masks"].to(
                self.storage_device)       # [K, L] bool
            ep_task = {int(k): int(v) for k, v in tp["ep_task"].items()}
            missing = keep - set(ep_task)
            if missing:
                raise KeyError(f"text_context.pt has no goal for episodes "
                               f"{sorted(missing)} — re-run encode_text_context.py "
                               f"--lerobot-root against THIS dataset")
            # per-window goal id, so batch() is one index_select like the rest
            self.win_task = torch.tensor(
                [ep_task[int(e)] for e in self.ep.tolist()],
                dtype=torch.long, device=self.storage_device)
            self.context = self.context_mask = None
        else:
            self.context = tp["context"].to(
                self.storage_device, dtype)   # [L, 4096]
            self.context_mask = tp["mask"].to(
                self.storage_device)        # [L] bool
        C, T, H, W = 3, 9, 256, 512
        assert T % 4 == 1 and H % 16 == 0 and W % 16 == 0
        self.source_video_shape = torch.tensor(
            [[C, T, H, W]], device=self.storage_device)

        n_bytes = sum(t.numel() * t.element_size() for t in
                      (self.latents, self.actions, self.states))
        txt = (f"{len(self.contexts)} per-episode goals"
               if self.per_episode_text else "1 shared goal")
        print(f"[GPUWindowCache] {len(self.latents)} windows from {len(eps)} "
              f"episodes, {n_bytes/2**30:.2f} GiB resident on "
              f"{self.storage_device} (batches -> {self.device}) "
              f"({int(self.is_exec.sum())} exec, {txt})", flush=True)

    def __len__(self):
        return len(self.latents)

    def exec_indices(self):
        return torch.nonzero(self.is_exec, as_tuple=False).squeeze(-1)

    def batch(self, idx):
        """Gather from the cache and move only the batch to the training GPU."""
        idx = idx.to(self.storage_device)
        B = idx.shape[0]
        if self.per_episode_text:
            t = self.win_task[idx]
            ctx, ctx_mask = self.contexts[t], self.context_masks[t]
        else:
            ctx = self.context.unsqueeze(0).expand(B, -1, -1)
            ctx_mask = self.context_mask.unsqueeze(0).expand(B, -1)

        def on_device(tensor):
            if tensor.device == self.device:
                return tensor
            return tensor.to(self.device, non_blocking=tensor.is_pinned())

        return {"video_latents": on_device(self.latents[idx]),
                "action": on_device(self.actions[idx]),
                # [B,1,P], not [B,P]: their build_inputs demands a 3D
                # [B,T,d] proprio and then takes `proprio[:, 0, :]`. One state
                # per window IS T=1, so this is a reshape, not a change of
                # meaning. `self.states` stays 2D — SlidingChain indexes it
                # directly as [E,J,P].
                "proprio": on_device(self.states[idx].unsqueeze(1)),
                "is_exec": on_device(self.is_exec[idx]),
                "context": on_device(ctx),
                "context_mask": on_device(ctx_mask),
                "source_video_shape": on_device(
                    self.source_video_shape.expand(B, -1))}
