"""Decode-once frame cache for RoboMMEDataset.

RoboMMEDataset.__getitem__ goes through LeRobotDataset, which PNG-decodes
every queried frame on every access — 4 decodes per sample, 25.6M samples
over a 200k-step run. This subclass decodes each frame exactly once at
startup into a uint8 tensor (optionally in shared memory so DataLoader
workers see one copy), then serves windows from RAM.

Faithfulness: __getitem__ reproduces LeRobotDataset's windowing exactly —
query index = clamp(idx + round(delta*fps), ep_start, ep_end-1) — and the
uint8/255 conversion equals torchvision ToTensor, so outputs are BIT-EXACT
vs the parent (asserted by check_cache_equiv.py; the *_is_pad keys the
parent discards are not reproduced). Sampler/index space is unchanged, so
training with the cache is numerically identical to training without it.

Memory: MoveCube = 39,388 frames x (6,256,256) uint8 = 15.5 GB per rank.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch

from eval_envs.dataset.robomme_dataset import RoboMMEDataset

# Threads, NOT a fork pool: by cache-build time pyarrow/HF-datasets have live
# thread pools, and forked children inherit their locked mutexes ->
# pd.read_parquet deadlocks in the workers (observed, 0% CPU forever).
# cv2.imdecode releases the GIL, so threads decode in true parallel, and PNG
# is lossless so pixels are bit-identical to the PIL path lerobot uses
# (modulo BGR->RGB, handled below).


def _decode_episode(job):
    img_c, st_c, ac_c, path, row_from, row_to = job
    df = pd.read_parquet(path, columns=["image", "wrist_image", "state", "actions"])
    assert len(df) == row_to - row_from, f"{path}: {len(df)} rows vs index {row_to - row_from}"
    for i in range(len(df)):
        pair = []
        for col in ("image", "wrist_image"):
            cell = df[col].iloc[i]
            buf = cell["bytes"] if isinstance(cell, dict) else cell
            a = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            pair.append(np.ascontiguousarray(a[:, :, ::-1].transpose(2, 0, 1)))
        img_c[row_from + i] = torch.from_numpy(np.concatenate(pair, axis=0))
    st_c[row_from:row_to] = torch.from_numpy(
        np.stack(df["state"].to_numpy()).astype(np.float32))
    ac_c[row_from:row_to] = torch.from_numpy(
        np.stack(df["actions"].to_numpy()).astype(np.float32))


class CachedRoboMMEDataset(RoboMMEDataset):

    def __init__(self, *args, cache_workers=16, cache_share=True,
                 max_episodes=None, **kwargs):
        super().__init__(*args, **kwargs)
        fps = self.lerobot_dataset.fps
        self._obs_deltas = [round(-(self.obs_horizon - 1 - i) / fps * fps)
                            for i in range(self.obs_horizon)]
        self._act_deltas = [round(i / fps * fps)
                            for i in range(self.action_pred_horizon)]

        ep_idx = self.lerobot_dataset.episode_data_index
        metas = list(self.lerobot_dataset.meta.episodes.values())
        if max_episodes is not None:
            metas = metas[:max_episodes]
            cutoff = ep_idx["to"][len(metas) - 1].item()
            self._valid_indices = [r for r in self._valid_indices if r < cutoff]

        n_rows = ep_idx["to"][len(metas) - 1].item()
        h, w = 256, 256
        t0 = time.time()
        print(f"[CachedRoboMMEDataset] decoding {n_rows} frames "
              f"({n_rows * 6 * h * w / 2**30:.1f} GiB uint8, "
              f"{cache_workers} workers, share={cache_share}) ...", flush=True)
        self._img = torch.empty((n_rows, 6, h, w), dtype=torch.uint8)
        self._state = torch.empty((n_rows, 8), dtype=torch.float32)
        self._act = torch.empty((n_rows, 8), dtype=torch.float32)
        if cache_share:
            for t in (self._img, self._state, self._act):
                t.share_memory_()

        # per-row episode bounds + episode id for O(1) lookup in __getitem__
        self._row_from = np.empty(n_rows, np.int64)
        self._row_to = np.empty(n_rows, np.int64)
        self._row_ep = np.empty(n_rows, np.int64)
        jobs = []
        for i, m in enumerate(metas):
            f, t = ep_idx["from"][i].item(), ep_idx["to"][i].item()
            self._row_from[f:t], self._row_to[f:t] = f, t
            self._row_ep[f:t] = m["episode_index"]
            chunk = m["episode_index"] // self._chunks_size
            path = self.dataset_root / self._data_path_template.format(
                episode_chunk=chunk, episode_index=m["episode_index"])
            jobs.append((self._img, self._state, self._act, str(path), f, t))

        with ThreadPoolExecutor(max_workers=max(1, cache_workers)) as pool:
            list(pool.map(_decode_episode, jobs))
        print(f"[CachedRoboMMEDataset] cache ready in {time.time()-t0:.0f}s",
              flush=True)

    def __getitem__(self, idx):
        row = self._valid_indices[idx]
        f, t = self._row_from[row], self._row_to[row]
        obs_ix = [max(f, min(t - 1, row + d)) for d in self._obs_deltas]
        act_ix = [max(f, min(t - 1, row + d)) for d in self._act_deltas]

        image = self._img[obs_ix].to(torch.float32).div_(255.0)  # (T, 6, H, W)
        out = {
            "image":  image.numpy(),
            "state":  self._state[obs_ix].numpy().copy(),
            "action": self._act[act_ix].numpy().copy(),
        }
        if self.embed_text:
            task_idx = self.episode_to_task.get(int(self._row_ep[row]), 0)
            embed = self._task_embeds.get(task_idx, np.zeros(512, dtype=np.float32))
            out["text_emb"] = np.stack([embed] * self.obs_horizon, axis=0)
        return self.data_transform.transform_in(out)
