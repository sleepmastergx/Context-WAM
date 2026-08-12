"""Episode-batched dataset over cached frozen MoveCube features.

Differences from the RMBench version (exp/ttt/dataset.py):
  * Episodes have a conditioning-VIDEO prefix followed by execution frames.
    The TTT rollout consumes the WHOLE episode (mask covers video + exec);
    training windows are sampled from EXECUTION frames only — the video has
    no policy actions, and stage-1 trained exactly this way.
  * Window convention matches stage-1 (lerobot delta_timestamps), not
    RMBench's SequenceSampler: a window is indexed by its CURRENT frame t
    (t in the exec range); obs = clamp([t-1, t], 0, T-1); action chunk =
    clamp([t .. t+15], <= T-1) (repeat-last padding). Memory is read at t.
  * State dims inside the features and the actions are normalized with the
    STAGE-1 stats.json (their minmax DataTransform), so the head sees the
    same numeric ranges the stage-1 UNet saw.

Val split: last episode only (RMBench precedent, val_episode=-1).
"""
import os
import sys

import numpy as np
import torch

_DP_REPO = os.environ.get("DP_REPO")
assert _DP_REPO, "set DP_REPO to your clone of github.com/RoboMME/DP"
sys.path.insert(0, _DP_REPO)

N_STATE = 8   # dims 128..135 of the base feature vector are the raw state


class EpisodeFeatureDatasetMC:

    def __init__(self, npz_path, stats_path, split="train", val_episode=-1):
        from eval_envs.utils.normalize import load as load_norm_stats
        from eval_envs.utils.transform import DataTransform
        d = np.load(npz_path, allow_pickle=True)
        feat = d["feats"].astype(np.float32)          # (N, 136) raw state tail
        action = d["actions"].astype(np.float32)      # (N, 8) raw
        is_demo = d["is_demo"].astype(bool)
        ep_from = d["ep_from"].astype(np.int64)
        ep_to = d["ep_to"].astype(np.int64)

        stats = load_norm_stats(stats_path, filename="stats.json")
        tf = DataTransform(norm_stats=stats, norm_type="minmax", mask=None,
                           use_delta_action=False)
        normed = tf.transform_in({"state": feat[:, -N_STATE:].copy(),
                                  "action": action.copy()})
        feat[:, -N_STATE:] = normed["state"]
        action = normed["action"]

        # per-episode CLIP text embedding, appended AFTER normalization so the
        # state tail stays at dims 128..135. Layout: [vis 128 | state 8 | text
        # 512] — eval_stage2.py builds frames in the same order. Present only
        # when cache_features.py ran --with-text (VideoUnmask-family tasks);
        # MoveCube caches have no text_embs and d_feat stays 136.
        if "text_embs" in d.files:
            te = d["text_embs"].astype(np.float32)          # (E, 512)
            cols = np.empty((feat.shape[0], te.shape[1]), np.float32)
            for i, (s, e) in enumerate(zip(ep_from, ep_to)):
                cols[s:e] = te[i]
            feat = np.concatenate([feat, cols], axis=1)

        E = len(ep_from)
        idx = np.arange(E)
        val_idx = idx[val_episode]
        keep = (idx != val_idx) if split == "train" else (idx == val_idx)

        self.episodes = [(int(s), int(e)) for s, e, k in zip(ep_from, ep_to, keep) if k]
        # first execution frame, episode-relative
        self.exec_starts = [int(np.argmin(is_demo[s:e]))
                            for (s, e), k in zip(zip(ep_from, ep_to), keep) if k]
        for (s, e), xs in zip(self.episodes, self.exec_starts):
            assert not is_demo[s + xs] and is_demo[s:s + xs].all(), \
                "video prefix is not contiguous"
        self.feat = feat
        self.action = action
        self.d_feat = feat.shape[1]
        self.d_action = action.shape[1]

    def __len__(self):
        return len(self.episodes)


class GPUEpisodeBatcher:
    """All episodes padded and GPU-resident; batches assembled on device.

    Same as the RMBench version plus a per-episode exec_start tensor so
    window sampling can stay inside the execution segment.
    """

    def __init__(self, ds, batch_episodes, device, shuffle=True, seed=0,
                 windows_per_episode=16, steps_per_epoch=None):
        self.device = torch.device(device)
        self.batch_episodes = batch_episodes
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        E = len(ds.episodes)
        lens = np.array([e - s for s, e in ds.episodes], dtype=np.int64)
        T = int(lens.max())
        feat = np.zeros((E, T, ds.d_feat), dtype=np.float32)
        action = np.zeros((E, T, ds.d_action), dtype=np.float32)
        mask = np.zeros((E, T), dtype=np.float32)
        for i, (s, e) in enumerate(ds.episodes):
            n = e - s
            feat[i, :n] = ds.feat[s:e]
            action[i, :n] = ds.action[s:e]
            mask[i, :n] = 1.0

        self.feat = torch.from_numpy(feat).to(self.device)
        self.action = torch.from_numpy(action).to(self.device)
        self.mask = torch.from_numpy(mask).to(self.device)
        self.lengths = torch.from_numpy(lens)
        self.exec_starts = torch.tensor(ds.exec_starts, dtype=torch.long)
        self.n_episodes = E

        # an epoch covers the same number of WINDOW samples as stage-1 would:
        # one window per execution frame
        total_windows = int((lens - np.array(ds.exec_starts)).sum())
        per_step = batch_episodes * windows_per_episode
        self.steps_per_epoch = (steps_per_epoch if steps_per_epoch is not None
                                else max(1, total_windows // per_step))

        n_bytes = sum(t.numel() * t.element_size()
                      for t in (self.feat, self.action, self.mask))
        print(f"[GPUEpisodeBatcher] {E} episodes padded to T={T}, "
              f"{n_bytes / 2**20:.1f} MiB resident on {self.device} | "
              f"{total_windows} exec windows -> {self.steps_per_epoch} "
              f"steps/epoch x {per_step} windows", flush=True)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            if self.shuffle:
                pick = self.rng.choice(self.n_episodes, size=self.batch_episodes,
                                       replace=self.n_episodes < self.batch_episodes)
            else:
                pick = np.arange(min(self.batch_episodes, self.n_episodes))
            sel = torch.from_numpy(pick).to(self.device)
            cpu_pick = torch.from_numpy(pick)
            yield {
                "feat": self.feat[sel],
                "action": self.action[sel],
                "mask": self.mask[sel],
                "length": self.lengths[cpu_pick],
                "exec_start": self.exec_starts[cpu_pick],
            }


def sample_windows(lengths, exec_starts, k, generator=None):
    """Draw k CURRENT-frame indices per episode, uniform over the exec range."""
    B = lengths.shape[0]
    starts = torch.empty(B, k, dtype=torch.long)
    for i in range(B):
        lo = int(exec_starts[i])
        hi = int(lengths[i]) - 1
        starts[i] = torch.randint(lo, hi + 1, (k,), generator=generator)
    return starts


def gather_windows(feat, action, m, starts, horizon, n_obs_steps, lengths):
    """Stage-1 window convention: starts are CURRENT-frame indices t.

    obs idx    = clamp([t-(n_obs-1) .. t], 0, T-1)
    action idx = clamp([t .. t+horizon-1], <= T-1)   (repeat-last padding)
    memory     = m at t
    Returns global_cond (B*k, n_obs*d_f), mem (B*k, d_m) | None,
    action_chunk (B*k, horizon, d_a).
    """
    B, T, d_f = feat.shape
    k = starts.shape[1]
    dev = feat.device
    Tm1 = (lengths - 1).view(B, 1, 1).to(dev)
    starts = starts.to(dev)

    off_o = torch.arange(-(n_obs_steps - 1), 1, device=dev).view(1, 1, -1)
    idx_o = (starts.unsqueeze(-1) + off_o).clamp(min=0)
    idx_o = torch.minimum(idx_o, Tm1)                                # (B,k,n_obs)

    off_a = torch.arange(horizon, device=dev).view(1, 1, -1)
    idx_a = (starts.unsqueeze(-1) + off_a).clamp(min=0)
    idx_a = torch.minimum(idx_a, Tm1)                                # (B,k,horizon)

    obs = torch.gather(feat.unsqueeze(1).expand(B, k, T, d_f), 2,
                       idx_o.unsqueeze(-1).expand(B, k, n_obs_steps, d_f))
    global_cond = obs.reshape(B * k, n_obs_steps * d_f)

    d_a = action.shape[-1]
    act = torch.gather(action.unsqueeze(1).expand(B, k, T, d_a), 2,
                       idx_a.unsqueeze(-1).expand(B, k, horizon, d_a))
    action_chunk = act.reshape(B * k, horizon, d_a)

    mem = None
    if m is not None:
        d_m = m.shape[-1]
        idx_m = idx_o[:, :, -1]                                      # current frame
        mem = torch.gather(m.unsqueeze(1).expand(B, k, T, d_m), 2,
                           idx_m.unsqueeze(-1).unsqueeze(-1).expand(B, k, 1, d_m))
        mem = mem.reshape(B * k, d_m)

    return global_cond, mem, action_chunk
