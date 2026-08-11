"""Sliding video-stream memory for the context-wam arm (w-cadence writes).

Design (settled 2026-08-11, see the design-review artifact §07 revision):
the memory is written from the VIDEO STREAM ONLY — never actions — over the
whole episode, one write every w raw steps, chained IN-GRAPH from the learned
init. A training window starting at raw step t reads the state after the last
COMPLETED write strictly before t. Exec windows are sampled uniformly at
random (the control's sampler), so the arms differ only by the memory.

w=8 REALIZATION — why writes consume trailing windows, not disjoint chunks:
the Wan VAE encodes only clips of T%4==1 subsampled frames (windows are 9
frames spanning 33 raw steps), so a disjoint 8-raw-step chunk has no latent
representation. The finest realizable stream is the cached per-window latents
at 8-step stride: write j consumes the window starting at s_j = w*(j-1),
covering raw steps s_j..s_j+32, and COMPLETES at raw step s_j+32. Writes
overlap by 25 raw steps; the TTT surprise gate is what de-duplicates re-seen
content (low surprise -> small write). The strictly-before rule is exact:

    readable writes at window start t:  j_max(t) = (t - SPAN)//w + 1

so t=43 reads writes {s=0, s=8} (content through raw 40), t=47 the same, and
t=49 additionally reads s=16 (its content ends at raw 48 <= t-1).
Exec windows always have j_max >= 20, so reads never see an empty state.

TRAINING (this file): one batched chain per optimizer step over the batch's
distinct episodes, rolled only to the furthest sampled window per episode,
kept fully IN-GRAPH — precomputing or detaching the chain severs the outer
loop and the memory can never learn WHAT to store (see the in-graph-chain
concept note). Chain inputs are cached VAE latents (frozen VAE): constant
data, so inputs are cacheable forever while states must be recomputed each
step because the write parameters just changed.

DEPLOY (eval, to be built): the same operator streamed — every w env steps,
VAE-encode the trailing 33-step window and advance once, detached. Gate
train == deploy with checks/check_sliding_chain.py test 2 (the
check_chunk_equiv analog) before trusting any number.

BPTT depth: max T=525 -> J ~ 62 writes, ~2x the deepest validated
full-episode BPTT (stage-2 ~33). `checkpoint_every > 0` recomputes chain
segments in backward — EXACT gradients, unlike detaching, which is the one
lever never pulled here.
"""
from typing import Dict

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from memory import ttt_with_state


class SlidingChain:
    """Runs the per-episode write chains and loads per-window prefix states
    into a PerLayerEpisodeMemory so `read_layer` can serve a batched forward.

    Duck-typed against GPUWindowCache: needs .latents [N,C,t,h,w], .states
    [N,P], .ep [N], .start [N], with stride-1 contiguous rows per episode
    (asserted at init — anything else would silently feed the wrong window).
    """

    SPAN = 33          # a window covers raw steps s..s+SPAN-1 (= horizon + 1)

    def __init__(self, cache, memory, w: int = 8, checkpoint_every: int = 0):
        self.cache = cache
        self.memory = memory
        self.w = int(w)
        self.ckpt_every = int(checkpoint_every)
        if self.w < 1:
            raise ValueError("w must be >= 1")

        ep = cache.ep.detach().cpu().numpy()
        start = cache.start.detach().cpu().numpy()
        self.row0: Dict[int, int] = {}
        self.n_win: Dict[int, int] = {}
        for e in np.unique(ep):
            rows = np.nonzero(ep == e)[0]
            if not (np.diff(rows) == 1).all() or \
               not (start[rows] == np.arange(len(rows))).all():
                raise ValueError(f"episode {e}: cache rows not stride-1 contiguous")
            self.row0[int(e)] = int(rows[0])
            self.n_win[int(e)] = int(len(rows))

    # ---- bookkeeping ------------------------------------------------------
    def j_max(self, t: int) -> int:
        """Completed writes readable by a window starting at raw step t
        (strictly-before rule, settled convention 2026-08-11)."""
        return max(0, (int(t) - self.SPAN) // self.w + 1)

    # ---- the in-graph chain ----------------------------------------------
    def _init_state(self, cell, E: int):
        # expand() of a Parameter keeps the graph: the learned init trains
        W1 = cell.W1_0.unsqueeze(0).expand(E, -1, -1)
        W2 = cell.W2_0.unsqueeze(0).expand(E, -1, -1)
        return W1, W2, torch.zeros_like(W1), torch.zeros_like(W2)

    def _seg(self, cell, lat_seg, pro_seg, W1, W2, M1, M2):
        """Advance `cell` over a segment of writes. lat_seg [E,j,C,t,h,w]."""
        state = (W1, W2, M1, M2)
        ms, sur = [], []
        for j in range(lat_seg.shape[1]):
            x = self.memory.pool_latents(lat_seg[:, j], pro_seg[:, j])
            x = x.to(cell.ln_in.weight.dtype)
            mask = torch.ones(x.shape[0], x.shape[1], device=x.device,
                              dtype=x.dtype)
            m, diag, state = ttt_with_state(cell, x, mask, state)
            ms.append(m[:, -1])
            sur.append(diag["surprise"].mean())
        return (torch.stack(ms, dim=1), torch.stack(sur).mean()) + state

    def load_states(self, idx: torch.Tensor) -> Dict[str, float]:
        """Run the batch's chains in-graph and set memory._m for read_layer.

        idx: [B] cache row indices of the sampled exec windows. Distinct
        episodes are batched into ONE chain per layer, rolled to the furthest
        sampled window; each window then reads its own prefix readout.
        Episodes shorter than the longest chain re-consume their final window
        (clamped rows) — those writes are never read and carry no gradient
        into any loss. Returns logging stats.
        """
        starts = self.cache.start[idx].detach().cpu().numpy().astype(np.int64)
        eps = self.cache.ep[idx].detach().cpu().numpy().astype(np.int64)

        uniq, inv = np.unique(eps, return_inverse=True)
        E = len(uniq)
        j_need = (starts - self.SPAN) // self.w + 1
        if (j_need < 1).any():
            raise ValueError("a sampled window precedes the first completed "
                             "write — sample exec windows only")
        J = int(j_need.max())

        dev = self.cache.latents.device
        w_starts = np.arange(J, dtype=np.int64) * self.w
        rows = np.empty((E, J), dtype=np.int64)
        for k, e in enumerate(uniq):
            rows[k] = self.row0[int(e)] + np.minimum(w_starts,
                                                     self.n_win[int(e)] - 1)
        rows_t = torch.from_numpy(rows).to(dev)

        # fp32, OUTSIDE any autocast: the inner TTT update is a forward-pass
        # gradient and quantizes to noise in bf16
        lat = self.cache.latents[rows_t].float()          # [E, J, C, t, h, w]
        pro = self.cache.states[rows_t].float()           # [E, J, P]

        inv_t = torch.from_numpy(inv).to(dev)
        jsel = torch.from_numpy(j_need - 1).to(dev)       # 0-indexed readout

        mem = self.memory
        seg = self.ckpt_every if self.ckpt_every > 0 else J
        sur_all = []
        for l in range(mem.n_layers):
            cell = mem._cell(l)
            state = self._init_state(cell, E)
            m_hist = []
            j = 0
            while j < J:
                e = min(j + seg, J)
                if self.ckpt_every > 0 and torch.is_grad_enabled():
                    out = checkpoint(
                        lambda ls, ps, a, b, c, d, _cell=cell:
                            self._seg(_cell, ls, ps, a, b, c, d),
                        lat[:, j:e], pro[:, j:e], *state, use_reentrant=False)
                else:
                    out = self._seg(cell, lat[:, j:e], pro[:, j:e], *state)
                m_seg, sur_seg, state = out[0], out[1], out[2:]
                m_hist.append(m_seg)
                sur_all.append(float(sur_seg.detach()))
                j = e
            M = torch.cat(m_hist, dim=1)                  # [E, J, d_out]
            mem._m[l] = M[inv_t, jsel]                    # [B, d_out], in-graph
        mem._states = [None] * mem.n_layers               # reads use _m only

        return {"chain_J": float(J),
                "chain_E": float(E),
                "surprise": float(np.mean(sur_all)) if sur_all else 0.0,
                "j_need_mean": float(j_need.mean())}
