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

2026-08-19 (490-episode / DDP recipe): the L per-layer cells are run as ONE
stacked chain (memory.ttt_with_state_stacked — exact, see check 5), the cache
may live on the CPU (only the E x J chain inputs are moved to the memory's
device each step), and `readouts()` exposes the in-graph per-layer readouts so
train.py can run the chain ONCE per optimizer step and still accumulate
gradients over micro-batches (see train.py "chain-once protocol").
"""
from typing import Dict

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from memory import stack_cells, stacked_init_state, ttt_with_state_stacked


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
    def _seg(self, P, lat_seg, pro_seg, W1, W2, M1, M2):
        """Advance ALL cells over a segment of writes. lat_seg [E,j,C,t,h,w].
        Returns (m [L,E,j,d_out], mean surprise, *state)."""
        state = (W1, W2, M1, M2)
        ms, sur = [], []
        for j in range(lat_seg.shape[1]):
            x = self.memory.pool_latents(lat_seg[:, j], pro_seg[:, j])
            x = x.to(P["W1_0"].dtype)
            mask = torch.ones(x.shape[0], x.shape[1], device=x.device,
                              dtype=x.dtype)
            m, surprise, state = ttt_with_state_stacked(P, x, mask, state)
            ms.append(m[:, :, -1])
            sur.append(surprise.mean())
        return (torch.stack(ms, dim=2), torch.stack(sur).mean()) + state

    def readouts(self, idx: torch.Tensor):
        """Run the batch's chains in-graph and return the per-layer readouts.

        idx: [B] cache row indices of the sampled exec windows (any device).
        Returns (ms, stats): ms is a list over memory layers of [B, d_out]
        fp32 tensors ON THE GRAPH (loss -> ms -> chain -> write params), stats
        is a dict of floats for logging. Distinct episodes are batched into ONE
        stacked chain over all layers, rolled to the furthest sampled window;
        each window then reads its own prefix readout. Episodes shorter than
        the longest chain re-consume their final window (clamped rows) — those
        writes are never read and carry no gradient into any loss.
        """
        mem = self.memory
        dev = next(mem.parameters()).device
        idx_c = idx.detach().to(self.cache.ep.device)
        starts = self.cache.start[idx_c].detach().cpu().numpy().astype(np.int64)
        eps = self.cache.ep[idx_c].detach().cpu().numpy().astype(np.int64)

        uniq, inv = np.unique(eps, return_inverse=True)
        E = len(uniq)
        j_need = (starts - self.SPAN) // self.w + 1
        if (j_need < 1).any():
            raise ValueError("a sampled window precedes the first completed "
                             "write — sample exec windows only")
        J = int(j_need.max())

        w_starts = np.arange(J, dtype=np.int64) * self.w
        rows = np.empty((E, J), dtype=np.int64)
        for k, e in enumerate(uniq):
            rows[k] = self.row0[int(e)] + np.minimum(w_starts,
                                                     self.n_win[int(e)] - 1)
        rows_t = torch.from_numpy(rows).to(self.cache.latents.device)

        # fp32, OUTSIDE any autocast: the inner TTT update is a forward-pass
        # gradient and quantizes to noise in bf16. The cache may live on the
        # CPU (490-episode recipe): only these E x J inputs cross to the GPU.
        lat = self.cache.latents[rows_t].to(dev, non_blocking=True).float()   # [E,J,C,t,h,w]
        pro = self.cache.states[rows_t].to(dev, non_blocking=True).float()    # [E,J,P]

        inv_t = torch.from_numpy(inv).to(dev)
        jsel = torch.from_numpy(j_need - 1).to(dev)       # 0-indexed readout

        cells = [mem._cell(l) for l in range(mem.n_layers)]
        P = stack_cells(cells)                            # [L,...] in-graph
        state = stacked_init_state(P, E)                  # learned init trains
        seg = self.ckpt_every if self.ckpt_every > 0 else J
        m_hist, sur_all = [], []
        j = 0
        while j < J:
            e = min(j + seg, J)
            if self.ckpt_every > 0 and torch.is_grad_enabled():
                out = checkpoint(
                    lambda ls, ps, a, b, c, d, _P=P:
                        self._seg(_P, ls, ps, a, b, c, d),
                    lat[:, j:e], pro[:, j:e], *state, use_reentrant=False)
            else:
                out = self._seg(P, lat[:, j:e], pro[:, j:e], *state)
            m_seg, sur_seg, state = out[0], out[1], out[2:]
            m_hist.append(m_seg)
            sur_all.append(float(sur_seg.detach()))
            j = e
        M = torch.cat(m_hist, dim=2)                      # [L, E, J, d_out]
        ms = [M[l, inv_t, jsel] for l in range(mem.n_layers)]   # [B, d_out] each
        stats = {"chain_J": float(J),
                 "chain_E": float(E),
                 "surprise": float(np.mean(sur_all)) if sur_all else 0.0,
                 "j_need_mean": float(j_need.mean())}
        return ms, stats

    def load_states(self, idx: torch.Tensor) -> Dict[str, float]:
        """readouts() + install them into the memory so read_layer serves the
        following forward(s). Returns the logging stats."""
        ms, stats = self.readouts(idx)
        self.set_readouts(ms)
        return stats

    def set_readouts(self, ms):
        """Install per-layer readouts (list of [B, d_out]) for read_layer."""
        mem = self.memory
        for l in range(mem.n_layers):
            mem._m[l] = ms[l]
        mem._states = [None] * mem.n_layers               # reads use _m only
