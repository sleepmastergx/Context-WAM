"""Episode memory for Fast-WAM: a TTT fast-weight state read as context tokens.

Design option A (see the design memo). The memory enters through the SAME door
`proprio_encoder` already uses — projected into the 4096-d text-context space and
appended as extra tokens — so both MoT experts attend to it via cross-attention
and NOTHING in self-attention changes. That is what preserves Fast-WAM's
inference trick: the video expert still runs once and the action expert still
denoises against cached video KV.

What the memory reads (write path):
    video latents [B, C, t, h, w] for the window -> mean-pool over (h, w) per
    latent frame -> [B, t, C]; optionally concat proprio broadcast per frame.
    So a 3-latent-frame window contributes exactly 3 inner TTT steps.

What the policy reads (read path):
    m_t (d_out) -> Linear(d_out -> text_dim) -> n_tokens context tokens.

State carry (RoboTTT's rule, and ours): fast weights persist ACROSS windows for
the whole episode; gradients are DETACHED at every window boundary. `state` is
an opaque tuple the trainer threads through consecutive windows of one episode
and resets at episode start.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ttt_cell import TTTMemory


class FastWAMEpisodeMemory(nn.Module):

    def __init__(
        self,
        latent_channels: int = 48,
        proprio_dim: Optional[int] = None,
        text_dim: int = 4096,
        d_k: int = 128,
        d_v: int = 128,
        d_hidden: int = 512,
        d_out: int = 256,
        chunk: int = 1,          # one inner step per latent frame
        n_tokens: int = 4,
        gate_init: float = 0.001,   # RoboTTT-style near-zero gate
    ):
        super().__init__()
        d_in = latent_channels + (proprio_dim or 0)
        self.proprio_dim = proprio_dim
        self.n_tokens = int(n_tokens)
        self.text_dim = int(text_dim)
        self.ttt = TTTMemory(d_in=d_in, d_k=d_k, d_v=d_v, d_hidden=d_hidden,
                             d_out=d_out, chunk=chunk)
        self.to_tokens = nn.Linear(d_out, self.text_dim * self.n_tokens)
        # Near-zero gate so that at init the memory contributes ~nothing and the
        # model behaves exactly like stock Fast-WAM. RoboTTT uses the same trick
        # (tanh(alpha), alpha init 1e-3) to preserve pretrained capability; our
        # RMBench `concat` inject LOST this parity and the arms were no longer
        # the same architecture at step 0.
        self.alpha = nn.Parameter(torch.full((1,), float(gate_init)))

    # ---- write ------------------------------------------------------------
    def pool_latents(self, video_latents: torch.Tensor,
                     proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[B, C, t, h, w] -> [B, t, d_in] stream for the TTT."""
        if video_latents.ndim != 5:
            raise ValueError(
                f"video_latents must be 5D [B,C,t,h,w], got {tuple(video_latents.shape)}")
        x = video_latents.mean(dim=(3, 4)).transpose(1, 2)          # [B, t, C]
        if self.proprio_dim is not None:
            if proprio is None:
                raise ValueError("memory was built with proprio_dim but got proprio=None")
            x = torch.cat([x, proprio.unsqueeze(1).expand(-1, x.shape[1], -1)], dim=-1)
        return x

    def forward(
        self,
        video_latents: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
    ):
        """Advance the memory over one window.

        Returns (m_last [B, d_out], new_state). `state` is carried across windows
        of the same episode and must be detached by the caller at the boundary.
        """
        x = self.pool_latents(video_latents, proprio).to(self.ttt.ln_in.weight.dtype)
        mask = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        m, _diag, new_state = ttt_with_state(self.ttt, x, mask, state)
        return m[:, -1], new_state

    # ---- read -------------------------------------------------------------
    def tokens(self, m_last: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """m_t -> [B, n_tokens, text_dim] context tokens, gated near zero at init."""
        B = m_last.shape[0]
        tok = self.to_tokens(m_last).view(B, self.n_tokens, self.text_dim)
        return (torch.tanh(self.alpha) * tok).to(dtype)

    def append_to_context(self, context, context_mask, m_last):
        tok = self.tokens(m_last, context.dtype)
        mask = torch.ones((context_mask.shape[0], self.n_tokens),
                          dtype=context_mask.dtype, device=context_mask.device)
        return (torch.cat([context, tok], dim=1),
                torch.cat([context_mask, mask], dim=1))


def ttt_with_state(cell: TTTMemory, x, mask, state=None):
    """Run TTTMemory over x, starting from `state` instead of the learned init.

    TTTMemory.forward always starts from (W1_0, W2_0, 0, 0). For episode-level
    carry we need to resume from the previous window's weights, so this mirrors
    its loop and additionally returns the final (W1, W2, M1, M2).
    """
    import torch.nn.functional as F

    B, T, _ = x.shape
    xn = cell.ln_in(x)
    K, V, Q = cell.to_k(xn), cell.to_v(xn), cell.to_q(xn)
    eta = F.softplus(cell.to_eta(xn)).squeeze(-1)
    alpha = torch.sigmoid(cell.to_alpha(xn)).squeeze(-1)
    beta = torch.sigmoid(cell.beta_logit)

    if state is None:
        W1 = cell.W1_0.unsqueeze(0).expand(B, -1, -1)
        W2 = cell.W2_0.unsqueeze(0).expand(B, -1, -1)
        M1, M2 = torch.zeros_like(W1), torch.zeros_like(W2)
    else:
        W1, W2, M1, M2 = state

    outs, surprises = [], []
    for s in range(0, T, cell.chunk):
        e = min(s + cell.chunk, T)
        kc, vc, qc = K[:, s:e], V[:, s:e], Q[:, s:e]
        mc = mask[:, s:e]

        preq = torch.einsum("bhk,bck->bch", W1, qc)
        hq = F.gelu(preq)
        mq = torch.einsum("bvh,bch->bcv", W2, hq)
        outs.append(cell.readout(mq))

        dPre, dY, h, kk, err = cell._inner_grad(W1, W2, kc, vc)
        w = (eta[:, s:e] * mc).unsqueeze(-1)
        denom = mc.sum(dim=1).clamp(min=1.0).view(B, 1, 1)
        g1 = torch.einsum("bch,bck->bhk", dPre * w, kk) / denom
        g2 = torch.einsum("bcv,bch->bvh", dY * w, h) / denom
        gn = (g1.flatten(1).pow(2).sum(1) + g2.flatten(1).pow(2).sum(1) + 1e-12).sqrt()
        scale = (cell.max_write / gn).clamp(max=1.0).view(B, 1, 1)
        g1, g2 = g1 * scale, g2 * scale
        surprises.append(((err.pow(2).sum(-1)) * mc).detach())

        a = (alpha[:, s:e] * mc).mean(dim=1).view(B, 1, 1)
        M1 = beta * M1 - g1
        M2 = beta * M2 - g2
        W1 = (1.0 - a) * W1 + M1
        W2 = (1.0 - a) * W2 + M2

    m = cell.ln(torch.cat(outs, dim=1))
    diag = {"surprise": torch.cat(surprises, dim=1)}
    return m, diag, (W1, W2, M1, M2)


def detach_state(state):
    """Truncated BPTT boundary: carry the values, cut the graph."""
    return None if state is None else tuple(t.detach() for t in state)


# ---------------------------------------------------------------------------
# Batched-over-cells variant (2026-08-19). The sliding chain runs one chain per
# memory layer; looping over L cells serialises L x J x T tiny kernels and was
# the chain's wall-clock bottleneck. This runs all L cells in ONE loop with a
# leading cell axis on every slow weight. torch.stack keeps the graph, so each
# cell's parameters receive exactly the gradient the looped version gives
# (checks/check_sliding_chain.py test 5 asserts this to 1e-5).
# ---------------------------------------------------------------------------
def stack_cells(cells):
    """Stack the slow weights of L identical-shape TTTMemory cells -> dict of
    tensors with a leading L axis (in-graph views of the parameters)."""
    c0 = cells[0]
    for c in cells[1:]:
        if c.chunk != c0.chunk or c.max_write != c0.max_write:
            raise ValueError("stacked cells must share chunk and max_write")
    st = lambda f: torch.stack([f(c) for c in cells])     # noqa: E731
    return {
        "ln_in_w": st(lambda c: c.ln_in.weight), "ln_in_b": st(lambda c: c.ln_in.bias),
        "ln_in_eps": c0.ln_in.eps,
        "Wk": st(lambda c: c.to_k.weight), "bk": st(lambda c: c.to_k.bias),
        "Wv": st(lambda c: c.to_v.weight), "bv": st(lambda c: c.to_v.bias),
        "Wq": st(lambda c: c.to_q.weight), "bq": st(lambda c: c.to_q.bias),
        "We": st(lambda c: c.to_eta.weight[0]), "be": st(lambda c: c.to_eta.bias[0]),
        "Wa": st(lambda c: c.to_alpha.weight[0]), "ba": st(lambda c: c.to_alpha.bias[0]),
        "beta_logit": st(lambda c: c.beta_logit),
        "W1_0": st(lambda c: c.W1_0), "W2_0": st(lambda c: c.W2_0),
        "Wr": st(lambda c: c.readout.weight), "br": st(lambda c: c.readout.bias),
        "ln_w": st(lambda c: c.ln.weight), "ln_b": st(lambda c: c.ln.bias),
        "ln_eps": c0.ln.eps,
        "chunk": int(c0.chunk), "max_write": float(c0.max_write),
    }


def stacked_init_state(P, E):
    """Learned init broadcast to E episodes: (W1, W2, M1, M2), each [L,E,...]."""
    W1 = P["W1_0"].unsqueeze(1).expand(-1, E, -1, -1)
    W2 = P["W2_0"].unsqueeze(1).expand(-1, E, -1, -1)
    return W1, W2, torch.zeros_like(W1), torch.zeros_like(W2)


def _gelu_grad(x):
    cdf = 0.5 * (1.0 + torch.erf(x / 2.0**0.5))
    pdf = torch.exp(-0.5 * x * x) / (2.0 * torch.pi) ** 0.5
    return cdf + x * pdf


def ttt_with_state_stacked(P, x, mask, state):
    """`ttt_with_state` for L cells at once.

    P     : stack_cells(...) output
    x     : [E, T, d_in]   the SAME input for every cell (pooled latents)
    mask  : [E, T]
    state : (W1, W2, M1, M2) each [L, E, ...]  (see stacked_init_state)
    returns m [L, E, T, d_out], surprise [L, E, T], new state
    """
    import torch.nn.functional as F

    L = P["W1_0"].shape[0]
    E, T, _ = x.shape
    xn = F.layer_norm(x, x.shape[-1:], eps=P["ln_in_eps"])             # [E,T,d]
    xn = xn.unsqueeze(0) * P["ln_in_w"][:, None, None] + P["ln_in_b"][:, None, None]
    K = torch.einsum("letd,lkd->letk", xn, P["Wk"]) + P["bk"][:, None, None]
    V = torch.einsum("letd,lvd->letv", xn, P["Wv"]) + P["bv"][:, None, None]
    Q = torch.einsum("letd,lkd->letk", xn, P["Wq"]) + P["bq"][:, None, None]
    eta = F.softplus(torch.einsum("letd,ld->let", xn, P["We"]) + P["be"][:, None, None])
    alpha = torch.sigmoid(torch.einsum("letd,ld->let", xn, P["Wa"]) + P["ba"][:, None, None])
    beta = torch.sigmoid(P["beta_logit"]).view(L, 1, 1, 1)
    W1, W2, M1, M2 = state
    chunk, max_write = P["chunk"], P["max_write"]

    outs, surprises = [], []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        kc, vc, qc = K[:, :, s:e], V[:, :, s:e], Q[:, :, s:e]            # [L,E,C,*]
        mc = mask[:, s:e]                                                 # [E,C]

        preq = torch.einsum("lehk,letk->leth", W1, qc)
        mq = torch.einsum("levh,leth->letv", W2, F.gelu(preq))
        outs.append(torch.einsum("letv,lov->leto", mq, P["Wr"]) + P["br"][:, None, None])

        pre = torch.einsum("lehk,letk->leth", W1, kc)
        h = F.gelu(pre)
        y = torch.einsum("levh,leth->letv", W2, h)
        err = y - vc
        dY = 2.0 * err
        dH = torch.einsum("levh,letv->leth", W2, dY)
        dPre = dH * _gelu_grad(pre)

        w = (eta[:, :, s:e] * mc).unsqueeze(-1)                           # [L,E,C,1]
        denom = mc.sum(dim=1).clamp(min=1.0).view(1, E, 1, 1)
        g1 = torch.einsum("leth,letk->lehk", dPre * w, kc) / denom
        g2 = torch.einsum("letv,leth->levh", dY * w, h) / denom
        gn = (g1.flatten(2).pow(2).sum(2) + g2.flatten(2).pow(2).sum(2) + 1e-12).sqrt()
        scale = (max_write / gn).clamp(max=1.0).view(L, E, 1, 1)
        g1, g2 = g1 * scale, g2 * scale
        surprises.append(((err.pow(2).sum(-1)) * mc).detach())            # [L,E,C]

        a = (alpha[:, :, s:e] * mc).mean(dim=2).view(L, E, 1, 1)
        M1 = beta * M1 - g1
        M2 = beta * M2 - g2
        W1 = (1.0 - a) * W1 + M1
        W2 = (1.0 - a) * W2 + M2

    m = torch.cat(outs, dim=2)
    m = F.layer_norm(m, m.shape[-1:], eps=P["ln_eps"])
    m = m * P["ln_w"][:, None, None] + P["ln_b"][:, None, None]
    return m, torch.cat(surprises, dim=2), (W1, W2, M1, M2)
