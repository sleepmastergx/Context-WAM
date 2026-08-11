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
