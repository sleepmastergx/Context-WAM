"""TTT memory cell: MLP fast weights with Titans-style surprise gating.

State is the weights of a 2-layer MLP, written by gradient descent on a
self-supervised reconstruction loss, one step per frame:

    l(W; x) = || f_W(K x) - V x ||^2          K, V learned slow projections
    W_t     = (1 - a_t) W_{t-1} + M_t         a_t learned forget gate
    M_t     = b M_{t-1} - eta_t grad_W l      eta_t learned inner LR, b momentum
    m_t     = readout( f_{W_t}(Q x) )         Q learned query projection

The inner gradient is written out explicitly as tensor ops rather than taken
with torch.autograd.grad, so W_t stays a differentiable function of the slow
weights and ordinary .backward() gives the meta-gradient through the whole
episode (BPTT). That is the point of the whole design: the gradient of a
readout at frame ~500 must reach the write at frame ~30.

MINI-BATCH TTT
--------------
A strict per-frame recurrence is 600 sequential Python steps per rollout, which
at ~235 steps/epoch x 600 epochs is far too slow. Following Sun et al. 2024 we
process the episode in chunks of `chunk`: every inner gradient inside a chunk is
taken at the chunk-start weights, so they compute in parallel, and the
momentum/forget recurrence advances once per chunk. Readouts within a chunk use
the chunk-start weights, i.e. the memory lags by at most `chunk` frames (~0.5 s
at chunk=16) -- negligible against the ~500-frame dependency being tested.

Set chunk=1 to recover the exact per-frame recurrence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTTMemory(nn.Module):

    def __init__(
        self,
        d_in: int,
        d_k: int = 128,
        d_v: int = 128,
        d_hidden: int = 512,
        d_out: int = 256,
        chunk: int = 16,
        eta_init: float = -6.0,    # softplus(-6) ~ 0.0025
        alpha_init: float = -4.0,  # sigmoid(-4) ~ 0.018, little forgetting at init
        beta_init: float = 0.0,    # sigmoid(0) = 0.5, ~2x momentum amplification
        max_write: float = 1.0,    # per-sample cap on ||grad||_F, see forward()
    ):
        super().__init__()
        self.d_k, self.d_v, self.d_hidden, self.d_out = d_k, d_v, d_hidden, d_out
        self.chunk = chunk
        self.max_write = max_write

        # Input LayerNorm: keeps k/v/q -- and therefore the inner reconstruction
        # problem -- at a fixed scale regardless of how the frozen encoder's
        # feature magnitudes drift across an episode.
        self.ln_in = nn.LayerNorm(d_in)

        # --- slow weights: what counts as a key, a value, a query -------------
        self.to_k = nn.Linear(d_in, d_k)
        self.to_v = nn.Linear(d_in, d_v)
        self.to_q = nn.Linear(d_in, d_k)

        # --- slow weights: write control (data-dependent, per Titans) ---------
        self.to_eta = nn.Linear(d_in, 1)
        self.to_alpha = nn.Linear(d_in, 1)
        nn.init.zeros_(self.to_eta.weight)
        nn.init.constant_(self.to_eta.bias, eta_init)
        nn.init.zeros_(self.to_alpha.weight)
        nn.init.constant_(self.to_alpha.bias, alpha_init)
        self.beta_logit = nn.Parameter(torch.tensor(float(beta_init)))

        # --- learned initial fast weights (the state at episode start) --------
        self.W1_0 = nn.Parameter(torch.randn(d_hidden, d_k) / d_k**0.5)
        self.W2_0 = nn.Parameter(torch.randn(d_v, d_hidden) / d_hidden**0.5)

        self.readout = nn.Linear(d_v, d_out)
        self.ln = nn.LayerNorm(d_out)

    # ---- inner model -------------------------------------------------------
    @staticmethod
    def _inner_forward(W1, W2, k):
        """f_W(k) for a batch. W1 (B,H,dk), W2 (B,dv,H), k (B,dk)."""
        pre = torch.einsum("bhk,bk->bh", W1, k)
        h = F.gelu(pre)
        y = torch.einsum("bvh,bh->bv", W2, h)
        return pre, h, y

    def _inner_grad(self, W1, W2, k, v):
        """d/dW of ||f_W(k) - v||^2, written out so autograd can see through it.

        Shapes: k (B,C,dk), v (B,C,dv) for a chunk of C frames evaluated at the
        same (W1, W2). Returns per-frame grads summed over the chunk.
        """
        B, C, _ = k.shape
        pre = torch.einsum("bhk,bck->bch", W1, k)          # (B,C,H)
        h = F.gelu(pre)
        y = torch.einsum("bvh,bch->bcv", W2, h)            # (B,C,dv)
        err = y - v                                        # (B,C,dv)

        dY = 2.0 * err                                     # dl/dy
        # dl/dW2 = dY (x) h ; dl/dW1 = (W2^T dY * gelu'(pre)) (x) k
        dH = torch.einsum("bvh,bcv->bch", W2, dY)
        dPre = dH * _gelu_grad(pre)
        return dPre, dY, h, k, err

    def forward(self, x, mask=None):
        """x (B,T,d_in), mask (B,T) 1 where the frame is real.

        Returns m (B,T,d_out) and a dict of diagnostics.
        """
        B, T, _ = x.shape
        if mask is None:
            mask = torch.ones(B, T, device=x.device, dtype=x.dtype)

        x = self.ln_in(x)
        K = self.to_k(x)
        V = self.to_v(x)
        Q = self.to_q(x)
        eta = F.softplus(self.to_eta(x)).squeeze(-1)       # (B,T) >0
        alpha = torch.sigmoid(self.to_alpha(x)).squeeze(-1)  # (B,T) in (0,1)
        beta = torch.sigmoid(self.beta_logit)

        W1 = self.W1_0.unsqueeze(0).expand(B, -1, -1)
        W2 = self.W2_0.unsqueeze(0).expand(B, -1, -1)
        M1 = torch.zeros_like(W1)
        M2 = torch.zeros_like(W2)

        outs, surprises = [], []
        for s in range(0, T, self.chunk):
            e = min(s + self.chunk, T)
            kc, vc, qc = K[:, s:e], V[:, s:e], Q[:, s:e]
            mc = mask[:, s:e]                               # (B,C)

            # --- read at chunk-start weights (all frames in the chunk) -------
            preq = torch.einsum("bhk,bck->bch", W1, qc)
            hq = F.gelu(preq)
            mq = torch.einsum("bvh,bch->bcv", W2, hq)
            outs.append(self.readout(mq))

            # --- write: inner grads for the whole chunk, at chunk-start W ----
            dPre, dY, h, kk, err = self._inner_grad(W1, W2, kc, vc)
            w = (eta[:, s:e] * mc).unsqueeze(-1)            # (B,C,1) 0 on padding
            # MEAN over the chunk, not sum: `chunk` is a compute knob and must
            # not silently multiply the inner learning rate by its size.
            denom = mc.sum(dim=1).clamp(min=1.0).view(B, 1, 1)
            g1 = torch.einsum("bch,bck->bhk", dPre * w, kk) / denom
            g2 = torch.einsum("bcv,bch->bvh", dY * w, h) / denom

            # Per-sample cap on the write. Surprise still scales the update
            # below the cap; the cap only stops the runaway feedback loop
            # (big write -> bad reconstruction -> bigger write -> inf).
            # eps INSIDE the sqrt: a fully-masked chunk gives gn == 0 exactly,
            # and d/dx sqrt(x) is infinite there, which NaNs the backward pass
            # even though the forward looks fine.
            gn = (g1.flatten(1).pow(2).sum(1) + g2.flatten(1).pow(2).sum(1) + 1e-12).sqrt()
            scale = (self.max_write / gn).clamp(max=1.0).view(B, 1, 1)
            g1, g2 = g1 * scale, g2 * scale

            surprises.append(((err.pow(2).sum(-1)) * mc).detach())

            # --- momentum + forget, once per chunk --------------------------
            a = (alpha[:, s:e] * mc).mean(dim=1).view(B, 1, 1)
            M1 = beta * M1 - g1
            M2 = beta * M2 - g2
            W1 = (1.0 - a) * W1 + M1
            W2 = (1.0 - a) * W2 + M2

        m = self.ln(torch.cat(outs, dim=1))
        diag = {
            "surprise": torch.cat(surprises, dim=1),
            "eta": eta.detach(),
            "alpha": alpha.detach(),
            "beta": beta.detach(),
        }
        return m, diag


def _gelu_grad(x):
    """d/dx of the exact (erf) GELU, matching F.gelu's default."""
    cdf = 0.5 * (1.0 + torch.erf(x / 2.0**0.5))
    pdf = torch.exp(-0.5 * x * x) / (2.0 * torch.pi) ** 0.5
    return cdf + x * pdf
