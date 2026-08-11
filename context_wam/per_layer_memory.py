"""RoboTTT-style per-layer TTT memory for Fast-WAM's action expert.

Placement follows RoboTTT (arXiv 2607.15275): one TTT state per action-expert
DiT layer, fused into the attention output with a near-zero gate

    O = tanh(alpha_l) * O_TTT_l + O_attn_l          alpha_l init 1e-3

so at init the model is bit-identical to stock Fast-WAM. In Fast-WAM the seam is
`MoT._apply_expert_post_block` (mot.py:111-133), where `mixed_attn_out` IS
O_attn — self_attn.forward is never called in MoT mode (Q/K/V are taken from the
block and run through one joint SDPA), so a module forward-hook would never fire.

THE WRITE/READ SPLIT (the fix this file exists for)
---------------------------------------------------
Fast-WAM trains flow matching with ONE forward pass per window
(`add_noise` once, fastwam.py:645) but infers with `num_inference_steps=20`
Euler steps (fastwam.py:907, 1057). If the fast weights updated per forward
pass, deploy would advance the state 20x faster than training ever saw —
the same train/deploy operator mismatch that cost 92% on RMBench.

So:
    advance(...)   ONCE per window, before denoising, from OBSERVATION content
                   (pooled video latents + proprio). Updates all L states.
    read_layer(l)  during every denoising step. Pure read, no state change.

`assert_write_once` in check_write_once.py gates exactly this: the state after a
1-step call and after a 20-step call must be identical.
"""
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from memory import ttt_with_state, detach_state          # noqa: F401
from ttt_cell import TTTMemory


class PerLayerEpisodeMemory(nn.Module):

    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,                 # action expert width (1024)
        latent_channels: int = 48,
        proprio_dim: Optional[int] = None,
        d_k: int = 128,
        d_v: int = 128,
        d_hidden: int = 512,
        d_out: int = 256,
        chunk: int = 1,                  # one inner step per latent frame
        gate_init: float = 1e-3,
        share_cell: bool = False,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        self.proprio_dim = proprio_dim
        d_in = latent_channels + (proprio_dim or 0)
        self.share_cell = bool(share_cell)

        n_cells = 1 if share_cell else self.n_layers
        self.cells = nn.ModuleList([
            TTTMemory(d_in=d_in, d_k=d_k, d_v=d_v, d_hidden=d_hidden,
                      d_out=d_out, chunk=chunk)
            for _ in range(n_cells)
        ])
        # per-layer readout -> action-expert width, gated near zero
        self.to_hidden = nn.ModuleList([
            nn.Linear(d_out, hidden_dim) for _ in range(self.n_layers)
        ])
        self.alpha = nn.Parameter(torch.full((self.n_layers,), float(gate_init)))

        self._states: List[Optional[Tuple[torch.Tensor, ...]]] = [None] * self.n_layers
        self._m: List[Optional[torch.Tensor]] = [None] * self.n_layers

    # ---- episode lifecycle -------------------------------------------------
    def reset(self, batch_size: Optional[int] = None):
        """Call at episode start. Fast weights fall back to the learned init."""
        self._states = [None] * self.n_layers
        self._m = [None] * self.n_layers

    def _cell(self, layer: int) -> TTTMemory:
        return self.cells[0] if self.share_cell else self.cells[layer]

    def pool_latents(self, video_latents, proprio=None):
        """[B, C, t, h, w] -> [B, t, d_in]; t inner steps per window."""
        if video_latents.ndim != 5:
            raise ValueError(
                f"video_latents must be 5D [B,C,t,h,w], got {tuple(video_latents.shape)}")
        x = video_latents.mean(dim=(3, 4)).transpose(1, 2)
        if self.proprio_dim is not None:
            if proprio is None:
                raise ValueError("built with proprio_dim but got proprio=None")
            x = torch.cat([x, proprio.unsqueeze(1).expand(-1, x.shape[1], -1)], dim=-1)
        return x

    # ---- WRITE: exactly once per window ------------------------------------
    def advance(self, video_latents, proprio=None, detach_carry: bool = True):
        """Advance every layer's fast weights by one window.

        MUST be called once per window, outside the denoising loop. Returns the
        mean surprise for logging.
        """
        x0 = self.pool_latents(video_latents, proprio)
        mask = torch.ones(x0.shape[0], x0.shape[1], device=x0.device, dtype=x0.dtype)
        surprise = 0.0
        for l in range(self.n_layers):
            cell = self._cell(l)
            x = x0.to(cell.ln_in.weight.dtype)
            m, diag, state = ttt_with_state(cell, x, mask, self._states[l])
            # truncated BPTT: values cross the window boundary, the graph does not
            self._states[l] = detach_state(state) if detach_carry else state
            self._m[l] = m[:, -1]
            surprise += float(diag["surprise"].mean())
        return surprise / max(self.n_layers, 1)

    # ---- READ: pure, called on every denoising step ------------------------
    def read_layer(self, layer: int, seq_len: int, dtype: torch.dtype):
        """Gated per-layer contribution, broadcast over the action tokens.

        Returns [B, seq_len, hidden_dim] to add to `mixed_attn_out`, or None
        before the first advance() (then the block is stock Fast-WAM).
        """
        m = self._m[layer]
        if m is None:
            return None
        h = self.to_hidden[layer](m).unsqueeze(1)              # [B, 1, D]
        return (torch.tanh(self.alpha[layer]) * h).to(dtype).expand(-1, seq_len, -1)


def patch_mot_action_expert(mot, memory: PerLayerEpisodeMemory,
                            action_key: str = "action"):
    """Wrap MoT._apply_expert_post_block so the ACTION expert reads the memory.

    Fast-WAM's checkout is read-only, so this rebinds the method on the instance
    rather than editing their source. The video expert is untouched, which is why
    the video KV cache stays valid.
    """
    original = mot._apply_expert_post_block
    layer_counter = {"i": 0, "expected": memory.n_layers}

    def wrapped(block, residual_x, mixed_attn_out, gate_msa, shift_mlp,
                scale_mlp, gate_mlp, context_payload, __orig=original):
        expert = getattr(block, "_fastwam_ttt_expert", None)
        layer = getattr(block, "_fastwam_ttt_layer", None)
        if expert == action_key and layer is not None:
            add = memory.read_layer(layer, mixed_attn_out.shape[1],
                                    mixed_attn_out.dtype)
            if add is not None:
                mixed_attn_out = mixed_attn_out + add
        return __orig(block=block, residual_x=residual_x,
                      mixed_attn_out=mixed_attn_out, gate_msa=gate_msa,
                      shift_mlp=shift_mlp, scale_mlp=scale_mlp,
                      gate_mlp=gate_mlp, context_payload=context_payload)

    mot._apply_expert_post_block = wrapped
    mot._fastwam_ttt_layers = layer_counter
    return mot


def tag_action_blocks(action_expert, action_key: str = "action"):
    """Stamp each action-expert block with its index so the wrapper can route.

    Done once at build time; without it the wrapper cannot tell which layer's
    memory to read and silently falls through to stock behaviour.
    """
    for i, block in enumerate(action_expert.blocks):
        block._fastwam_ttt_expert = action_key
        block._fastwam_ttt_layer = i
    return action_expert
