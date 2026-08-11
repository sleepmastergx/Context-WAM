"""Tiny stand-in for FastWAM: exercises the TRAINING LOOP, not the model.

Mirrors only the surface the trainer touches:
  * `.mot.mixtures['action'].blocks`  so tag_action_blocks/patch can attach
  * `._apply_expert_post_block(...)`  the seam the memory reads through
  * `.training_loss(sample) -> loss`  flow-matching-shaped, samples its own sigma

Everything expensive (the 5B Wan2.2 video expert, the VAE, T5) is absent. A
green smoke here means the segment/cadence/EMA/optimizer mechanics work — it
says NOTHING about the real model.
"""
import torch
import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def gate(self, x, g, branch):
        return x + branch


class _Expert(nn.Module):
    def __init__(self, n_layers, d):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(d) for _ in range(n_layers)])


class _MoT(nn.Module):
    def __init__(self, n_layers, d):
        super().__init__()
        self.mixtures = nn.ModuleDict({"action": _Expert(n_layers, d)})

    def _apply_expert_post_block(self, block, residual_x, mixed_attn_out,
                                 gate_msa, shift_mlp, scale_mlp, gate_mlp,
                                 context_payload):
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))
        return block.gate(x, gate_mlp, block.ffn(x))


class SyntheticFastWAM(nn.Module):
    def __init__(self, n_action_layers=5, hidden_dim=1024, action_dim=8,
                 latent_c=48):
        super().__init__()
        self.mot = _MoT(n_action_layers, hidden_dim)
        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.cond = nn.Linear(latent_c, hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)

    def training_loss(self, sample):
        """Flow matching: sigma ~ U(0,1) PER CALL, target = noise - clean.

        Sampling sigma here (not outside) is what makes sequence action forcing
        automatic when the trainer calls once per window.
        """
        a = sample["action"]
        lat = sample["video_latents"].mean(dim=(2, 3, 4))          # [B, C]
        noise = torch.randn_like(a)
        sigma = torch.rand(a.shape[0], 1, 1, device=a.device)
        noisy = (1 - sigma) * a + sigma * noise
        target = noise - a

        h = self.action_in(noisy) + self.cond(lat).unsqueeze(1)
        for i, blk in enumerate(self.mot.mixtures["action"].blocks):
            blk._fastwam_ttt_layer = getattr(blk, "_fastwam_ttt_layer", i)
            h = self.mot._apply_expert_post_block(
                block=blk, residual_x=h, mixed_attn_out=h,
                gate_msa=None, shift_mlp=None, scale_mlp=None, gate_mlp=None,
                context_payload=None)
        return nn.functional.mse_loss(self.head(h), target)
