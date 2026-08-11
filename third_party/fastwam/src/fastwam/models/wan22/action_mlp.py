from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan_video_dit import sinusoidal_embedding_1d


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ActionMixerBlock(nn.Module):
    """DiT-style block with MLP-Mixer token mixing instead of attention."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        max_seq_len: int,
        eps: float,
        dropout: float,
    ):
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.token_mixer = nn.ModuleList(
            [
                nn.Linear(max_seq_len, max_seq_len*4),
                nn.GELU(approximate="tanh"),
                # nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Identity(), # remove this dropout
                nn.Linear(max_seq_len*4, max_seq_len),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ]
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            # nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Identity(), # remove this dropout
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)

    def _mix_tokens(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Joint token length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )
        x = x.transpose(1, 2)
        fc1, act, drop1, fc2, drop2 = self.token_mixer
        x = F.linear(x, fc1.weight[:seq_len, :seq_len], fc1.bias[:seq_len])
        x = drop1(act(x))
        x = F.linear(x, fc2.weight[:seq_len, :seq_len], fc2.bias[:seq_len])
        x = drop2(x)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor, cond_mod: torch.Tensor) -> torch.Tensor:
        shift_mix, scale_mix, gate_mix, shift_ffn, scale_ffn, gate_ffn = (
            self.modulation.to(dtype=cond_mod.dtype, device=cond_mod.device) + cond_mod
        ).chunk(6, dim=1)
        shift_mix = shift_mix.squeeze(1)
        scale_mix = scale_mix.squeeze(1)
        gate_mix = gate_mix.squeeze(1)
        shift_ffn = shift_ffn.squeeze(1)
        scale_ffn = scale_ffn.squeeze(1)
        gate_ffn = gate_ffn.squeeze(1)

        x = x + gate_mix.unsqueeze(1) * self._mix_tokens(
            modulate(self.norm1(x), shift_mix, scale_mix)
        )
        x = x + gate_ffn.unsqueeze(1) * self.ffn(modulate(self.norm2(x), shift_ffn, scale_ffn))
        return x


class ActionMeanPoolMLP(nn.Module):
    """Action diffusion head conditioned on mean-pooled video tokens."""

    def __init__(
        self,
        action_dim: int,
        video_hidden_dim: int,
        hidden_dim: int,
        mlp_dim: int,
        freq_dim: int,
        num_layers: int = 2,
        eps: float = 1.0e-6,
        max_action_len: int = 1024,
        max_video_len: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"`num_layers` must be >= 1, got {num_layers}")
        if max_action_len <= 0:
            raise ValueError(f"`max_action_len` must be > 0, got {max_action_len}")
        if max_video_len <= 0:
            raise ValueError(f"`max_video_len` must be > 0, got {max_video_len}")
        if hidden_dim % 2 != 0:
            raise ValueError(f"`hidden_dim` must be even for sinusoidal action positions, got {hidden_dim}")

        self.action_dim = int(action_dim)
        self.video_hidden_dim = int(video_hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.mlp_dim = int(mlp_dim)
        self.freq_dim = int(freq_dim)
        self.num_layers = int(num_layers)
        self.max_action_len = int(max_action_len)
        self.max_video_len = int(max_video_len)
        self.max_seq_len = self.max_video_len + self.max_action_len

        self.video_norm = nn.LayerNorm(video_hidden_dim, eps=eps, elementwise_affine=False)
        self.video_proj = nn.Sequential(
            nn.Linear(video_hidden_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.video_token_type = nn.Parameter(torch.randn(1, 1, hidden_dim) / hidden_dim**0.5)
        self.action_token_type = nn.Parameter(torch.randn(1, 1, hidden_dim) / hidden_dim**0.5)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))

        self.blocks = nn.ModuleList(
            [
                ActionMixerBlock(
                    hidden_dim=hidden_dim,
                    ffn_dim=mlp_dim,
                    max_seq_len=self.max_seq_len,
                    eps=eps,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, action_dim)

    @classmethod
    def from_config(
        cls,
        action_mlp_config: dict[str, Any],
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionMeanPoolMLP":
        if action_mlp_config is None:
            raise ValueError("`action_mlp_config` is required for ActionMeanPoolMLP.")
        return cls(**action_mlp_config).to(device=device, dtype=torch_dtype)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        video_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if video_tokens.ndim != 3:
            raise ValueError(
                f"`video_tokens` must be 3D [B, S, hidden_dim], got {tuple(video_tokens.shape)}"
            )
        if video_tokens.shape[0] != action_tokens.shape[0]:
            raise ValueError(
                f"Batch mismatch between action/video tokens: {action_tokens.shape[0]} vs {video_tokens.shape[0]}"
            )
        if video_tokens.shape[2] != self.video_hidden_dim:
            raise ValueError(
                f"`video_tokens` last dim must be {self.video_hidden_dim}, got {video_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got {tuple(timestep.shape)}")

        batch_size, action_seq_len, _ = action_tokens.shape
        video_seq_len = video_tokens.shape[1]
        if action_seq_len > self.max_action_len:
            raise ValueError(
                f"Action token length {action_seq_len} exceeds max_action_len={self.max_action_len}."
            )
        if video_seq_len > self.max_video_len:
            raise ValueError(
                f"Video token length {video_seq_len} exceeds max_video_len={self.max_video_len}."
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        time_emb = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        cond_mod = self.time_projection(time_emb).unflatten(1, (6, self.hidden_dim))

        video_emb = self.video_proj(self.video_norm(video_tokens))
        video_emb = video_emb + self.video_token_type.to(
            device=video_emb.device,
            dtype=video_emb.dtype,
        )
        positions = torch.arange(
            action_seq_len,
            device=action_tokens.device,
            dtype=action_tokens.dtype,
        )
        pos_emb = sinusoidal_embedding_1d(self.hidden_dim, positions).unsqueeze(0)
        action_emb = self.action_encoder(action_tokens) + pos_emb.to(
            device=action_tokens.device,
            dtype=action_tokens.dtype,
        ) + self.action_token_type.to(
            device=action_tokens.device,
            dtype=action_tokens.dtype,
        )
        x = torch.cat([video_emb, action_emb], dim=1)

        for block in self.blocks:
            x = block(x, cond_mod)


        return self.head(x[:, video_seq_len:])
