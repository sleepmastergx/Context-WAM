# Author: Rui Heng Yang

"""KV Fusion modules for MLP-Mixer-style layer fusion in FlashWAM decoupled MoT.

Author: Rui Heng Yang
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .wan_video_dit import RMSNorm
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class KVFusionMLP(nn.Module):
    """Fuses N video layers' K (or V) into a single output via layer-mixing MLP.

    Args:
        num_layers: Number of video layers to fuse (N).
        hidden_dim: MLP hidden dimension. 0 = simple learned weighted sum (Linear(N,1)),
            >0 = two-layer MLP (Linear(N,hidden) -> GELU -> Linear(hidden,1)).
        use_norm: Whether to apply RMSNorm after fusion.
        feature_dim: Feature dimension of K/V (num_heads * attn_head_dim).
        eps: Epsilon for RMSNorm.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int = 64,
        use_norm: bool = True,
        feature_dim: int = 3072,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if hidden_dim < 0:
            raise ValueError(f"hidden_dim must be >= 0, got {hidden_dim}")
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_norm = use_norm
        self.feature_dim = feature_dim

        if hidden_dim > 0:
            self.fc1 = nn.Linear(num_layers, hidden_dim)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_dim, 1)
        else:
            self.linear = nn.Linear(num_layers, 1)

        if use_norm:
            self.norm = RMSNorm(feature_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse N layers into 1.

        Args:
            x: Stacked K or V from N video layers, shape [B, Sv, N, D].

        Returns:
            Fused output, shape [B, Sv, D].
        """
        x = x.permute(0, 1, 3, 2)  # [B, Sv, D, N]

        if self.hidden_dim > 0:
            x = self.fc2(self.act(self.fc1(x)))  # [B, Sv, D, 1]
        else:
            x = self.linear(x)  # [B, Sv, D, 1]

        x = x.squeeze(-1)  # [B, Sv, D]

        if self.use_norm:
            x = self.norm(x)

        return x


class KVFusionModule(nn.Module):
    """Container for M action layers, each with separate K and V fusion MLPs.

    Args:
        num_action_layers: Number of action layers (M).
        num_video_layers: Number of video layers (N).
        attn_hidden_dim: K/V feature dimension (num_heads * attn_head_dim).
        fusion_hidden_dim: MLP hidden dimension per fusion MLP.
        fusion_use_norm: Whether to apply RMSNorm after each fusion.
        eps: Epsilon for RMSNorm.
        dtype: Optional parameter dtype for fusion weights.
    """

    def __init__(
        self,
        num_action_layers: int,
        num_video_layers: int,
        attn_hidden_dim: int,
        fusion_hidden_dim: int = 64,
        fusion_use_norm: bool = True,
        eps: float = 1e-6,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_action_layers < 1:
            raise ValueError(f"num_action_layers must be >= 1, got {num_action_layers}")
        if num_video_layers < 1:
            raise ValueError(f"num_video_layers must be >= 1, got {num_video_layers}")
        if attn_hidden_dim < 1:
            raise ValueError(f"attn_hidden_dim must be >= 1, got {attn_hidden_dim}")

        self.num_action_layers = num_action_layers
        self.num_video_layers = num_video_layers
        self.attn_hidden_dim = attn_hidden_dim

        self.k_fusions = nn.ModuleList([
            KVFusionMLP(num_video_layers, fusion_hidden_dim, fusion_use_norm, attn_hidden_dim, eps)
            for _ in range(num_action_layers)
        ])
        self.v_fusions = nn.ModuleList([
            KVFusionMLP(num_video_layers, fusion_hidden_dim, fusion_use_norm, attn_hidden_dim, eps)
            for _ in range(num_action_layers)
        ])

        if dtype is not None:
            self.to(dtype)

    def forward(
        self,
        all_k: torch.Tensor,
        all_v: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Fuse all video layers' KV into per-action-layer KV.

        Args:
            all_k: Stacked K from all video layers, shape [B, Sv, N, D].
            all_v: Stacked V from all video layers, shape [B, Sv, N, D].

        Returns:
            List of length M, each dict {"k": [B,Sv,D], "v": [B,Sv,D]}.
        """
        if all_k.ndim != 4:
            raise ValueError(
                f"all_k must have shape [B, Sv, N, D], got {tuple(all_k.shape)}"
            )
        if all_v.ndim != 4:
            raise ValueError(
                f"all_v must have shape [B, Sv, N, D], got {tuple(all_v.shape)}"
            )
        if all_k.shape != all_v.shape:
            raise ValueError(
                f"all_k and all_v must have identical shapes, got "
                f"{tuple(all_k.shape)} and {tuple(all_v.shape)}"
            )
        if all_k.shape[2] != self.num_video_layers:
            raise ValueError(
                f"all_k/all_v video-layer dimension must be {self.num_video_layers}, "
                f"got {all_k.shape[2]}"
            )
        if all_k.shape[3] != self.attn_hidden_dim:
            raise ValueError(
                f"all_k/all_v feature dimension must be {self.attn_hidden_dim}, "
                f"got {all_k.shape[3]}"
            )

        result: list[dict[str, torch.Tensor]] = []
        for i in range(self.num_action_layers):
            fused_k = self.k_fusions[i](all_k)
            fused_v = self.v_fusions[i](all_v)
            result.append({"k": fused_k, "v": fused_v})
        return result
