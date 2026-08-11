# Author: Rui Heng Yang

"""MoTDecoupled -- Mixture-of-Transformers with asymmetric layer counts.

Author: Rui Heng Yang

This module implements a decoupled variant of the MoT architecture where the
video expert and action expert can have different numbers of transformer layers
(e.g., video=30, action=5). The action expert cross-attends to video K/V
selected by ``kv_source_mapping``, rather than receiving per-layer K/V from a
lock-step joint forward pass. Supports kv_source_mode: final_only,
uniform_end, uniform_middle, fused_kv, new_fused_kv and fused_mlp. In
``fused_kv``/``new_fused_kv`` mode, each action layer and attention head
learns one shared K/V weighted sum over the mapped video K/V sources,
followed by separate K/V flattened channel projections. The fused_mlp mode uses a learned
KV fusion module (injected via the ``kv_fusion`` parameter) to combine all
video layers' K/V rather than selecting from a single static layer.

The class inherits all per-block helper methods from MoT (e.g.,
``_build_expert_attention_io``, ``_mixed_attention``, ``_apply_expert_post_block``,
``_apply_post_with_optional_checkpoint``, ``_split_modulation``) and adds three
new methods for the decoupled forward path:

- ``forward_decoupled`` -- training forward with mapped video K/V
- ``prefill_video_kv`` -- inference video prefill (cache all video-layer KV)
- ``forward_action_with_video_kv`` -- inference action denoising with cached KV
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from .mot import MoT
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

# Expert key constants used to index into the mixtures dict.
VIDEO_EXPERT_KEY = "video"
ACTION_EXPERT_KEY = "action"


class MLPMixerFusedKVBlock(nn.Module):
    """One residual MLP-Mixer block for layer-wise video KV fusion."""

    def __init__(
        self,
        *,
        num_layer_tokens: int,
        hidden_dim: int,
        token_mlp_ratio: float,
        channel_mlp_ratio: float,
        norm_cls: type[nn.Module],
        norm_eps: float,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        token_hidden_dim = max(1, round(num_layer_tokens * token_mlp_ratio))
        channel_hidden_dim = max(1, round(hidden_dim * channel_mlp_ratio))

        # The MLP itself is shared between K and V; their RMSNorms remain
        # separate because the two tensors have different feature statistics.
        self.k_token_norm = norm_cls(hidden_dim, eps=norm_eps)
        self.v_token_norm = norm_cls(hidden_dim, eps=norm_eps)
        self.shared_token_mlp = nn.Sequential(
            nn.Linear(num_layer_tokens, token_hidden_dim),
            nn.GELU(),
            nn.Linear(token_hidden_dim, num_layer_tokens),
        )
        self.k_channel_norm = norm_cls(hidden_dim, eps=norm_eps)
        self.v_channel_norm = norm_cls(hidden_dim, eps=norm_eps)
        self.k_channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, channel_hidden_dim),
            nn.GELU(),
            nn.Linear(channel_hidden_dim, hidden_dim),
        )
        self.v_channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, channel_hidden_dim),
            nn.GELU(),
            nn.Linear(channel_hidden_dim, hidden_dim),
        )
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def _token_mix(self, x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        # x: [layers, batch, tokens, channels]. The token MLP acts on layers.
        normalized = norm(x).permute(1, 2, 3, 0)
        return self.shared_token_mlp(normalized).permute(3, 0, 1, 2)

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mix stacked layer and channel axes without changing K/V shapes."""
        k = k + self._token_mix(k, self.k_token_norm)
        v = v + self._token_mix(v, self.v_token_norm)
        k = k + self.k_channel_mlp(self.k_channel_norm(k))
        v = v + self.v_channel_mlp(self.v_channel_norm(v))
        return k, v


def compute_kv_source_mapping(
    mode: str,
    video_num_layers: int,
    action_num_layers: int,
) -> list[int]:
    """Compute which video layer each action layer cross-attends to.

    Author: Rui Heng Yang

    Args:
        mode: KV source strategy.
            ``"final_only"`` -- all action layers read from the final video layer.
            ``"uniform_end"`` -- divide N video layers into M equal segments,
            pick the last layer of each segment.
            ``"uniform_middle"`` -- divide N video layers into M equal segments,
            pick the center layer of each segment.
            ``"fused_kv"`` -- learn a weighted sum over all video-layer K/V.
            ``"new_fused_kv"`` -- same layer/channel projection as fused_kv,
            with fixed-RoPE variants caching pre-RoPE K at a projection-dependent
            pre- or post-norm tap and applying a post-projection key-only RMSNorm.
        video_num_layers: Number of video backbone layers (N).
        action_num_layers: Number of action expert layers (M).

    Returns:
        For non-fused modes, a list of length M where entry k is the video
        layer index that action layer k cross-attends to. For fused modes,
        a source list used by every action layer's learned weighted sum.

    Examples:
        >>> compute_kv_source_mapping("final_only", 30, 5)
        [29, 29, 29, 29, 29]
        >>> compute_kv_source_mapping("uniform_end", 30, 5)
        [5, 11, 17, 23, 29]
        >>> compute_kv_source_mapping("uniform_end", 30, 1)
        [29]
        >>> compute_kv_source_mapping("uniform_middle", 30, 5)
        [2, 8, 14, 20, 26]
        >>> compute_kv_source_mapping("uniform_middle", 30, 1)
        [14]
        >>> compute_kv_source_mapping("fused_kv", 30, 5)
        [0, 1, 2, ..., 29]
    """
    N, M = video_num_layers, action_num_layers
    if mode == "final_only":
        mapping = [N - 1] * M
    elif mode == "uniform_end":
        if M > N:
            raise ValueError(
                f"uniform_end requires action_num_layers <= video_num_layers, "
                f"got M={M} > N={N}"
            )
        mapping = [round((k + 1) * N / M) - 1 for k in range(M)]
    elif mode == "uniform_middle":
        if M > N:
            raise ValueError(
                f"uniform_middle requires action_num_layers <= video_num_layers, "
                f"got M={M} > N={N}"
            )
        mapping = []
        for k in range(M):
            start = round(k * N / M)
            end = round((k + 1) * N / M) - 1
            mapping.append((start + end) // 2)
    elif mode in {"fused_kv", "new_fused_kv"}:
        mapping = list(range(N))
    else:
        raise ValueError(
            f"Unknown kv_source mode: {mode!r}. "
            f"Expected 'final_only', 'uniform_end', 'uniform_middle', "
            f"'fused_kv', or 'new_fused_kv'."
        )
    if mode not in {"fused_kv", "new_fused_kv"}:
        assert len(mapping) == M
    else:
        assert len(mapping) > 0

    assert all(0 <= v < N for v in mapping), (
        f"Mapping values out of range [0, {N}): {mapping}"
    )
    return mapping


class MoTDecoupled(MoT):
    """Mixture-of-Transformers with decoupled (asymmetric) layer counts.

    Unlike the base ``MoT`` which requires all experts to share the same
    number of transformer layers, ``MoTDecoupled`` allows the video expert
    to have ``video_num_layers`` layers and the action expert to have
    ``action_num_layers`` layers (typically fewer, e.g. 5 vs 30).

    The action expert cross-attends to video layer K/V using concatenated SDPA
    with a rectangular attention mask of shape ``[Sa, Sv + Sa]``. In
    ``fused_kv`` mode, each action layer learns one K/V-shared distribution per
    attention head over ``kv_source_mapping``, then applies separate K/V linear
    projections over each head's channels.
    The action expert cross-attends to video K/V using one of two routing
    mechanisms: selected source layers from ``kv_source_mapping`` (for
    final_only, uniform_end, and uniform_middle) or learned fused K/V from all
    video layers when ``kv_fusion`` is provided by ``kv_source_mode="fused_mlp"``.
    Both paths use concatenated SDPA with a rectangular attention mask of shape
    ``[Sa, Sv + Sa]``.

    Inherited base-class methods that assume lock-step layer iteration
    (``forward``, ``prefill_video_cache``, ``forward_action_with_video_cache``)
    are overridden to raise ``NotImplementedError``.

    Args:
        mixtures: Dict mapping expert names to expert modules. Must contain
            both ``"video"`` and ``"action"`` keys.
        video_num_layers: Number of transformer layers for the video expert.
        action_num_layers: Number of transformer layers for the action expert.
        num_heads: Number of attention heads. Must be identical between experts
            (hard constraint from concatenated SDPA reshape).
        attn_head_dim: Per-head dimension. Must be identical between experts.
        mot_checkpoint_mixed_attn: Whether to apply gradient checkpointing to
            the mixed-attention SDPA call.
        kv_source_mode: KV source strategy. ``"fused_kv"`` enables learned
            weighted sums over ``kv_source_mapping``.
    """

    enable_new_fused_kv_key_norm: bool = False
    supported_new_fused_kv_projection_modes = {
        "full",
        "simple",
        "simple+PE",
        "simple+PE-postnorm",
        "per_head_channel",
        "HeadFusedKV",
        "HeadFusedKV+Sin2DPE",
        "simple_head_fused",
        "simple_head_softmax",
        "MLPMixerFusedKV",
    }
    supported_new_fused_kv_simple_head_softmax_fuse_modes = {
        "all",
        "shared",
        "uniform_end",
        "uni_end",
    }
    supported_new_fused_kv_head_fused_kv_fuse_modes = {
        "all",
        "shared",
    }

    def __init__(
        self,
        mixtures: dict[str, nn.Module],
        video_num_layers: int,
        action_num_layers: int,
        num_heads: int,
        attn_head_dim: int,
        mot_checkpoint_mixed_attn: bool = True,
        kv_source_mapping: list[int] | None = None,
        kv_source_mode: str = "final_only",
        new_fused_kv_projection_mode: str = "full",
        new_fused_kv_pos_embed_max_tokens: int = 4096,
        new_fused_kv_pos_embed_dim: int = 128,
        new_fused_kv_mlp_mixer_num_blocks: int = 1,
        new_fused_kv_mlp_mixer_token_mlp_ratio: float = 4.0,
        new_fused_kv_mlp_mixer_channel_mlp_ratio: float = 4.0,
        new_fused_kv_head_fused_kv_sin2d_pe_mlp_hidden_dim: int = 512,
        new_fused_kv_simple_head_softmax_fuse_mode: str = "all",
        new_fused_kv_head_fused_kv_fuse_mode: str = "all",
        kv_fusion: "nn.Module | None" = None,
    ) -> None:
        # Bypass MoT.__init__ to avoid the equal-layer-count check (mot.py:39-41).
        # We call nn.Module.__init__ directly so that PyTorch module registration
        # (e.g., nn.ModuleDict) still works correctly.
        nn.Module.__init__(self)

        # ---- Validate mixtures dict ----
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if VIDEO_EXPERT_KEY not in mixtures or ACTION_EXPERT_KEY not in mixtures:
            raise ValueError(
                "`mixtures` must include both 'video' and 'action' experts."
            )

        # ---- Store as nn.ModuleDict (critical for parameter tracking) ----
        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())

        # ---- Store layer counts per expert ----
        self.video_num_layers = video_num_layers
        self.action_num_layers = action_num_layers

        # Validate that the experts actually have the expected number of blocks.
        video_expert = self.mixtures[VIDEO_EXPERT_KEY]
        action_expert = self.mixtures[ACTION_EXPERT_KEY]

        if len(video_expert.blocks) != video_num_layers:
            raise ValueError(
                f"Video expert has {len(video_expert.blocks)} blocks but "
                f"video_num_layers={video_num_layers}"
            )
        if len(action_expert.blocks) != action_num_layers:
            raise ValueError(
                f"Action expert has {len(action_expert.blocks)} blocks but "
                f"action_num_layers={action_num_layers}"
            )

        # ---- Enforce matching head structure (hard constraint from concat SDPA) ----
        # See design doc Section 3.5: both experts must use the same num_heads
        # and attn_head_dim so that the single reshape in flash_attention does
        # not scramble head boundaries.
        for name in self.expert_order:
            expert = self.mixtures[name]
            if expert.num_heads != num_heads:
                raise ValueError(
                    f"Expert '{name}' has num_heads={expert.num_heads} but "
                    f"MoTDecoupled requires num_heads={num_heads}"
                )
            if expert.attn_head_dim != attn_head_dim:
                raise ValueError(
                    f"Expert '{name}' has attn_head_dim={expert.attn_head_dim} "
                    f"but MoTDecoupled requires attn_head_dim={attn_head_dim}"
                )

        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn

        if mot_checkpoint_mixed_attn:
            logger.info(
                "MoTDecoupled: Using gradient checkpointing for mixed attention."
            )

        # NOTE: We intentionally do NOT set self.num_layers. This prevents any
        # inherited method that references self.num_layers from silently
        # operating on a wrong layer count.

        supported_kv_source_modes = {
            "final_only",
            "uniform_end",
            "uniform_middle",
            "fused_kv",
            "new_fused_kv",
            "fused_mlp",
        }
        if kv_source_mode not in supported_kv_source_modes:
            raise ValueError(
                f"Unknown kv_source_mode {kv_source_mode!r}; expected one of "
                f"{sorted(supported_kv_source_modes)}"
            )
        self.kv_source_mode = kv_source_mode
        if new_fused_kv_projection_mode not in self.supported_new_fused_kv_projection_modes:
            raise ValueError(
                f"Unknown new_fused_kv_projection_mode {new_fused_kv_projection_mode!r}; "
                f"expected one of {sorted(self.supported_new_fused_kv_projection_modes)}"
            )
        if kv_source_mode != "new_fused_kv" and new_fused_kv_projection_mode != "full":
            raise ValueError(
                "new_fused_kv_projection_mode is only configurable when "
                "kv_source_mode=new_fused_kv; use full for other modes."
            )
        self.new_fused_kv_projection_mode = new_fused_kv_projection_mode
        if (
            new_fused_kv_simple_head_softmax_fuse_mode
            not in self.supported_new_fused_kv_simple_head_softmax_fuse_modes
        ):
            raise ValueError(
                "Unknown new_fused_kv_simple_head_softmax_fuse_mode "
                f"{new_fused_kv_simple_head_softmax_fuse_mode!r}; expected one of "
                f"{sorted(self.supported_new_fused_kv_simple_head_softmax_fuse_modes)}"
            )
        if new_fused_kv_simple_head_softmax_fuse_mode == "uni_end":
            new_fused_kv_simple_head_softmax_fuse_mode = "uniform_end"
        if (
            new_fused_kv_simple_head_softmax_fuse_mode != "all"
            and new_fused_kv_projection_mode != "simple_head_softmax"
        ):
            raise ValueError(
                "new_fused_kv_simple_head_softmax_fuse_mode is only configurable "
                "when new_fused_kv_projection_mode='simple_head_softmax'; use "
                "'all' for other projection modes."
            )
        self.new_fused_kv_simple_head_softmax_fuse_mode = (
            new_fused_kv_simple_head_softmax_fuse_mode
        )
        if (
            new_fused_kv_head_fused_kv_fuse_mode
            not in self.supported_new_fused_kv_head_fused_kv_fuse_modes
        ):
            raise ValueError(
                "Unknown new_fused_kv_head_fused_kv_fuse_mode "
                f"{new_fused_kv_head_fused_kv_fuse_mode!r}; expected one of "
                f"{sorted(self.supported_new_fused_kv_head_fused_kv_fuse_modes)}"
            )
        if (
            new_fused_kv_head_fused_kv_fuse_mode != "all"
            and new_fused_kv_projection_mode
            not in {"HeadFusedKV", "HeadFusedKV+Sin2DPE"}
        ):
            raise ValueError(
                "new_fused_kv_head_fused_kv_fuse_mode is only configurable "
                "when new_fused_kv_projection_mode is 'HeadFusedKV' or "
                "'HeadFusedKV+Sin2DPE'; use 'all' for other projection modes."
            )
        self.new_fused_kv_head_fused_kv_fuse_mode = (
            new_fused_kv_head_fused_kv_fuse_mode
        )
        self.new_fused_kv_pos_embed_max_tokens = int(new_fused_kv_pos_embed_max_tokens)
        if self.new_fused_kv_pos_embed_max_tokens <= 0:
            raise ValueError("new_fused_kv_pos_embed_max_tokens must be positive")
        self.new_fused_kv_pos_embed_dim = int(new_fused_kv_pos_embed_dim)
        if self.new_fused_kv_pos_embed_dim <= 0:
            raise ValueError("new_fused_kv_pos_embed_dim must be positive")
        if self.new_fused_kv_pos_embed_dim % 4 != 0:
            raise ValueError("new_fused_kv_pos_embed_dim must be divisible by 4")
        self.new_fused_kv_mlp_mixer_num_blocks = int(new_fused_kv_mlp_mixer_num_blocks)
        if self.new_fused_kv_mlp_mixer_num_blocks <= 0:
            raise ValueError("new_fused_kv_mlp_mixer_num_blocks must be positive")
        self.new_fused_kv_mlp_mixer_token_mlp_ratio = float(
            new_fused_kv_mlp_mixer_token_mlp_ratio
        )
        if self.new_fused_kv_mlp_mixer_token_mlp_ratio <= 0:
            raise ValueError("new_fused_kv_mlp_mixer_token_mlp_ratio must be positive")
        self.new_fused_kv_mlp_mixer_channel_mlp_ratio = float(
            new_fused_kv_mlp_mixer_channel_mlp_ratio
        )
        if self.new_fused_kv_mlp_mixer_channel_mlp_ratio <= 0:
            raise ValueError("new_fused_kv_mlp_mixer_channel_mlp_ratio must be positive")
        self.new_fused_kv_head_fused_kv_sin2d_pe_mlp_hidden_dim = int(
            new_fused_kv_head_fused_kv_sin2d_pe_mlp_hidden_dim
        )
        if self.new_fused_kv_head_fused_kv_sin2d_pe_mlp_hidden_dim <= 0:
            raise ValueError(
                "new_fused_kv_head_fused_kv_sin2d_pe_mlp_hidden_dim must be positive"
            )

        # ---- Guard: new_fused_kv requires the action-aligned 3D-RoPE subclass ----
        # new_fused_kv is DEFINED by the action-aligned 3D-RoPE path plus the
        # fused-key RMSNorm, both of which live only in the subclass that sets
        # enable_new_fused_kv_key_norm=True (MoTDecoupledActionAlignedVideoRoPE,
        # fasterwam_decoupled.py). The base class lacks that RoPE path and never
        # allocates k_fused_norm, so constructing base MoTDecoupled with
        # new_fused_kv would SILENTLY train a structurally different model. We key
        # the predicate on the class ATTRIBUTE (not the class name) so any future
        # subclass that opts in via enable_new_fused_kv_key_norm=True still works.
        if (
            kv_source_mode == "new_fused_kv"
            and type(self).enable_new_fused_kv_key_norm is False
        ):
            raise ValueError(
                "kv_source_mode='new_fused_kv' is only supported by the "
                "action-aligned fixed-RoPE subclass "
                "(MoTDecoupledActionAlignedVideoRoPE, i.e. enable "
                "enable_new_fused_kv_key_norm=True / set fixed_rope=true); the "
                "base MoTDecoupled lacks the action-aligned 3D-RoPE path and the "
                "fused-key norm that define new_fused_kv, so constructing it here "
                "would silently train a different model. Use the fixed-RoPE class "
                "(fixed_rope=true) or choose kv_source_mode='fused_kv' instead."
            )

        if kv_source_mapping is None:
            if kv_source_mode in {"fused_kv", "new_fused_kv"}:
                kv_source_mapping = list(range(video_num_layers))
            else:
                kv_source_mapping = [video_num_layers - 1] * action_num_layers

        if (
            kv_source_mode not in {"fused_kv", "new_fused_kv"}
            and len(kv_source_mapping) != action_num_layers
        ):
            raise ValueError(
                f"kv_source_mapping length {len(kv_source_mapping)} != "
                f"action_num_layers {action_num_layers}"
            )
        if kv_source_mode in {"fused_kv", "new_fused_kv"} and len(kv_source_mapping) == 0:
            raise ValueError(f"{kv_source_mode} requires at least one kv_source_mapping entry")
        if (
            kv_source_mode == "new_fused_kv"
            and new_fused_kv_projection_mode == "simple_head_softmax"
            and self.new_fused_kv_simple_head_softmax_fuse_mode == "uniform_end"
            and len(kv_source_mapping) < action_num_layers
        ):
            raise ValueError(
                "new_fused_kv_simple_head_softmax_fuse_mode='uniform_end' "
                "requires at least as many selected video layers as action "
                f"layers, got {len(kv_source_mapping)} video sources for "
                f"{action_num_layers} action layers."
            )
        if any(v < 0 or v >= video_num_layers for v in kv_source_mapping):
            raise ValueError(
                f"kv_source_mapping values must be in [0, {video_num_layers}), "
                f"got {kv_source_mapping}"
            )
        if list(kv_source_mapping) != sorted(kv_source_mapping):
            raise ValueError(
                f"kv_source_mapping must be sorted ascending, "
                f"got {kv_source_mapping}"
            )
        self.kv_source_mapping: list[int] = list(kv_source_mapping)

        self._action_schedule: dict[int, list[int]] = {}
        if kv_source_mode in {"fused_kv", "new_fused_kv"}:
            for video_idx in self.kv_source_mapping:
                self._action_schedule[video_idx] = list(range(action_num_layers))
            attn_hidden_dim = num_heads * attn_head_dim
            self.k_fused_norm = nn.ModuleList([
                self._make_attn_rms_norm(video_expert)
                for _ in range(action_num_layers)
            ]) if (
                kv_source_mode == "new_fused_kv"
                and self.enable_new_fused_kv_key_norm
            ) else None

            if (
                kv_source_mode == "new_fused_kv"
                and self.new_fused_kv_projection_mode in {
                    "simple",
                    "simple+PE",
                    "simple+PE-postnorm",
                }
            ):
                # Simple ablation: one shared layer-fusion scalar per source
                # video layer and action layer. K/V share this [N, M] matrix,
                # and no flattened channel projection is applied. The simple+PE
                # variants add fixed 2D sin/cos spatial PE through a learned
                # projection either before or after the K-only RMSNorm.
                self.simple_kv_fusing_layer = nn.Parameter(
                    torch.zeros(len(self.kv_source_mapping), action_num_layers)
                )
                nn.init.trunc_normal_(
                    self.simple_kv_fusing_layer,
                    mean=1.0 / len(self.kv_source_mapping),
                    std=0.02,
                )
                if self.new_fused_kv_projection_mode in {
                    "simple+PE",
                    "simple+PE-postnorm",
                }:
                    self.k_video_pos_projection = nn.Parameter(
                        torch.empty(
                            action_num_layers,
                            self.new_fused_kv_pos_embed_dim,
                            attn_hidden_dim,
                        )
                    )
                    nn.init.trunc_normal_(
                        self.k_video_pos_projection,
                        mean=0.0,
                        std=0.02,
                    )
                else:
                    self.register_buffer("k_video_pos_projection", None, persistent=False)
                self.register_buffer("per_head_kv_fusing_layer", None, persistent=False)
                self.register_buffer("k_head_channel_projection", None, persistent=False)
                self.register_buffer("k_head_channel_bias", None, persistent=False)
                self.register_buffer("kv_fusing_layer", None, persistent=False)
                self.register_buffer("k_channel_projection", None, persistent=False)
                self.register_buffer("v_channel_projection", None, persistent=False)
                self.register_buffer("k_channel_bias", None, persistent=False)
                self.register_buffer("v_channel_bias", None, persistent=False)
                self.register_buffer("head_fused_kv_k_channel_projection", None, persistent=False)
                self.register_buffer("head_fused_kv_v_channel_projection", None, persistent=False)
                self.register_buffer("head_fused_kv_layer_mixing", None, persistent=False)
            elif (
                kv_source_mode == "new_fused_kv"
                and self.new_fused_kv_projection_mode == "per_head_channel"
            ):
                # Middle-capacity ablation: each head learns a shared K/V layer
                # fusion over video layers, then K alone receives an in-head
                # channel projection. V keeps only the shared layer fusion.
                self.per_head_kv_fusing_layer = nn.Parameter(
                    torch.zeros(len(self.kv_source_mapping), action_num_layers, num_heads)
                )
                nn.init.trunc_normal_(
                    self.per_head_kv_fusing_layer,
                    mean=1.0 / len(self.kv_source_mapping),
                    std=0.02,
                )
                head_eye = torch.eye(attn_head_dim).expand(
                    action_num_layers,
                    num_heads,
                    attn_head_dim,
                    attn_head_dim,
                ).clone()
                self.k_head_channel_projection = nn.Parameter(head_eye)
                self.k_head_channel_bias = nn.Parameter(
                    torch.zeros(action_num_layers, num_heads, attn_head_dim)
                )
                self.register_buffer("simple_kv_fusing_layer", None, persistent=False)
                self.register_buffer("k_video_pos_projection", None, persistent=False)
                self.register_buffer("kv_fusing_layer", None, persistent=False)
                self.register_buffer("k_channel_projection", None, persistent=False)
                self.register_buffer("v_channel_projection", None, persistent=False)
                self.register_buffer("k_channel_bias", None, persistent=False)
                self.register_buffer("v_channel_bias", None, persistent=False)
                self.register_buffer("head_fused_kv_k_channel_projection", None, persistent=False)
                self.register_buffer("head_fused_kv_v_channel_projection", None, persistent=False)
                self.register_buffer("head_fused_kv_layer_mixing", None, persistent=False)
            elif (
                kv_source_mode == "new_fused_kv"
                and self.new_fused_kv_projection_mode
                in {
                    "HeadFusedKV",
                    "HeadFusedKV+Sin2DPE",
                    "simple_head_fused",
                    "simple_head_softmax",
                }
            ):
                # Use one shared K/V per-head layer mixer [H, L, M]. HeadFusedKV
                # first learns separate full-width K/V channel bases;
                # the simple variants remove those two channel matrices.
                if self.new_fused_kv_projection_mode in {
                    "simple_head_fused",
                    "simple_head_softmax",
                }:
                    self.register_buffer(
                        "head_fused_kv_k_channel_projection", None, persistent=False
                    )
                    self.register_buffer(
                        "head_fused_kv_v_channel_projection", None, persistent=False
                    )
                else:
                    self.head_fused_kv_k_channel_projection = nn.Parameter(
                        torch.empty(attn_hidden_dim, attn_hidden_dim)
                    )
                    self.head_fused_kv_v_channel_projection = nn.Parameter(
                        torch.empty(attn_hidden_dim, attn_hidden_dim)
                    )
                    nn.init.trunc_normal_(
                        self.head_fused_kv_k_channel_projection,
                        mean=0.0,
                        std=0.02,
                    )
                    nn.init.trunc_normal_(
                        self.head_fused_kv_v_channel_projection,
                        mean=0.0,
                        std=0.02,
                    )
                self.head_fused_kv_layer_mixing = nn.Parameter(
                    torch.zeros(
                        num_heads,
                        len(self.kv_source_mapping),
                        action_num_layers,
                    )
                )
                if self.new_fused_kv_projection_mode != "simple_head_softmax":
                    nn.init.trunc_normal_(
                        self.head_fused_kv_layer_mixing,
                        mean=0.0,
                        std=0.02,
                    )
                self.register_buffer("simple_kv_fusing_layer", None, persistent=False)
                self.register_buffer("k_video_pos_projection", None, persistent=False)
                self.register_buffer("per_head_kv_fusing_layer", None, persistent=False)
                self.register_buffer("k_head_channel_projection", None, persistent=False)
                self.register_buffer("k_head_channel_bias", None, persistent=False)
                self.register_buffer("kv_fusing_layer", None, persistent=False)
                self.register_buffer("k_channel_projection", None, persistent=False)
                self.register_buffer("v_channel_projection", None, persistent=False)
                self.register_buffer("k_channel_bias", None, persistent=False)
                self.register_buffer("v_channel_bias", None, persistent=False)
                self.mlp_mixer_fused_kv_blocks = None
                if self.new_fused_kv_projection_mode == "HeadFusedKV+Sin2DPE":
                    norm_k = video_expert.blocks[0].self_attn.norm_k
                    norm_weight = getattr(norm_k, "weight", None)
                    self.head_fused_kv_sin2d_pe_mlps = nn.ModuleList()
                    for _ in range(action_num_layers):
                        pos_mlp = nn.Sequential(
                            nn.Linear(
                                self.new_fused_kv_pos_embed_dim,
                                self.new_fused_kv_head_fused_kv_sin2d_pe_mlp_hidden_dim,
                            ),
                            nn.GELU(),
                            nn.Linear(
                                self.new_fused_kv_head_fused_kv_sin2d_pe_mlp_hidden_dim,
                                attn_hidden_dim,
                            ),
                        )
                        if norm_weight is not None:
                            pos_mlp = pos_mlp.to(
                                device=norm_weight.device,
                                dtype=norm_weight.dtype,
                            )
                        self.head_fused_kv_sin2d_pe_mlps.append(pos_mlp)
                else:
                    self.head_fused_kv_sin2d_pe_mlps = None
            elif (
                kv_source_mode == "new_fused_kv"
                and self.new_fused_kv_projection_mode == "MLPMixerFusedKV"
            ):
                # MLP-Mixer operates on the full flattened channel dimension
                # and treats the selected video layers as tokens. K/V share
                # token mixing, but retain separate channel mixers.
                norm_k = video_expert.blocks[0].self_attn.norm_k
                norm_weight = getattr(norm_k, "weight", None)
                self.mlp_mixer_fused_kv_blocks = nn.ModuleList([
                    MLPMixerFusedKVBlock(
                        num_layer_tokens=len(self.kv_source_mapping),
                        hidden_dim=attn_hidden_dim,
                        token_mlp_ratio=self.new_fused_kv_mlp_mixer_token_mlp_ratio,
                        channel_mlp_ratio=self.new_fused_kv_mlp_mixer_channel_mlp_ratio,
                        norm_cls=type(norm_k),
                        norm_eps=getattr(norm_k, "eps", 1e-6),
                        device=(None if norm_weight is None else norm_weight.device),
                        dtype=(None if norm_weight is None else norm_weight.dtype),
                    )
                    for _ in range(self.new_fused_kv_mlp_mixer_num_blocks)
                ])
                self.register_buffer("simple_kv_fusing_layer", None, persistent=False)
                self.register_buffer("k_video_pos_projection", None, persistent=False)
                self.register_buffer("per_head_kv_fusing_layer", None, persistent=False)
                self.register_buffer("k_head_channel_projection", None, persistent=False)
                self.register_buffer("k_head_channel_bias", None, persistent=False)
                self.register_buffer("kv_fusing_layer", None, persistent=False)
                self.register_buffer("k_channel_projection", None, persistent=False)
                self.register_buffer("v_channel_projection", None, persistent=False)
                self.register_buffer("k_channel_bias", None, persistent=False)
                self.register_buffer("v_channel_bias", None, persistent=False)
                self.register_buffer("head_fused_kv_k_channel_projection", None, persistent=False)
                self.register_buffer("head_fused_kv_v_channel_projection", None, persistent=False)
                self.register_buffer("head_fused_kv_layer_mixing", None, persistent=False)
            else:
                # Full path: each action layer, attention head, and per-head
                # channel learns one depthwise layer distribution. K and V share
                # this layer projection, then use separate flattened-channel
                # projections below.
                self.kv_fusing_layer = nn.Parameter(
                    torch.zeros(
                        len(self.kv_source_mapping),
                        action_num_layers,
                        num_heads,
                        attn_head_dim,
                    )
                )
                nn.init.trunc_normal_(
                    self.kv_fusing_layer,
                    mean=1.0 / len(self.kv_source_mapping),
                    std=0.02,
                )

                channel_eye = torch.eye(attn_hidden_dim).expand(
                    action_num_layers,
                    attn_hidden_dim,
                    attn_hidden_dim,
                ).clone()
                self.k_channel_projection = nn.Parameter(channel_eye)
                self.v_channel_projection = nn.Parameter(channel_eye.clone())
                self.k_channel_bias = nn.Parameter(
                    torch.zeros(action_num_layers, attn_hidden_dim)
                )
                self.v_channel_bias = nn.Parameter(
                    torch.zeros(action_num_layers, attn_hidden_dim)
                )
                self.register_buffer("simple_kv_fusing_layer", None, persistent=False)
                self.register_buffer("k_video_pos_projection", None, persistent=False)
                self.register_buffer("per_head_kv_fusing_layer", None, persistent=False)
                self.register_buffer("k_head_channel_projection", None, persistent=False)
                self.register_buffer("k_head_channel_bias", None, persistent=False)
                self.register_buffer("head_fused_kv_k_channel_projection", None, persistent=False)
                self.register_buffer("head_fused_kv_v_channel_projection", None, persistent=False)
                self.register_buffer("head_fused_kv_layer_mixing", None, persistent=False)
                self.mlp_mixer_fused_kv_blocks = None
            self.register_buffer(
                "kv_source_indices",
                torch.tensor(self.kv_source_mapping, dtype=torch.long),
                persistent=False,
            )
        else:
            for action_idx, video_idx in enumerate(self.kv_source_mapping):
                self._action_schedule.setdefault(video_idx, []).append(action_idx)
            self.register_buffer(
                "kv_source_indices",
                torch.tensor(self.kv_source_mapping, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer("kv_fusing_layer", None, persistent=False)
            self.register_buffer("simple_kv_fusing_layer", None, persistent=False)
            self.register_buffer("k_video_pos_projection", None, persistent=False)
            self.register_buffer("per_head_kv_fusing_layer", None, persistent=False)
            self.register_buffer("k_head_channel_projection", None, persistent=False)
            self.register_buffer("k_head_channel_bias", None, persistent=False)
            self.register_buffer("k_channel_projection", None, persistent=False)
            self.register_buffer("v_channel_projection", None, persistent=False)
            self.register_buffer("k_channel_bias", None, persistent=False)
            self.register_buffer("v_channel_bias", None, persistent=False)
            self.register_buffer("head_fused_kv_k_channel_projection", None, persistent=False)
            self.register_buffer("head_fused_kv_v_channel_projection", None, persistent=False)
            self.register_buffer("head_fused_kv_layer_mixing", None, persistent=False)
            self.mlp_mixer_fused_kv_blocks = None
            self.k_fused_norm = None


        # ---- KV Fusion module (optional, created by factory for fused_mlp mode) ----
        self.kv_fusion = kv_fusion
        if self.kv_source_mode == "fused_mlp" and self.kv_fusion is None:
            raise ValueError(
                "kv_source_mode='fused_mlp' requires a kv_fusion module to be "
                "provided, but kv_fusion is None. Ensure the factory passes the "
                "KVFusionModule via the kv_fusion kwarg."
            )
        if self.kv_fusion is not None and self.kv_source_mode != "fused_mlp":
            raise ValueError(
                f"kv_fusion module was provided but kv_source_mode="
                f"'{self.kv_source_mode}' (expected 'fused_mlp'). Either set "
                f"kv_source_mode='fused_mlp' or remove the kv_fusion argument."
            )
        if self.kv_fusion is not None:
            # The external fused-MLP path routes K/V through self.kv_fusion and
            # uses a dedicated forward. Clear the static routing schedule so the
            # forward dispatch has exactly one active routing mechanism.
            self._action_schedule = {}

        # ---- Logging ----
        if self.kv_fusion is not None:
            logger.info(
                f"Initialized MoTDecoupled with experts: {self.expert_order}, "
                f"video_num_layers={self.video_num_layers}, "
                f"action_num_layers={self.action_num_layers}, "
                f"kv_source=fused_mlp (all {self.video_num_layers} video layers)"
            )
        else:
            logger.info(
                f"Initialized MoTDecoupled with experts: {self.expert_order}, "
                f"video_num_layers={self.video_num_layers}, "
                f"action_num_layers={self.action_num_layers}, "
                f"kv_source_mode={self.kv_source_mode}, "
                f"kv_source_mapping={self.kv_source_mapping}"
            )
        action_cross_attn = getattr(action_expert, "action_text_cross_attn", True)
        for name in self.expert_order:
            expert = self.mixtures[name]
            num_params = sum(p.numel() for p in expert.parameters())
            num_trainable = sum(p.numel() for p in expert.parameters() if p.requires_grad)
            extra = ""
            if name == ACTION_EXPERT_KEY:
                extra = f", text_cross_attn={'on' if action_cross_attn else 'off'}"
            logger.info(
                f"  Expert '{name}': total={num_params / 1e9:.2f} B, "
                f"trainable={num_trainable / 1e9:.2f} B{extra}"
            )
        if self.kv_fusion is not None:
            num_fusion_params = sum(p.numel() for p in self.kv_fusion.parameters())
            num_fusion_trainable = sum(p.numel() for p in self.kv_fusion.parameters() if p.requires_grad)
            logger.info(f"  KV Fusion: total={num_fusion_params / 1e3:.1f} K, trainable={num_fusion_trainable / 1e3:.1f} K")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"  MoTDecoupled total: {total_params / 1e9:.2f} B, "
            f"trainable: {trainable_params / 1e9:.2f} B"
        )

    def _make_attn_rms_norm(self, expert: nn.Module) -> nn.Module:
        """Create an attention-width RMSNorm matching the expert implementation."""
        norm_k = expert.blocks[0].self_attn.norm_k
        norm_cls = type(norm_k)
        eps = getattr(norm_k, "eps", 1e-6)
        norm = norm_cls(self.num_heads * self.attn_head_dim, eps=eps)
        weight = getattr(norm_k, "weight", None)
        if weight is not None:
            norm = norm.to(device=weight.device, dtype=weight.dtype)
        return norm

    # ------------------------------------------------------------------
    # Override inherited lock-step methods with NotImplementedError
    # ------------------------------------------------------------------

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ) -> None:
        """Raises NotImplementedError -- use ``forward_decoupled`` for training
        or ``prefill_video_kv`` + ``forward_action_with_video_kv`` for
        inference."""
        raise NotImplementedError(
            "Use forward_decoupled() for training or "
            "prefill_video_kv + forward_action_with_video_kv for inference"
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> None:
        """Raises NotImplementedError -- use ``prefill_video_kv`` instead."""
        raise NotImplementedError("Use prefill_video_kv instead")

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> None:
        """Raises NotImplementedError -- use ``forward_action_with_video_kv``
        instead."""
        raise NotImplementedError("Use forward_action_with_video_kv instead")

    # ------------------------------------------------------------------
    # New decoupled methods
    # ------------------------------------------------------------------

    def _forward_decoupled_fused(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_masks: Dict[str, torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Two-phase forward with MLP-fused KV from all video layers.

        Phase 1: Run all video layers, collect K/V from every layer.
        Phase 2: Stack and fuse via self.kv_fusion.
        Phase 3: Run all action layers with fused KV.
        """
        video_mask = attention_masks[VIDEO_EXPERT_KEY]
        action_mask = attention_masks[ACTION_EXPERT_KEY]

        video_expert = self.mixtures[VIDEO_EXPERT_KEY]
        x_video = embeds_all[VIDEO_EXPERT_KEY]
        video_freqs = freqs_all[VIDEO_EXPERT_KEY]
        video_t_mod = t_mod_all[VIDEO_EXPERT_KEY]
        video_context = context_all.get(VIDEO_EXPERT_KEY)

        # Phase 1: Run all video layers, collect K/V at every layer.
        all_k_list: list[torch.Tensor] = []
        all_v_list: list[torch.Tensor] = []

        for layer_idx in range(self.video_num_layers):
            block = video_expert.blocks[layer_idx]
            (
                q, k, v,
                residual_x,
                gate_msa,
                shift_mlp, scale_mlp, gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=video_expert,
                block=block,
                x=x_video,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )

            mixed = self._mixed_attention(
                q_cat=q, k_cat=k, v_cat=v,
                attention_mask=video_mask,
            )

            x_video = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context,
            )

            all_k_list.append(k)
            all_v_list.append(v)

        # Phase 2: Stack and fuse.
        all_k = torch.stack(all_k_list, dim=2)  # [B, Sv, N, D]
        all_v = torch.stack(all_v_list, dim=2)  # [B, Sv, N, D]
        fused_kv_per_layer = self.kv_fusion(all_k, all_v)

        # Phase 3: Run all action layers with fused KV.
        action_expert = self.mixtures[ACTION_EXPERT_KEY]
        x_action = embeds_all[ACTION_EXPERT_KEY]
        action_freqs = freqs_all[ACTION_EXPERT_KEY]
        action_t_mod = t_mod_all[ACTION_EXPERT_KEY]
        action_context = context_all.get(ACTION_EXPERT_KEY)

        for action_layer_idx in range(self.action_num_layers):
            fused = fused_kv_per_layer[action_layer_idx]
            k_video_fused = fused["k"]
            v_video_fused = fused["v"]

            block = action_expert.blocks[action_layer_idx]
            (
                q_a, k_a, v_a,
                res_a,
                gate_a,
                sh_a, sc_a, g_a,
                gc_a,
            ) = self._build_expert_attention_io(
                expert=action_expert,
                block=block,
                x=x_action,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )

            k_cat = torch.cat([k_video_fused, k_a], dim=1)
            v_cat = torch.cat([v_video_fused, v_a], dim=1)

            mixed_a = self._mixed_attention(
                q_cat=q_a,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_mask,
            )

            x_action = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=res_a,
                gate_msa=gate_a,
                shift_mlp=sh_a,
                scale_mlp=sc_a,
                gate_mlp=g_a,
                use_gradient_checkpointing=gc_a,
                mixed_slice=mixed_a,
                context_payload=action_context,
            )

        return {
            VIDEO_EXPERT_KEY: x_video,
            ACTION_EXPERT_KEY: x_action,
        }

    def _forward_decoupled_two_phase(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_masks: Dict[str, torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Reference two-phase forward restricted to ``final_only`` routing.

        It runs all video layers first, captures final-layer K/V, then runs all
        action layers with that single K/V. The main selected-KV path now uses
        the same video-then-action ordering while supporting every routing mode.
        """
        video_mask = attention_masks[VIDEO_EXPERT_KEY]
        action_mask = attention_masks[ACTION_EXPERT_KEY]

        video_expert = self.mixtures[VIDEO_EXPERT_KEY]
        x_video = embeds_all[VIDEO_EXPERT_KEY]
        video_freqs = freqs_all[VIDEO_EXPERT_KEY]
        video_t_mod = t_mod_all[VIDEO_EXPERT_KEY]
        video_context = context_all.get(VIDEO_EXPERT_KEY)

        video_kv_cache: list[dict[str, torch.Tensor]] = []

        for layer_idx in range(self.video_num_layers):
            block = video_expert.blocks[layer_idx]
            (
                q, k, v,
                residual_x,
                gate_msa,
                shift_mlp, scale_mlp, gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=video_expert,
                block=block,
                x=x_video,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )

            mixed = self._mixed_attention(
                q_cat=q, k_cat=k, v_cat=v,
                attention_mask=video_mask,
            )

            x_video = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context,
            )

            video_kv_cache.append({"k": k, "v": v})

        action_expert = self.mixtures[ACTION_EXPERT_KEY]
        x_action = embeds_all[ACTION_EXPERT_KEY]
        action_freqs = freqs_all[ACTION_EXPERT_KEY]
        action_t_mod = t_mod_all[ACTION_EXPERT_KEY]
        action_context = context_all.get(ACTION_EXPERT_KEY)
        stacked_k, stacked_v = self._stack_video_kv(video_kv_cache)

        for layer_idx in range(self.action_num_layers):
            block = action_expert.blocks[layer_idx]
            (
                q_action, k_action, v_action,
                residual_x,
                gate_msa,
                shift_mlp, scale_mlp, gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=action_expert,
                block=block,
                x=x_action,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            k_video, v_video = self._select_or_mix_stacked_video_kv(
                stacked_k=stacked_k,
                stacked_v=stacked_v,
                action_layer_idx=layer_idx,
            )
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)

            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_mask,
            )

            x_action = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context,
            )

        return {
            VIDEO_EXPERT_KEY: x_video,
            ACTION_EXPERT_KEY: x_action,
        }

    def forward_decoupled(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_masks: Dict[str, torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run the decoupled training forward for selected-KV or fused-KV routing.

        Selected-KV and on-module fused-KV modes run all video layers, stack their
        K/V, then run all action layers. Static modes select the mapped stack entry;
        fused-KV modes learn a mix over the stack. External fused-MLP routing
        dispatches to ``_forward_decoupled_fused()`` with the same broad
        video-then-action ordering.

        Gradients flow from the action loss back through the video K/V into
        the video expert -- there is NO detach.

        Args:
            embeds_all: Dict with keys ``"video"`` and ``"action"`` mapping to
                token tensors of shape ``[B, Sv, D_video]`` and
                ``[B, Sa, D_action]`` respectively.
            attention_masks: Dict with keys ``"video"`` and ``"action"``:
                - ``"video"``: square boolean mask ``[Sv, Sv]`` for video
                  self-attention.
                - ``"action"``: rectangular boolean mask ``[Sa, Sv + Sa]`` for
                  action attention (action queries attend to concatenated
                  video K/V + action K/V).
            freqs_all: Dict with RoPE frequency tensors per expert.
            context_all: Dict with optional cross-attention context dicts per
                expert (keys ``"context"`` and ``"mask"``).
            t_mod_all: Dict with time-modulation tensors per expert.

        Returns:
            Dict with ``"video"`` and ``"action"`` keys mapping to updated
            token tensors of shape ``[B, Sv, D_video]`` and
            ``[B, Sa, D_action]``.
        """
        # ---- Input validation ----
        for key in (VIDEO_EXPERT_KEY, ACTION_EXPERT_KEY):
            if key not in embeds_all:
                raise ValueError(f"Missing '{key}' in embeds_all")
            if key not in freqs_all:
                raise ValueError(f"Missing '{key}' in freqs_all")
            if key not in t_mod_all:
                raise ValueError(f"Missing '{key}' in t_mod_all")
            if key not in attention_masks:
                raise ValueError(f"Missing '{key}' in attention_masks")

        video_mask = attention_masks[VIDEO_EXPERT_KEY]
        action_mask = attention_masks[ACTION_EXPERT_KEY]

        if video_mask.ndim != 2 or video_mask.shape[0] != video_mask.shape[1]:
            raise ValueError(
                f"Video attention mask must be square 2D, got shape "
                f"{tuple(video_mask.shape)}"
            )

        video_seq_len = embeds_all[VIDEO_EXPERT_KEY].shape[1]
        action_seq_len = embeds_all[ACTION_EXPERT_KEY].shape[1]

        if video_mask.shape[0] != video_seq_len:
            raise ValueError(
                f"Video mask seq length {video_mask.shape[0]} != "
                f"video token seq length {video_seq_len}"
            )

        # Action mask should be [Sa, Sv + Sa]
        expected_action_mask_shape = (action_seq_len, video_seq_len + action_seq_len)
        if action_mask.shape != torch.Size(expected_action_mask_shape):
            raise ValueError(
                f"Action attention mask must have shape {expected_action_mask_shape}, "
                f"got {tuple(action_mask.shape)}"
            )

        # ---- Dispatch: external fused-MLP path vs shared stacked-KV path ----
        if not self._action_schedule and self.kv_fusion is None:
            raise RuntimeError(
                "_action_schedule is empty and kv_fusion is None: "
                "no action layers would execute. This is a misconfiguration."
            )

        if self.kv_fusion is not None:
            return self._forward_decoupled_fused(
                embeds_all, attention_masks, freqs_all, context_all, t_mod_all
            )

        # ---- Shared stacked-KV forward ----
        # Complete the video pass before selecting or mixing cached K/V for the
        # subsequent action pass.
        video_expert = self.mixtures[VIDEO_EXPERT_KEY]
        action_expert = self.mixtures[ACTION_EXPERT_KEY]
        x_video = embeds_all[VIDEO_EXPERT_KEY]
        x_action = embeds_all[ACTION_EXPERT_KEY]
        video_freqs = freqs_all[VIDEO_EXPERT_KEY]
        video_t_mod = t_mod_all[VIDEO_EXPERT_KEY]
        action_freqs = freqs_all[ACTION_EXPERT_KEY]
        action_t_mod = t_mod_all[ACTION_EXPERT_KEY]
        video_context = context_all.get(VIDEO_EXPERT_KEY)
        action_context = context_all.get(ACTION_EXPERT_KEY)
        video_kv_cache: list[dict[str, torch.Tensor]] = []

        for video_layer_idx in range(self.video_num_layers):
            video_block = video_expert.blocks[video_layer_idx]
            (
                q, k, v,
                residual_x,
                gate_msa,
                shift_mlp, scale_mlp, gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=video_expert,
                block=video_block,
                x=x_video,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )

            mixed = self._mixed_attention(
                q_cat=q, k_cat=k, v_cat=v,
                attention_mask=video_mask,
            )

            x_video = self._apply_post_with_optional_checkpoint(
                block=video_block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context,
            )

            video_kv_cache.append({"k": k, "v": v})

        stacked_k, stacked_v = self._stack_video_kv(video_kv_cache)

        # ---- Action forward with selected or learned-mixed video K/V ----
        for action_layer_idx in range(self.action_num_layers):
            action_block = action_expert.blocks[action_layer_idx]
            (
                q_a, k_a, v_a,
                res_a,
                gate_a,
                sh_a, sc_a, g_a,
                gc_a,
            ) = self._build_expert_attention_io(
                expert=action_expert,
                block=action_block,
                x=x_action,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )

            k_video, v_video = self._select_or_mix_stacked_video_kv(
                stacked_k=stacked_k,
                stacked_v=stacked_v,
                action_layer_idx=action_layer_idx,
            )
            k_cat = torch.cat([k_video, k_a], dim=1)
            v_cat = torch.cat([v_video, v_a], dim=1)

            mixed_a = self._mixed_attention(
                q_cat=q_a,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_mask,
            )

            x_action = self._apply_post_with_optional_checkpoint(
                block=action_block,
                residual_x=res_a,
                gate_msa=gate_a,
                shift_mlp=sh_a,
                scale_mlp=sc_a,
                gate_mlp=g_a,
                use_gradient_checkpointing=gc_a,
                mixed_slice=mixed_a,
                context_payload=action_context,
            )

        return {
            VIDEO_EXPERT_KEY: x_video,
            ACTION_EXPERT_KEY: x_action,
        }

    def prefill_video_kv(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Run video expert through all layers and return per-action-layer K/V.

        All modes return a list of length ``action_num_layers`` (M):
        - selected-KV modes (final_only / uniform_end / uniform_middle):
          entry i is the raw K/V of video layer ``kv_source_mapping[i]``;
        - fused-MLP mode (``kv_fusion`` set): entry i is the learned fused K/V
          over all video layers for action layer i;
        - fused_kv mode: entry i is the mixed K/V produced by
          ``_select_or_mix_stacked_video_kv`` from the FULL video-layer stack
          (every layer is captured, mirroring the training path).

        Args:
            video_tokens: Video tokens before layer 0, shape ``[B, Sv, D]``.
            video_freqs: Video RoPE frequencies, shape ``[Sv, 1, rope_dim]``.
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional cross-attention context dict with
                keys ``"context"`` and ``"mask"``.
            video_attention_mask: Video self-attention mask, shape ``[Sv, Sv]``.

        Returns:
            List of dicts with ``"k"`` and ``"v"`` tensors of shape
            ``[B, Sv, H * Dh]``.
        """
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"video_attention_mask must be 2D [Sv, Sv], got shape "
                f"{tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"video_attention_mask must be square, got shape "
                f"{tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                f"video_attention_mask seq length {video_attention_mask.shape[0]} "
                f"!= video_tokens seq length {video_tokens.shape[1]}"
            )

        video_expert = self.mixtures[VIDEO_EXPERT_KEY]
        x = video_tokens

        if self.kv_fusion is not None:
            # Fusion mode: capture ALL layers, then fuse.
            all_k_list: list[torch.Tensor] = []
            all_v_list: list[torch.Tensor] = []

            for layer_idx in range(self.video_num_layers):
                block = video_expert.blocks[layer_idx]
                (
                    q, k, v,
                    residual_x,
                    gate_msa,
                    shift_mlp, scale_mlp, gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=video_expert,
                    block=block,
                    x=x,
                    freqs=video_freqs,
                    t_mod=video_t_mod,
                )

                mixed = self._mixed_attention(
                    q_cat=q, k_cat=k, v_cat=v,
                    attention_mask=video_attention_mask,
                )

                x = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=residual_x,
                    gate_msa=gate_msa,
                    shift_mlp=shift_mlp,
                    scale_mlp=scale_mlp,
                    gate_mlp=gate_mlp,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    mixed_slice=mixed,
                    context_payload=video_context_payload,
                )

                all_k_list.append(k)
                all_v_list.append(v)

            all_k = torch.stack(all_k_list, dim=2)  # [B, Sv, N, D]
            all_v = torch.stack(all_v_list, dim=2)
            return self.kv_fusion(all_k, all_v)

        # Non-fusion: run all video layers, capture KV from needed source layers.
        # fused_kv must capture EVERY layer: _select_or_mix_stacked_video_kv
        # index_selects with ORIGINAL layer indices (kv_source_indices) and
        # _stack_video_kv requires the full N-layer stack, exactly like the
        # training path (forward_decoupled). Capturing only the mapped layers
        # produced a compacted stack that raised (or mis-indexed) at inference
        # for any non-identity mapping while training worked. Selected modes
        # keep capturing only the mapped layers (they are consumed by original
        # index directly from captured_kv).
        if self.kv_source_mode in {"fused_kv", "new_fused_kv"}:
            needed = set(range(self.video_num_layers))
        else:
            needed = set(self.kv_source_mapping)
        captured_kv: dict[int, dict[str, torch.Tensor]] = {}

        for layer_idx in range(self.video_num_layers):
            block = video_expert.blocks[layer_idx]
            (
                q, k, v,
                residual_x,
                gate_msa,
                shift_mlp, scale_mlp, gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=video_expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )

            mixed = self._mixed_attention(
                q_cat=q, k_cat=k, v_cat=v,
                attention_mask=video_attention_mask,
            )

            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )

            if layer_idx in needed:
                captured_kv[layer_idx] = {"k": k, "v": v}

        if self.kv_source_mode in {"fused_kv", "new_fused_kv"}:
            all_kv = [captured_kv[i] for i in sorted(captured_kv)]
            stacked_k, stacked_v = self._stack_video_kv(all_kv)
            result = []
            for action_idx in range(self.action_num_layers):
                k_mixed, v_mixed = self._select_or_mix_stacked_video_kv(
                    stacked_k, stacked_v, action_idx,
                )
                result.append({"k": k_mixed, "v": v_mixed})
            return result

        return [captured_kv[src] for src in self.kv_source_mapping]

    def prefill_video_final_kv(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Backward-compat wrapper. Prefer ``prefill_video_kv()`` instead."""
        result_list = self.prefill_video_kv(
            video_tokens, video_freqs, video_t_mod,
            video_context_payload, video_attention_mask,
        )
        return result_list[-1]

    def _stack_video_kv(
        self,
        video_kv_cache: list[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate and stack flattened video KV as [layer, B, S, H, Dh]."""
        if len(video_kv_cache) != self.video_num_layers:
            raise ValueError(
                f"video_kv_cache length {len(video_kv_cache)} != "
                f"video_num_layers {self.video_num_layers}"
            )

        expected_hidden = self.num_heads * self.attn_head_dim
        k_per_layer: list[torch.Tensor] = []
        v_per_layer: list[torch.Tensor] = []
        expected_shape: torch.Size | None = None
        for layer_idx, layer_kv in enumerate(video_kv_cache):
            if "k" not in layer_kv or "v" not in layer_kv:
                raise ValueError(
                    f"video_kv_cache[{layer_idx}] must contain 'k' and 'v'."
                )
            k = layer_kv["k"]
            v = layer_kv["v"]
            if k.shape != v.shape:
                raise ValueError(
                    f"video_kv_cache[{layer_idx}] K/V shape mismatch: "
                    f"{tuple(k.shape)} vs {tuple(v.shape)}"
                )
            if k.ndim != 3 or k.shape[-1] != expected_hidden:
                raise ValueError(
                    f"video_kv_cache[{layer_idx}] tensors must be [B, Sv, "
                    f"{expected_hidden}], got {tuple(k.shape)}"
                )
            if expected_shape is None:
                expected_shape = k.shape
            elif k.shape != expected_shape:
                raise ValueError(
                    f"video_kv_cache[{layer_idx}] shape {tuple(k.shape)} != "
                    f"first layer shape {tuple(expected_shape)}"
                )
            k_per_layer.append(k.unflatten(-1, (self.num_heads, self.attn_head_dim)))
            v_per_layer.append(v.unflatten(-1, (self.num_heads, self.attn_head_dim)))

        return torch.stack(k_per_layer), torch.stack(v_per_layer)

    def _select_or_mix_stacked_video_kv(
        self,
        stacked_k: torch.Tensor,
        stacked_v: torch.Tensor,
        action_layer_idx: int,
        softmax_fusing: bool=False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select one mapped source or mix mapped sources for one action layer.

        Fused layer weights are per source layer, action layer, attention
        head, and per-head channel, shared by K/V. After layer fusion,
        each action layer applies separate flattened-channel projections
        for K and V.
        """
        if self.kv_source_mode not in {"fused_kv", "new_fused_kv"}:
            source_idx = self.kv_source_mapping[action_layer_idx]
            return stacked_k[source_idx].flatten(-2), stacked_v[source_idx].flatten(-2)

        source_indices = self.kv_source_indices.to(device=stacked_k.device)
        selected_k = stacked_k.index_select(0, source_indices)
        selected_v = stacked_v.index_select(0, source_indices)

        if (
            self.kv_source_mode == "new_fused_kv"
            and self.new_fused_kv_projection_mode in {
                "simple",
                "simple+PE",
                "simple+PE-postnorm",
            }
        ):
            assert self.simple_kv_fusing_layer is not None
            if softmax_fusing:
                weights = self.simple_kv_fusing_layer[:, action_layer_idx].softmax(dim=0)
            else:
                weights = self.simple_kv_fusing_layer[:, action_layer_idx]
            weights = weights.to(device=selected_k.device, dtype=selected_k.dtype)
            mixed_k = torch.einsum("lbshd,l->bshd", selected_k, weights).flatten(-2)
            mixed_v = torch.einsum("lbshd,l->bshd", selected_v, weights).flatten(-2)
            if (
                self.new_fused_kv_projection_mode != "simple+PE"
                and self.k_fused_norm is not None
            ):
                mixed_k = self.k_fused_norm[action_layer_idx](mixed_k)
            return mixed_k, mixed_v

        if (
            self.kv_source_mode == "new_fused_kv"
            and self.new_fused_kv_projection_mode
            in {
                "HeadFusedKV",
                "HeadFusedKV+Sin2DPE",
                "simple_head_fused",
                "simple_head_softmax",
            }
        ):
            if self.new_fused_kv_projection_mode in {
                "simple_head_fused",
                "simple_head_softmax",
            }:
                mixed_k_basis = selected_k
                mixed_v_basis = selected_v
            else:
                assert self.head_fused_kv_k_channel_projection is not None
                assert self.head_fused_kv_v_channel_projection is not None
                k_projection = self.head_fused_kv_k_channel_projection.to(
                    device=selected_k.device,
                    dtype=selected_k.dtype,
                )
                v_projection = self.head_fused_kv_v_channel_projection.to(
                    device=selected_v.device,
                    dtype=selected_v.dtype,
                )
                mixed_k_basis = torch.matmul(
                    selected_k.flatten(-2), k_projection
                ).unflatten(-1, (self.num_heads, self.attn_head_dim))
                mixed_v_basis = torch.matmul(
                    selected_v.flatten(-2), v_projection
                ).unflatten(-1, (self.num_heads, self.attn_head_dim))
            assert self.head_fused_kv_layer_mixing is not None
            layer_mix_action_idx = action_layer_idx
            if self.new_fused_kv_projection_mode == "simple_head_softmax":
                if self.new_fused_kv_simple_head_softmax_fuse_mode == "shared":
                    layer_mix_action_idx = 0
                weights = self.head_fused_kv_layer_mixing[:, :, layer_mix_action_idx]
                if self.new_fused_kv_simple_head_softmax_fuse_mode == "uniform_end":
                    num_sources = selected_k.shape[0]
                    start = round(action_layer_idx * num_sources / self.action_num_layers)
                    end = round((action_layer_idx + 1) * num_sources / self.action_num_layers)
                    if end <= start:
                        raise RuntimeError(
                            "simple_head_softmax uniform_end produced an empty "
                            f"video-layer segment for action layer {action_layer_idx} "
                            f"with {num_sources} sources and {self.action_num_layers} "
                            "action layers."
                        )
                    mixed_k_basis = mixed_k_basis[start:end]
                    mixed_v_basis = mixed_v_basis[start:end]
                    weights = weights[:, start:end]
            else:
                if (
                    self.new_fused_kv_projection_mode
                    in {"HeadFusedKV", "HeadFusedKV+Sin2DPE"}
                    and self.new_fused_kv_head_fused_kv_fuse_mode == "shared"
                ):
                    layer_mix_action_idx = 0
                weights = self.head_fused_kv_layer_mixing[:, :, layer_mix_action_idx]
            if (
                self.new_fused_kv_projection_mode == "simple_head_softmax"
                or softmax_fusing
            ):
                weights = weights.float().softmax(dim=1)
            weights = weights.to(device=selected_k.device, dtype=selected_k.dtype)
            mixed_k = torch.einsum("lbshd,hl->bshd", mixed_k_basis, weights).flatten(-2)
            mixed_v = torch.einsum("lbshd,hl->bshd", mixed_v_basis, weights).flatten(-2)
            if (
                self.k_fused_norm is not None
                and self.new_fused_kv_projection_mode != "HeadFusedKV+Sin2DPE"
            ):
                mixed_k = self.k_fused_norm[action_layer_idx](mixed_k)
            return mixed_k, mixed_v

        if (
            self.kv_source_mode == "new_fused_kv"
            and self.new_fused_kv_projection_mode == "MLPMixerFusedKV"
        ):
            assert self.mlp_mixer_fused_kv_blocks is not None
            mixed_k = selected_k.flatten(-2)
            mixed_v = selected_v.flatten(-2)
            for mixer_block in self.mlp_mixer_fused_kv_blocks:
                mixed_k, mixed_v = mixer_block(mixed_k, mixed_v)
            # Pool the layer/token axis after all residual Mixer blocks.
            mixed_k = mixed_k.mean(dim=0)
            mixed_v = mixed_v.mean(dim=0)
            if self.k_fused_norm is not None:
                mixed_k = self.k_fused_norm[action_layer_idx](mixed_k)
            return mixed_k, mixed_v

        if (
            self.kv_source_mode == "new_fused_kv"
            and self.new_fused_kv_projection_mode == "per_head_channel"
        ):
            assert self.per_head_kv_fusing_layer is not None
            assert self.k_head_channel_projection is not None
            assert self.k_head_channel_bias is not None
            if softmax_fusing:
                weights = self.per_head_kv_fusing_layer[:, action_layer_idx, :].softmax(dim=0)
            else:
                weights = self.per_head_kv_fusing_layer[:, action_layer_idx, :]
            weights = weights.to(device=selected_k.device, dtype=selected_k.dtype)
            mixed_k = torch.einsum("lbshd,lh->bshd", selected_k, weights)
            mixed_v = torch.einsum("lbshd,lh->bshd", selected_v, weights)
            k_head_projection = self.k_head_channel_projection[action_layer_idx].to(
                device=mixed_k.device,
                dtype=mixed_k.dtype,
            )
            k_head_bias = self.k_head_channel_bias[action_layer_idx].to(
                device=mixed_k.device,
                dtype=mixed_k.dtype,
            )
            mixed_k = (
                torch.einsum("bshd,hde->bshe", mixed_k, k_head_projection)
                + k_head_bias.unsqueeze(0).unsqueeze(0)
            )
            mixed_k = mixed_k.flatten(-2)
            mixed_v = mixed_v.flatten(-2)
            if self.k_fused_norm is not None:
                mixed_k = self.k_fused_norm[action_layer_idx](mixed_k)
            return mixed_k, mixed_v

        assert self.kv_fusing_layer is not None
        assert self.k_channel_projection is not None
        assert self.v_channel_projection is not None
        assert self.k_channel_bias is not None
        assert self.v_channel_bias is not None
        if softmax_fusing:
            weights = self.kv_fusing_layer[:, action_layer_idx, :, :].softmax(dim=0)
        else:
            weights = self.kv_fusing_layer[:, action_layer_idx, :, :]

        k_weights = weights.to(device=selected_k.device, dtype=selected_k.dtype)
        v_weights = weights.to(device=selected_v.device, dtype=selected_v.dtype)

        mixed_k = torch.einsum("lbshd,lhd->bshd", selected_k, k_weights)
        mixed_v = torch.einsum("lbshd,lhd->bshd", selected_v, v_weights)
        k_channel_projection = self.k_channel_projection[action_layer_idx].to(
            device=mixed_k.device,
            dtype=mixed_k.dtype,
        )
        v_channel_projection = self.v_channel_projection[action_layer_idx].to(
            device=mixed_v.device,
            dtype=mixed_v.dtype,
        )
        k_channel_bias = self.k_channel_bias[action_layer_idx].to(
            device=mixed_k.device,
            dtype=mixed_k.dtype,
        )
        v_channel_bias = self.v_channel_bias[action_layer_idx].to(
            device=mixed_v.device,
            dtype=mixed_v.dtype,
        )
        mixed_k = mixed_k.flatten(-2)
        mixed_v = mixed_v.flatten(-2)
        mixed_k = (
            torch.matmul(mixed_k, k_channel_projection)
            + k_channel_bias.unsqueeze(0).unsqueeze(0)
        )
        mixed_v = (
            torch.matmul(mixed_v, v_channel_projection)
            + v_channel_bias.unsqueeze(0).unsqueeze(0)
        )
        if self.kv_source_mode == "new_fused_kv" and self.k_fused_norm is not None:
            mixed_k = self.k_fused_norm[action_layer_idx](mixed_k)
        return mixed_k, mixed_v

    def forward_action_with_video_kv(
        self,
        video_kv_per_layer: list[dict[str, torch.Tensor]],
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run action expert using per-action-layer K/V from prefill_video_kv.

        Args:
            video_kv_per_layer: List of length ``action_num_layers``, each a
                dict with ``"k"`` and ``"v"`` tensors of shape
                ``[B, Sv, H*Dh]``. Source selection or fusion is already done
                by ``prefill_video_kv()``.
            action_tokens: Action tokens before layer 0, shape ``[B, Sa, D]``.
            action_freqs: Action RoPE frequencies.
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional cross-attention context dict.
            action_attention_mask: Rectangular mask ``[Sa, Sv + Sa]``.

        Returns:
            Updated action tokens, shape ``[B, Sa, D_action]``.
        """
        if len(video_kv_per_layer) != self.action_num_layers:
            raise ValueError(
                f"video_kv_per_layer length {len(video_kv_per_layer)} != "
                f"action_num_layers {self.action_num_layers}"
            )
        action_expert = self.mixtures[ACTION_EXPERT_KEY]
        x = action_tokens

        for layer_idx in range(self.action_num_layers):
            kv = video_kv_per_layer[layer_idx]
            k_video = kv["k"]
            v_video = kv["v"]

            if layer_idx == 0:
                video_seq_len = k_video.shape[1]
                action_seq_len = action_tokens.shape[1]
                expected_mask_shape = (action_seq_len, video_seq_len + action_seq_len)
                if action_attention_mask.ndim != 2:
                    raise ValueError(
                        f"action_attention_mask must be 2D, got ndim={action_attention_mask.ndim}"
                    )
                if action_attention_mask.shape != torch.Size(expected_mask_shape):
                    raise ValueError(
                        f"action_attention_mask shape must be {expected_mask_shape}, "
                        f"got {tuple(action_attention_mask.shape)}"
                    )

            block = action_expert.blocks[layer_idx]
            (
                q_action, k_action, v_action,
                residual_x,
                gate_msa,
                shift_mlp, scale_mlp, gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=action_expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )

            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)

            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )

            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )

        return x

    def forward_action_with_final_kv(
        self,
        final_video_kv,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Backward-compat wrapper. Prefer ``forward_action_with_video_kv()`` instead."""
        if isinstance(final_video_kv, dict):
            if "k" not in final_video_kv or "v" not in final_video_kv:
                raise ValueError(
                    "final_video_kv must contain 'k' and 'v' keys."
                )
            video_kv_per_layer = [final_video_kv] * self.action_num_layers
        else:
            video_kv_per_layer = final_video_kv
        return self.forward_action_with_video_kv(
            video_kv_per_layer=video_kv_per_layer,
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=action_t_mod,
            action_context_payload=action_context_payload,
            action_attention_mask=action_attention_mask,
        )
