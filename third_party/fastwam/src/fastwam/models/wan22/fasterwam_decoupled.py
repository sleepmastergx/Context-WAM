# Author: Rui Heng Yang

"""FasterWAMDecoupled -- decoupled variant with action-aligned video RoPE.

This module defines a parallel variant where video self-attention still uses
the video's native 3D RoPE, but the video K cached for action attention is kept
without video-self-attention RoPE (except ``legacy_3d``, which caches post-3D-RoPE
K). Before action attention concatenates video K/V with action K/V, the
selected/fused video K is re-positioned in the action attention coordinate
system. ``new_fused_kv`` can use aligned 3D RoPE (``aligned_3d``), the same plus
per-head camera-region masking (``aligned_3dp``), temporal-only aligned 1D RoPE,
original video 3D RoPE with action 1D RoPE, or ``legacy_3d`` (fuse post-3D-RoPE
K, then no extra video RoPE like ``video_zero_1d``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .fastwam_decoupled import FastWAMDecoupled
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot_decoupled import ACTION_EXPERT_KEY, VIDEO_EXPERT_KEY, MoTDecoupled
from .wan_video_dit import modulate, precompute_freqs_cis, rope_apply

logger = get_logger(__name__)


def _validate_new_fused_kv_projection_mode(
    kv_source_mode: str,
    projection_mode: str,
) -> str:
    """Validate and normalize the projection-mode configuration."""
    projection_mode = str(projection_mode)
    if projection_mode not in MoTDecoupled.supported_new_fused_kv_projection_modes:
        raise ValueError(
            f"Unknown new_fused_kv_projection_mode {projection_mode!r}; "
            "expected one of "
            f"{sorted(MoTDecoupled.supported_new_fused_kv_projection_modes)}"
        )
    if kv_source_mode != "new_fused_kv" and projection_mode != "full":
        raise ValueError(
            "new_fused_kv_projection_mode is only configurable when "
            "kv_source_mode=new_fused_kv; use full for other modes."
        )
    return projection_mode


class MoTDecoupledActionAlignedVideoRoPE(MoTDecoupled):
    """Decoupled MoT with raw video K cache and action-aligned video K RoPE.

    Differences from ``MoTDecoupled``:
    - video self-attention uses native video 3D RoPE exactly as before;
    - cached video K for action attention is stored before 3D RoPE by default
      (``legacy_3d`` instead caches post-3D-RoPE K);
    - before concatenating ``[video K, action K]``, video K is re-positioned
      according to the selected KV/RoPE mode.
    """

    enable_new_fused_kv_key_norm = True
    # EEF-relative camera RoPE siblings. They share every code path except
    # action-mask construction; see `_build_ee_rope_freqs`.
    ee_rope_modes = frozenset({"ee_rope", "exclusive_ee_rope"})
    supported_new_fused_kv_rope_modes = {
        "aligned_3d",
        "aligned_3dp",
        "aligned_3d_overlap",
        "aligned_1d",
        "video_zero_1d",
        "original_3d",
        "legacy_3d",
        "ee_rope",
        "exclusive_ee_rope",
    }
    aligned_3d_action_spatial_anchor_presets = {
        "center": ((0.5, 0.5),),
        "horizontal": ((0.5, 0.25), (0.5, 0.75)),
        "libero": ((0.5, 0.25), (0.5, 0.75)),
        # LIBERO with the wrist anchor moved off the image centre and onto the
        # place the gripper actually appears. The wrist camera is rigidly mounted
        # to the end effector, so the EEF projects to the SAME camera-local token
        # in every frame of every task -- (5.358016, 2.993164) on the 7x7
        # per-camera grid, measured across all 10030 LIBERO-Plus entries with
        # zero variance. `horizontal` puts the wrist head at composite
        # (3.0, 10.0); the true gripper is at composite (5.358016, 9.993164), so
        # the column was already right and only the row was off, by 2.36 tokens.
        # Normalized for `anchor * size - 0.5` on the 7x14 LIBERO composite:
        # row (5.358016 + 0.5) / 7, column (9.993164 + 0.5) / 14.
        # Reproduce by projecting the EEF origin through `wrist_mount_T_eef_from_C`
        # in configs/calibration/libero_plus_anchors_v1.json: the point lands at
        # the translation of inv(mount), independent of the robot's pose.
        "libero_wrist_grounded": (
            (0.5, 0.25),
            (0.836859390551857, 0.74951171875),
        ),
        "robotwin": (
            (1.0 / 3.0, 0.5),
            (5.0 / 6.0, 0.25),
            (5.0 / 6.0, 0.75),
        ),
    }
    _aligned_3d_family_rope_modes = frozenset({"aligned_3d", "aligned_3dp"})

    # Geometry identity is a CONJUNCTION, not the calibration digest alone
    # (plan Section 20.1). A dataset re-exported at a different raw resolution
    # changes every anchor while leaving the digest untouched.
    EEF_GEOMETRY_IDENTITY_FIELDS = (
        "calibration_digest",
        "raw_source_resolution",
        "token_grid_h",
        "token_grid_w",
        "camera_order",
        "projection_version",
    )

    def __init__(
        self,
        *args,
        new_fused_kv_rope_mode: str = "aligned_3d",
        aligned_3d_action_spatial_anchor_layout: str | None = "center",
        eef_geometry_identity: dict[str, Any] | None = None,
        **kwargs,
    ):
        if new_fused_kv_rope_mode not in self.supported_new_fused_kv_rope_modes:
            raise ValueError(
                "Unknown new_fused_kv_rope_mode "
                f"{new_fused_kv_rope_mode!r}; expected one of "
                f"{sorted(self.supported_new_fused_kv_rope_modes)}"
            )
        self.new_fused_kv_rope_mode = new_fused_kv_rope_mode
        anchor_layout = (
            "center"
            if aligned_3d_action_spatial_anchor_layout is None
            else str(aligned_3d_action_spatial_anchor_layout)
        )
        if anchor_layout not in self.aligned_3d_action_spatial_anchor_presets:
            raise ValueError(
                "Unknown aligned_3d_action_spatial_anchor_layout "
                f"{anchor_layout!r}; expected one of "
                f"{sorted(self.aligned_3d_action_spatial_anchor_presets)}"
            )
        self.aligned_3d_action_spatial_anchor_layout = anchor_layout
        self.aligned_3d_action_spatial_anchors = (
            self.aligned_3d_action_spatial_anchor_presets[anchor_layout]
        )
        # Fail HERE, not at the first action-path forward. The count check needs
        # no video grid, so a layout this mode cannot express is rejected while
        # the config is still in front of a human -- a queued job can otherwise
        # sit for days before the builder-level guard fires. Only
        # `aligned_3d_overlap` folds the charts, so no other mode is restricted.
        if new_fused_kv_rope_mode == "aligned_3d_overlap":
            self._validate_aligned_3d_overlap_anchor_count()
        if eef_geometry_identity is not None:
            missing = [
                field
                for field in self.EEF_GEOMETRY_IDENTITY_FIELDS
                if field not in eef_geometry_identity
            ]
            if missing:
                raise ValueError(
                    "eef_geometry_identity is incomplete; geometry identity is a "
                    f"conjunction and every field is load-bearing. Missing: {missing}"
                )
            eef_geometry_identity = {
                field: eef_geometry_identity[field]
                for field in self.EEF_GEOMETRY_IDENTITY_FIELDS
            }
        elif new_fused_kv_rope_mode in self.ee_rope_modes:
            raise ValueError(
                f"new_fused_kv_rope_mode={new_fused_kv_rope_mode!r} requires "
                "eef_geometry_identity. Without it a checkpoint cannot record "
                "which geometry it was trained against, and a dataset "
                "re-exported at another raw resolution would silently produce "
                "different anchors under the same calibration digest "
                "(plan Section 20.1)."
            )
        self.eef_geometry_identity = eef_geometry_identity
        super().__init__(*args, **kwargs)

    @property
    def _cache_post_3d_rope_video_k(self) -> bool:
        """Whether action-path video K should be cached after video 3D RoPE."""
        return (
            self.kv_source_mode == "new_fused_kv"
            and self.new_fused_kv_rope_mode == "legacy_3d"
        )

    def _build_expert_attention_io_with_raw_k(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build attention tensors and expose both raw and RoPE-applied K."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self._split_modulation(block, t_mod)
        )
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k_raw = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k_rope = rope_apply(k_raw, freqs, block.num_heads)

        use_gradient_checkpointing = bool(
            getattr(expert, "use_gradient_checkpointing", False)
        )
        return (
            q,
            k_rope,
            k_raw,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _build_expert_attention_io_with_pre_norm_k(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build video attention while exposing K before RMSNorm and RoPE."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self._split_modulation(block, t_mod)
        )
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k_pre_norm = block.self_attn.k(attn_input)
        k_rope = rope_apply(block.self_attn.norm_k(k_pre_norm), freqs, block.num_heads)
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)

        use_gradient_checkpointing = bool(
            getattr(expert, "use_gradient_checkpointing", False)
        )
        return (
            q,
            k_rope,
            k_pre_norm,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_action_zero_rope_to_video_k(
        self,
        k_video_raw: torch.Tensor,
        action_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate every video key token with action token 0's 1D RoPE."""
        if action_freqs.ndim != 3 or action_freqs.shape[0] < 1:
            raise ValueError(
                f"action_freqs must be [Sa, 1, rope_dim] with Sa >= 1, "
                f"got {tuple(action_freqs.shape)}"
            )
        action_zero_freq = action_freqs[:1].to(device=k_video_raw.device)
        return rope_apply(k_video_raw, action_zero_freq, self.num_heads)

    def _split_3d_rope_dims(self) -> tuple[int, int, int]:
        """Return temporal/height/width RoPE complex dims for one attention head."""
        temporal_real_dim = self.attn_head_dim - 2 * (self.attn_head_dim // 3)
        spatial_real_dim = self.attn_head_dim // 3
        return (
            temporal_real_dim // 2,
            spatial_real_dim // 2,
            spatial_real_dim // 2,
        )

    def _infer_video_tokens_per_frame(self, video_freqs: torch.Tensor) -> int:
        """Infer the contiguous video tokens-per-frame from repeated temporal freqs."""
        t_dim, _, _ = self._split_3d_rope_dims()
        if video_freqs.ndim != 3 or video_freqs.shape[0] < 1:
            raise ValueError(
                f"video_freqs must be [Sv, 1, rope_dim], got {tuple(video_freqs.shape)}"
            )
        temporal = video_freqs[:, 0, :t_dim]
        first_temporal = temporal[:1]
        same_as_first = torch.isclose(
            temporal,
            first_temporal.expand_as(temporal),
        ).all(dim=-1)
        first_different = (~same_as_first).nonzero(as_tuple=False)
        if first_different.numel() == 0:
            return int(video_freqs.shape[0])
        return int(first_different[0, 0].item())

    def _infer_video_spatial_grid_size(
        self,
        video_freqs: torch.Tensor,
        tokens_per_frame: int,
    ) -> tuple[int, int]:
        """Infer one-frame token grid height/width from the 3D RoPE layout."""
        if tokens_per_frame <= 1:
            return 1, max(tokens_per_frame, 1)
        t_dim, h_dim, w_dim = self._split_3d_rope_dims()
        first_frame = video_freqs[:tokens_per_frame, 0]
        w_start = t_dim + h_dim
        w_part = first_frame[:, w_start:w_start + w_dim]
        width = None
        for idx in range(1, tokens_per_frame + 1):
            if tokens_per_frame % idx != 0:
                continue
            if torch.allclose(w_part[idx - 1], w_part[-1]):
                width = idx
                break
        if width is None or width <= 0:
            width = tokens_per_frame
        return max(tokens_per_frame // width, 1), width

    def _infer_video_spatial_center_index(
        self,
        video_freqs: torch.Tensor,
        tokens_per_frame: int,
    ) -> int:
        """Infer the center token index inside one video frame from h/w RoPE cycles."""
        if tokens_per_frame <= 1:
            return 0
        t_dim, h_dim, w_dim = self._split_3d_rope_dims()
        first_frame = video_freqs[:tokens_per_frame, 0]
        w_start = t_dim + h_dim
        w_part = first_frame[:, w_start:w_start + w_dim]
        width = None
        for idx in range(1, tokens_per_frame + 1):
            if tokens_per_frame % idx != 0:
                continue
            if torch.allclose(w_part[idx - 1], w_part[-1]):
                width = idx
                break
        if width is None or width <= 0:
            return tokens_per_frame // 2
        height = max(tokens_per_frame // width, 1)
        return min((height // 2) * width + (width // 2), tokens_per_frame - 1)

    def _build_2d_sincos_spatial_pe(
        self,
        height: int,
        width: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build fixed DETR-style 2D sine/cosine PE for one video frame."""
        if dim % 4 != 0:
            raise ValueError(f"2D sin/cos PE dim must be divisible by 4, got {dim}")
        quarter = dim // 4
        omega = torch.arange(quarter, device=device, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / max(quarter, 1)))
        y = torch.arange(height, device=device, dtype=torch.float32)
        x = torch.arange(width, device=device, dtype=torch.float32)
        y_embed = y[:, None] * omega[None, :]
        x_embed = x[:, None] * omega[None, :]
        y_embed = torch.cat([y_embed.sin(), y_embed.cos()], dim=-1)
        x_embed = torch.cat([x_embed.sin(), x_embed.cos()], dim=-1)
        pe = torch.cat([
            y_embed[:, None, :].expand(height, width, -1),
            x_embed[None, :, :].expand(height, width, -1),
        ], dim=-1)
        return pe.reshape(height * width, dim).to(dtype=dtype)

    def _build_simple_pe_for_video_k(
        self,
        k_video: torch.Tensor,
        action_layer_idx: int,
        video_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Build spatial-only fixed 2D sin/cos PE through the learned projection."""
        projection = getattr(self, "k_video_pos_projection", None)
        if projection is None:
            raise ValueError(
                "simple+PE modes require k_video_pos_projection to be initialized."
            )
        tokens_per_frame = self._infer_video_tokens_per_frame(video_freqs)
        height, width = self._infer_video_spatial_grid_size(
            video_freqs=video_freqs,
            tokens_per_frame=tokens_per_frame,
        )
        if height * width != tokens_per_frame:
            raise ValueError(
                f"Invalid inferred video spatial grid {(height, width)} for "
                f"tokens_per_frame={tokens_per_frame}."
            )
        pe_small = self._build_2d_sincos_spatial_pe(
            height=height,
            width=width,
            dim=projection.shape[1],
            device=k_video.device,
            dtype=k_video.dtype,
        )
        pe_projected = torch.matmul(
            pe_small,
            projection[action_layer_idx].to(device=k_video.device, dtype=k_video.dtype),
        )
        video_tokens = k_video.shape[1]
        spatial_idx = torch.arange(video_tokens, device=k_video.device) % tokens_per_frame
        return pe_projected.index_select(0, spatial_idx).unsqueeze(0)

    def _build_head_fused_kv_sin2d_pe(
        self,
        k_video: torch.Tensor,
        action_layer_idx: int,
        video_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Map fixed 2D sin/cos positions through the HeadFusedKV PE MLP."""
        pos_mlps = getattr(self, "head_fused_kv_sin2d_pe_mlps", None)
        if pos_mlps is None:
            raise ValueError("HeadFusedKV+Sin2DPE requires initialized PE MLPs.")
        tokens_per_frame = self._infer_video_tokens_per_frame(video_freqs)
        height, width = self._infer_video_spatial_grid_size(
            video_freqs=video_freqs,
            tokens_per_frame=tokens_per_frame,
        )
        if height * width != tokens_per_frame:
            raise ValueError(
                f"Invalid inferred video spatial grid {(height, width)} for "
                f"tokens_per_frame={tokens_per_frame}."
            )
        pe_small = self._build_2d_sincos_spatial_pe(
            height=height,
            width=width,
            dim=self.new_fused_kv_pos_embed_dim,
            device=k_video.device,
            dtype=k_video.dtype,
        )
        video_tokens = k_video.shape[1]
        spatial_idx = torch.arange(video_tokens, device=k_video.device) % tokens_per_frame
        return pos_mlps[action_layer_idx](pe_small.index_select(0, spatial_idx)).unsqueeze(0)

    def _apply_simple_pe_to_video_k(
        self,
        k_video: torch.Tensor,
        action_layer_idx: int,
        video_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Add spatial-only fixed 2D sin/cos PE through the learned projection."""
        return k_video + self._build_simple_pe_for_video_k(
            k_video=k_video,
            action_layer_idx=action_layer_idx,
            video_freqs=video_freqs,
        )

    def _allocate_aligned_3d_head_anchor_indices(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        """Allocate action-attention heads to camera-view anchors."""
        num_anchors = len(self.aligned_3d_action_spatial_anchors)
        if num_anchors == 1:
            return torch.zeros(self.num_heads, device=device, dtype=torch.long)
        if self.num_heads % 2 != 0:
            raise ValueError(
                "aligned_3d main-half head allocation requires an even "
                f"num_heads, got {self.num_heads}."
            )
        num_main_heads = self.num_heads // 2
        num_wrist_heads = self.num_heads - num_main_heads
        num_wrists = num_anchors - 1
        wrist_base, wrist_remainder = divmod(num_wrist_heads, num_wrists)
        head_anchor_indices = [0] * num_main_heads
        for wrist_idx in range(num_wrists):
            wrist_count = wrist_base + int(wrist_idx < wrist_remainder)
            head_anchor_indices.extend([wrist_idx + 1] * wrist_count)
        return torch.tensor(head_anchor_indices, device=device, dtype=torch.long)

    def _build_camera_region_spatial_masks(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a disjoint spatial-token mask for each camera region."""
        layout = self.aligned_3d_action_spatial_anchor_layout
        num_cameras = len(self.aligned_3d_action_spatial_anchors)
        if layout == "center":
            return torch.ones(1, height * width, dtype=torch.bool, device=device)
        if layout in {"horizontal", "libero", "libero_wrist_grounded"}:
            if width < 2:
                raise ValueError(
                    f"aligned_3dp layout {layout!r} requires width >= 2, "
                    f"got width={width}."
                )
            mid_w = width // 2
            regions = (
                (0, height, 0, mid_w),
                (0, height, mid_w, width),
            )
        elif layout == "robotwin":
            if height < 2 or width < 2:
                raise ValueError(
                    "aligned_3dp layout 'robotwin' requires height >= 2 and "
                    f"width >= 2, got {(height, width)}."
                )
            main_h = (height * 2) // 3
            mid_w = width // 2
            regions = (
                (0, main_h, 0, width),
                (main_h, height, 0, mid_w),
                (main_h, height, mid_w, width),
            )
        else:
            raise ValueError(
                "aligned_3dp has no camera-region partition for "
                f"aligned_3d_action_spatial_anchor_layout={layout!r}."
            )
        if len(regions) != num_cameras:
            raise ValueError(
                f"Camera-region count {len(regions)} does not match anchor count "
                f"{num_cameras} for layout {layout!r}."
            )
        masks = []
        for h0, h1, w0, w1 in regions:
            mask = torch.zeros(height, width, dtype=torch.bool, device=device)
            mask[h0:h1, w0:w1] = True
            flat = mask.reshape(-1)
            if not bool(flat.any()):
                raise ValueError(
                    f"aligned_3dp layout {layout!r} produced an empty camera "
                    f"region for grid {(height, width)}."
                )
            masks.append(flat)
        stacked = torch.stack(masks, dim=0)
        if int(stacked.sum()) != height * width or not bool(
            (stacked.sum(dim=0) == 1).all()
        ):
            raise ValueError(
                f"aligned_3dp layout {layout!r} camera regions must partition "
                f"the frame without gaps or overlap; got coverage "
                f"{int(stacked.sum())} for {height * width} tokens."
            )
        return stacked

    def _build_aligned_3dp_per_head_action_mask(
        self,
        base_action_mask: torch.Tensor,
        video_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Restrict each action-attention head to its own camera region."""
        if base_action_mask.ndim != 2:
            raise ValueError(
                "aligned_3dp expects a 2D base action mask "
                f"[Sa, Sv + Sa], got shape {tuple(base_action_mask.shape)}"
            )
        action_seq_len, total_kv_len = base_action_mask.shape
        if total_kv_len <= action_seq_len:
            raise ValueError(
                "aligned_3dp base action mask has no video columns: "
                f"shape {tuple(base_action_mask.shape)}"
            )
        video_seq_len = total_kv_len - action_seq_len
        if video_freqs.shape[0] != video_seq_len:
            raise ValueError(
                "aligned_3dp video_freqs sequence length "
                f"{video_freqs.shape[0]} != action-mask video columns "
                f"{video_seq_len}."
            )
        tokens_per_frame = self._infer_video_tokens_per_frame(video_freqs)
        height, width = self._infer_video_spatial_grid_size(
            video_freqs=video_freqs,
            tokens_per_frame=tokens_per_frame,
        )
        if height * width != tokens_per_frame:
            raise ValueError(
                f"Invalid inferred video spatial grid {(height, width)} for "
                f"tokens_per_frame={tokens_per_frame}."
            )
        region_masks = self._build_camera_region_spatial_masks(
            height=height,
            width=width,
            device=base_action_mask.device,
        )
        head_anchor_indices = self._allocate_aligned_3d_head_anchor_indices(
            device=base_action_mask.device,
        )
        head_spatial = region_masks.index_select(0, head_anchor_indices)
        spatial_idx = (
            torch.arange(video_seq_len, device=base_action_mask.device)
            % tokens_per_frame
        )
        head_video_visible = head_spatial[:, spatial_idx]
        per_head_mask = base_action_mask.unsqueeze(0).expand(
            self.num_heads, -1, -1
        ).clone()
        per_head_mask[:, :, :video_seq_len] &= head_video_visible.unsqueeze(1)
        return per_head_mask

    def _build_new_fused_kv_3d_rope_freqs(
        self,
        video_freqs: torch.Tensor,
        action_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build 3D RoPE freqs for new_fused_kv video K and action Q/K.

        Video K keeps each video token's original spatial h/w RoPE and uses the
        temporal RoPE of action token 0. Each action-attention head uses the full
        video temporal/h/w frequency basis at one camera-view anchor. Half of the
        heads use the main-camera anchor; the remainder are divided evenly among
        wrist-camera anchors. Camera centers are represented continuously in the
        compressed video-token coordinate system.
        """
        t_dim, h_dim, w_dim = self._split_3d_rope_dims()
        expected_rope_dim = t_dim + h_dim + w_dim
        if video_freqs.ndim != 3 or video_freqs.shape[1] != 1:
            raise ValueError(
                f"video_freqs must be [Sv, 1, rope_dim], got {tuple(video_freqs.shape)}"
            )
        if video_freqs.shape[-1] != expected_rope_dim:
            raise ValueError(
                f"video_freqs rope dim {video_freqs.shape[-1]} != expected "
                f"{expected_rope_dim} for attn_head_dim={self.attn_head_dim}"
            )

        temporal_freqs = precompute_freqs_cis(
            self.attn_head_dim - 2 * (self.attn_head_dim // 3),
            end=max(action_seq_len, 1),
        ).to(device=video_freqs.device)
        video_anchor_temporal = temporal_freqs[:1].to(dtype=video_freqs.dtype)
        video_rope_freqs = video_freqs.clone()
        video_rope_freqs[:, :, :t_dim] = video_anchor_temporal.view(1, 1, t_dim)

        tokens_per_frame = self._infer_video_tokens_per_frame(video_freqs)
        height, width = self._infer_video_spatial_grid_size(
            video_freqs=video_freqs,
            tokens_per_frame=tokens_per_frame,
        )
        if height * width != tokens_per_frame:
            raise ValueError(
                f"Invalid inferred video spatial grid {(height, width)} for "
                f"tokens_per_frame={tokens_per_frame}."
            )

        # Recover the supplied video's spatial frequency bases at coordinate 1.
        # Their angles are all in [0, 1], so torch.angle does not wrap them.
        if height > 1:
            h_unit = video_freqs[width, 0, t_dim:t_dim + h_dim]
        else:
            h_unit = precompute_freqs_cis(
                self.attn_head_dim // 3, end=2
            )[1].to(device=video_freqs.device, dtype=video_freqs.dtype)
        if width > 1:
            w_unit = video_freqs[1, 0, t_dim + h_dim:]
        else:
            w_unit = precompute_freqs_cis(
                self.attn_head_dim // 3, end=2
            )[1].to(device=video_freqs.device, dtype=video_freqs.dtype)

        anchor_positions = torch.tensor(
            self.aligned_3d_action_spatial_anchors,
            device=video_freqs.device,
            dtype=torch.angle(h_unit).dtype,
        )
        anchor_h_coordinates = anchor_positions[:, 0] * height - 0.5
        anchor_w_coordinates = anchor_positions[:, 1] * width - 0.5
        h_angles = anchor_h_coordinates[:, None] * torch.angle(h_unit)[None, :]
        w_angles = anchor_w_coordinates[:, None] * torch.angle(w_unit)[None, :]
        anchor_h_freqs = torch.polar(torch.ones_like(h_angles), h_angles)
        anchor_w_freqs = torch.polar(torch.ones_like(w_angles), w_angles)
        anchor_spatial = torch.cat([anchor_h_freqs, anchor_w_freqs], dim=-1)

        head_anchor_indices = self._allocate_aligned_3d_head_anchor_indices(
            device=video_freqs.device,
        )
        action_spatial = anchor_spatial.index_select(0, head_anchor_indices)

        action_temporal = temporal_freqs[:action_seq_len].to(dtype=video_freqs.dtype)
        action_rope_freqs = torch.cat(
            [
                action_temporal.view(action_seq_len, 1, t_dim).expand(
                    -1, self.num_heads, -1
                ),
                action_spatial.unsqueeze(0).expand(action_seq_len, -1, -1),
            ],
            dim=-1,
        )
        return video_rope_freqs, action_rope_freqs

    def _validate_aligned_3d_overlap_anchor_count(self) -> None:
        """Reject anchor layouts ``aligned_3d_overlap`` cannot express.

        The camera helper only ever emits camera ids 0 and 1, so a layout with
        any other anchor count is not merely unsupported -- ``center`` would
        IndexError on camera 1, and ``robotwin`` would SILENTLY drop its third
        anchor and mis-assign two of three cameras.

        Deliberately grid-free so it can run at construction time as well as
        inside the RoPE builder.

        Raises:
            ValueError: when the resolved layout does not supply exactly 2
            anchors.
        """
        anchors_preset = self.aligned_3d_action_spatial_anchors
        if len(anchors_preset) == 2:
            return
        raise ValueError(
            "new_fused_kv_rope_mode='aligned_3d_overlap' requires exactly "
            "one anchor per camera for the 2-camera column split, but "
            "aligned_3d_action_spatial_anchor_layout="
            f"{self.aligned_3d_action_spatial_anchor_layout!r} supplies "
            f"{len(anchors_preset)} anchor(s). This mode assigns every video "
            "token to camera 0 or camera 1 by composite column, so only "
            "2-anchor layouts are meaningful."
        )

    def _build_aligned_3d_overlap_rope_freqs(
        self,
        video_freqs: torch.Tensor,
        action_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build overlapped-chart 3D RoPE freqs for new_fused_kv video K/action Q/K.

        This mode takes ``aligned_3d``'s STATIC preset anchors but folds the two
        camera charts onto each other so that every head shares ONE anchor
        frame. ``aligned_3d`` splits the 24 action-attention heads across two
        anchors at different composite positions (heads 0-11 at the main-camera
        center, 12-23 at the wrist-camera center), so each half of the head
        budget specializes on one view. Here all 24 heads instead see a single
        superimposed frame, so the whole head budget learns from both views
        together.

        Coordinate construction (``ee_rope``'s "Rule B", with static preset
        anchors instead of per-sample EEF anchors):

        - each video token is expressed in CAMERA-LOCAL coordinates relative to
          ITS OWN camera's anchor -- ``rel_row = row - anchor_row[camera]`` and
          ``rel_col = local_col - anchor_col[camera]``;
        - every action token sits at the shared spatial ORIGIN, so its spatial
          phase is identically 1 and is head-invariant;
        - the temporal axis is unchanged from the sibling modes: video K is
          pinned to action time 0 and action token ``j`` sits at time ``j``.

        Under ``horizontal``/``libero`` both anchors are camera-local
        ``(3.0, 3.0)`` on the 7x14 LIBERO composite, so the two charts coincide
        exactly -- the intended arm. The relative formulation above is
        equivalent to placing the action tokens at ``(j, 3.0, 3.0)`` in
        camera-local coordinates in that case, and still degrades gracefully to
        two offset-but-shared charts when the anchors do not coincide (e.g.
        ``libero_wrist_grounded``).

        No attention masking is involved: unlike ``aligned_3dp`` this mode never
        touches the base 2D action mask. No EEF state is read either, so both
        returned tensors are batch-independent.

        Returns:
            ``(video [Sv, 1, rope_dim], action [Sa, 1, rope_dim])``.
        """
        t_dim, h_dim, w_dim = self._split_3d_rope_dims()
        expected_rope_dim = t_dim + h_dim + w_dim
        if video_freqs.ndim != 3 or video_freqs.shape[1] != 1:
            raise ValueError(
                "aligned_3d_overlap video_freqs must be [Sv, 1, rope_dim], got "
                f"{tuple(video_freqs.shape)}"
            )
        if video_freqs.shape[-1] != expected_rope_dim:
            raise ValueError(
                f"aligned_3d_overlap video_freqs rope dim {video_freqs.shape[-1]}"
                f" != expected {expected_rope_dim} for attn_head_dim="
                f"{self.attn_head_dim}"
            )

        # Defense in depth: `__init__` already rejected an invalid layout, but
        # `aligned_3d_action_spatial_anchors` is a plain attribute and a caller
        # could have mutated it after construction.
        #
        # This runs BEFORE any grid/width work on purpose: a config that is wrong
        # in both ways (e.g. a `robotwin` layout on an odd-width grid) must report
        # the mode-level anchor-count problem, which is the one the operator has
        # to fix, rather than the downstream geometry symptom.
        self._validate_aligned_3d_overlap_anchor_count()
        anchors_preset = self.aligned_3d_action_spatial_anchors

        device = video_freqs.device
        video_seq_len = video_freqs.shape[0]
        tokens_per_frame = self._infer_video_tokens_per_frame(video_freqs)
        height, width = self._infer_video_spatial_grid_size(
            video_freqs=video_freqs,
            tokens_per_frame=tokens_per_frame,
        )
        # The camera helper below derives (row, col) from `width` alone and does
        # NOT check that the grid closes, so an inconsistent inference would
        # silently mis-assign cameras. Reject it here, as the siblings do.
        if height * width != tokens_per_frame:
            raise ValueError(
                f"Invalid inferred video spatial grid {(height, width)} for "
                f"tokens_per_frame={tokens_per_frame}."
            )
        # Validate the width HERE rather than letting the shared camera helper
        # raise: that helper is also used by `ee_rope`/`exclusive_ee_rope` and
        # names `ee_rope` in its message, which misdirects an operator running
        # this mode. Same rejection, mode-accurate wording.
        if width % 2 != 0:
            raise ValueError(
                "new_fused_kv_rope_mode='aligned_3d_overlap' requires an even "
                f"composite token width, got {width} (inferred grid "
                f"{(height, width)} for tokens_per_frame={tokens_per_frame}). "
                "This mode splits every video frame into 2 camera-local column "
                "halves and assigns each token to camera 0 or camera 1, so an "
                "odd composite width cannot be split evenly between the cameras."
            )

        camera, token_row, token_local_col = self._ee_rope_camera_of_each_video_token(
            video_seq_len=video_seq_len,
            tokens_per_frame=tokens_per_frame,
            height=height,
            width=width,
            device=device,
        )

        # Recover the supplied video's spatial frequency bases at coordinate 1,
        # exactly as the aligned_3d and ee_rope builders do, so all three share
        # one definition of "one token of travel".
        if height > 1:
            h_unit = video_freqs[width, 0, t_dim:t_dim + h_dim]
        else:
            h_unit = precompute_freqs_cis(
                self.attn_head_dim // 3, end=2
            )[1].to(device=device, dtype=video_freqs.dtype)
        if width > 1:
            w_unit = video_freqs[1, 0, t_dim + h_dim:]
        else:
            w_unit = precompute_freqs_cis(
                self.attn_head_dim // 3, end=2
            )[1].to(device=device, dtype=video_freqs.dtype)
        h_angle_unit = torch.angle(h_unit)
        w_angle_unit = torch.angle(w_unit)

        anchor_positions = torch.tensor(
            anchors_preset,
            device=device,
            dtype=h_angle_unit.dtype,
        )
        # Presets are normalized against the COMPOSITE grid; convert to grid
        # coordinates and then subtract each camera's column offset so both
        # anchors live in the same camera-local frame as the tokens.
        local_width = width // 2
        camera_index = torch.arange(
            anchor_positions.shape[0], device=device, dtype=h_angle_unit.dtype
        )
        anchor_h = anchor_positions[:, 0] * height - 0.5
        anchor_w_local = (
            anchor_positions[:, 1] * width - 0.5 - camera_index * local_width
        )

        # Log once so a non-overlapping layout is visible rather than silently
        # assumed to overlap.
        if not getattr(self, "_logged_aligned_3d_overlap_anchor_frames", False):
            charts_coincide = bool(
                torch.allclose(anchor_h[0], anchor_h[1])
                and torch.allclose(anchor_w_local[0], anchor_w_local[1])
            )
            logger.info(
                "aligned_3d_overlap camera-local anchors on a %dx%d grid "
                "(layout=%r): camera0=(%.6f, %.6f) camera1=(%.6f, %.6f); "
                "charts_coincide=%s",
                height,
                width,
                self.aligned_3d_action_spatial_anchor_layout,
                float(anchor_h[0]),
                float(anchor_w_local[0]),
                float(anchor_h[1]),
                float(anchor_w_local[1]),
                charts_coincide,
            )
            self._logged_aligned_3d_overlap_anchor_frames = True

        # Rule B: each token is offset by its OWN camera's anchor, never by an
        # attending head's anchor. This is what superimposes the two charts.
        rel_row = token_row.to(h_angle_unit.dtype) - anchor_h.index_select(0, camera)
        rel_col = (
            token_local_col.to(w_angle_unit.dtype)
            - anchor_w_local.index_select(0, camera)
        )
        h_angles = rel_row.unsqueeze(-1) * h_angle_unit.view(1, h_dim)
        w_angles = rel_col.unsqueeze(-1) * w_angle_unit.view(1, w_dim)
        video_h = torch.polar(torch.ones_like(h_angles), h_angles)
        video_w = torch.polar(torch.ones_like(w_angles), w_angles)

        temporal_freqs = precompute_freqs_cis(
            self.attn_head_dim - 2 * (self.attn_head_dim // 3),
            end=max(action_seq_len, 1),
        ).to(device=device)
        # Video K is read by action queries, so it sits at action time 0.
        video_temporal = temporal_freqs[:1].to(dtype=video_freqs.dtype)
        video_rope_freqs = torch.cat(
            [
                video_temporal.view(1, t_dim).expand(video_seq_len, t_dim),
                video_h.to(video_freqs.dtype),
                video_w.to(video_freqs.dtype),
            ],
            dim=-1,
        ).unsqueeze(1)                                             # [Sv, 1, D]

        # Every action token sits at the shared spatial ORIGIN, so its spatial
        # phase is identically 1 and carries no head axis.
        action_temporal = temporal_freqs[:action_seq_len].to(dtype=video_freqs.dtype)
        action_spatial = torch.ones(
            action_seq_len,
            h_dim + w_dim,
            device=device,
            dtype=video_freqs.dtype,
        )
        action_rope_freqs = torch.cat(
            [action_temporal.view(action_seq_len, t_dim), action_spatial],
            dim=-1,
        ).unsqueeze(1)                                             # [Sa, 1, D]
        return video_rope_freqs, action_rope_freqs

    # ------------------------------------------------------------------
    # EEF-relative camera RoPE (`ee_rope` / `exclusive_ee_rope`).
    #
    # Rule B: a visual token's coordinate is its own position minus the anchor
    # of THAT TOKEN'S OWN CAMERA -- never the attending head's camera. No head
    # "carries" an anchor, so neither frequency tensor has a head axis:
    #
    #     visual  [B, Sv, 1, rope_dim]     (batch: anchors differ per sample)
    #     action  [Sa, 1, rope_dim]        (every action token sits at the origin)
    #
    # The two modes share every line below. They diverge at exactly one point:
    # `_build_exclusive_ee_rope_action_mask`, which only `exclusive_ee_rope`
    # calls. That single divergence is what makes the arm-2-vs-arm-3 ablation a
    # controlled comparison, so it is asserted by a bitwise-equality test.
    # ------------------------------------------------------------------

    def _ee_rope_camera_of_each_video_token(
        self,
        video_seq_len: int,
        tokens_per_frame: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-token camera id, row, and camera-local column.

        Video tokens flatten as ``(frame, row, col)`` with column changing
        fastest, so each camera's tokens are **interleaved, not contiguous**:
        for a 7x14 composite the main camera owns 0-6, 14-20, 28-34, ... A
        contiguous ``arange(49)`` split silently hands the main group 21 wrist
        tokens and drops 21 real main tokens while still training plausibly.
        """
        if width % 2 != 0:
            raise ValueError(
                f"ee_rope requires an even composite token width, got {width}"
            )
        local_w = width // 2
        index = torch.arange(video_seq_len, device=device)
        within_frame = index % tokens_per_frame
        row = torch.div(within_frame, width, rounding_mode="floor")
        col = within_frame % width
        camera = (col >= local_w).long()
        local_col = col - camera * local_w
        return camera, row.to(torch.float64), local_col.to(torch.float64)

    def _build_ee_rope_freqs(
        self,
        video_freqs: torch.Tensor,
        action_seq_len: int,
        eef_anchor_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build EEF-relative visual/action RoPE frequencies (Rule B).

        Args:
            video_freqs: native video freqs ``[Sv, 1, rope_dim]``.
            action_seq_len: number of action tokens.
            eef_anchor_token: ``[B, 2, 2]`` continuous camera-local anchors,
                ordered ``(main, wrist)`` on axis 1 and **``(y, x)`` -- row
                first** on axis 2. The diagnostic tooling returns ``(x, y)``;
                a swap is silent, so callers must transpose deliberately.

        Returns:
            ``(visual [B, Sv, 1, rope_dim], action [Sa, 1, rope_dim])``.
        """
        t_dim, h_dim, w_dim = self._split_3d_rope_dims()
        expected_rope_dim = t_dim + h_dim + w_dim
        if video_freqs.ndim != 3 or video_freqs.shape[1] != 1:
            raise ValueError(
                f"ee_rope video_freqs must be [Sv, 1, rope_dim], got "
                f"{tuple(video_freqs.shape)}"
            )
        if video_freqs.shape[-1] != expected_rope_dim:
            raise ValueError(
                f"ee_rope video_freqs rope dim {video_freqs.shape[-1]} != "
                f"expected {expected_rope_dim} for attn_head_dim="
                f"{self.attn_head_dim}"
            )
        if eef_anchor_token.ndim != 3 or eef_anchor_token.shape[1:] != (2, 2):
            raise ValueError(
                f"eef_anchor_token must be [B, 2, 2] (cameras=(main,wrist), "
                f"coords=(y,x)), got {tuple(eef_anchor_token.shape)}"
            )
        if not torch.isfinite(eef_anchor_token).all():
            raise ValueError(
                "eef_anchor_token contains nonfinite values; an invalid "
                "projection must fail at resolution time, never reach RoPE."
            )

        device = video_freqs.device
        video_seq_len = video_freqs.shape[0]
        tokens_per_frame = self._infer_video_tokens_per_frame(video_freqs)
        height, width = self._infer_video_spatial_grid_size(
            video_freqs=video_freqs,
            tokens_per_frame=tokens_per_frame,
        )
        if height * width != tokens_per_frame:
            raise ValueError(
                f"Invalid inferred video spatial grid {(height, width)} for "
                f"tokens_per_frame={tokens_per_frame}."
            )

        # Recover the per-dimension angular bases exactly as the aligned_3d
        # builder does, so both share one definition of "one token of travel".
        if height > 1:
            h_unit = video_freqs[width, 0, t_dim:t_dim + h_dim]
        else:
            h_unit = precompute_freqs_cis(
                self.attn_head_dim // 3, end=2
            )[1].to(device=device, dtype=video_freqs.dtype)
        if width > 1:
            w_unit = video_freqs[1, 0, t_dim + h_dim:]
        else:
            w_unit = precompute_freqs_cis(
                self.attn_head_dim // 3, end=2
            )[1].to(device=device, dtype=video_freqs.dtype)
        h_angle_unit = torch.angle(h_unit)
        w_angle_unit = torch.angle(w_unit)

        camera, token_row, token_local_col = self._ee_rope_camera_of_each_video_token(
            video_seq_len=video_seq_len,
            tokens_per_frame=tokens_per_frame,
            height=height,
            width=width,
            device=device,
        )

        anchors = eef_anchor_token.to(device=device, dtype=h_angle_unit.dtype)
        # Gather each token's OWN camera anchor -- this is Rule B.
        anchor_row = anchors[:, :, 0].index_select(1, camera)      # [B, Sv]
        anchor_col = anchors[:, :, 1].index_select(1, camera)      # [B, Sv]
        rel_row = token_row.unsqueeze(0).to(anchor_row.dtype) - anchor_row
        rel_col = token_local_col.unsqueeze(0).to(anchor_col.dtype) - anchor_col

        h_angles = rel_row.unsqueeze(-1) * h_angle_unit.view(1, 1, h_dim)
        w_angles = rel_col.unsqueeze(-1) * w_angle_unit.view(1, 1, w_dim)
        visual_h = torch.polar(torch.ones_like(h_angles), h_angles)
        visual_w = torch.polar(torch.ones_like(w_angles), w_angles)

        temporal_freqs = precompute_freqs_cis(
            self.attn_head_dim - 2 * (self.attn_head_dim // 3),
            end=max(action_seq_len, 1),
        ).to(device=device)
        # Visual K is read by action queries, so it sits at action time 0,
        # matching the aligned_3d convention.
        visual_temporal = temporal_freqs[:1].to(dtype=video_freqs.dtype)
        visual_temporal = visual_temporal.view(1, 1, t_dim).expand(
            anchors.shape[0], video_seq_len, t_dim
        )
        visual_rope_freqs = torch.cat(
            [
                visual_temporal,
                visual_h.to(video_freqs.dtype),
                visual_w.to(video_freqs.dtype),
            ],
            dim=-1,
        ).unsqueeze(2)                                             # [B, Sv, 1, D]

        # Every action token sits at the spatial ORIGIN: the anchor is the
        # origin, so its relative coordinate is (0, 0) and the spatial phase is
        # identically 1. This is why action freqs carry no head axis.
        action_temporal = temporal_freqs[:action_seq_len].to(dtype=video_freqs.dtype)
        action_spatial = torch.ones(
            action_seq_len,
            h_dim + w_dim,
            device=device,
            dtype=video_freqs.dtype,
        )
        action_rope_freqs = torch.cat(
            [action_temporal.view(action_seq_len, t_dim), action_spatial],
            dim=-1,
        ).unsqueeze(1)                                             # [Sa, 1, D]
        return visual_rope_freqs, action_rope_freqs

    def _build_exclusive_ee_rope_action_mask(
        self,
        base_action_mask: torch.Tensor,
        video_freqs: torch.Tensor,
        video_seq_len: int,
        action_seq_len: int,
    ) -> torch.Tensor:
        """Restrict each action head to one camera's first-frame visual tokens.

        Heads ``0..H/2-1`` see main-camera tokens, ``H/2..H-1`` see wrist. All
        heads keep every action key. Later visual frames stay masked because
        the incoming 2D mask already excludes them; this only ever removes
        visibility, never adds it.

        Returns ``[num_heads, Sa, Sv + Sa]``, which broadcasts against SDPA's
        ``[B, num_heads, Sq, Sk]`` without touching the attention kernel.
        """
        if base_action_mask.ndim != 2:
            raise ValueError(
                "exclusive_ee_rope expects a 2D base action mask to refine, "
                f"got ndim={base_action_mask.ndim}"
            )
        expected = (action_seq_len, video_seq_len + action_seq_len)
        if base_action_mask.shape != torch.Size(expected):
            raise ValueError(
                f"exclusive_ee_rope base action mask must be {expected}, got "
                f"{tuple(base_action_mask.shape)}"
            )
        if self.num_heads % 2 != 0:
            raise ValueError(
                "exclusive_ee_rope requires an even num_heads for the 50/50 "
                f"camera split, got {self.num_heads}"
            )

        device = base_action_mask.device
        tokens_per_frame = self._infer_video_tokens_per_frame(video_freqs)
        height, width = self._infer_video_spatial_grid_size(
            video_freqs=video_freqs,
            tokens_per_frame=tokens_per_frame,
        )
        camera, _, _ = self._ee_rope_camera_of_each_video_token(
            video_seq_len=video_seq_len,
            tokens_per_frame=tokens_per_frame,
            height=height,
            width=width,
            device=device,
        )

        num_main_heads = self.num_heads // 2
        head_camera = torch.cat(
            [
                torch.zeros(num_main_heads, device=device, dtype=torch.long),
                torch.ones(
                    self.num_heads - num_main_heads, device=device, dtype=torch.long
                ),
            ]
        )
        # [num_heads, Sv] -- True where the token belongs to that head's camera.
        visual_visible = head_camera[:, None] == camera[None, :]
        action_visible = torch.ones(
            self.num_heads, action_seq_len, device=device, dtype=torch.bool
        )
        head_key_visible = torch.cat([visual_visible, action_visible], dim=1)

        return base_action_mask.unsqueeze(0) & head_key_visible[:, None, :]

    # ------------------------------------------------------------------
    # Shared building blocks (2026-07-05 dedup). The validation preamble,
    # the raw-KV video loop, and the action loop previously existed as
    # 2/4/4 near-identical copies across forward_decoupled,
    # _forward_decoupled_fused_rope, prefill_video_kv, and
    # forward_action_with_video_kv; a fix landing in one copy silently
    # missed the others. The four public methods now differ ONLY in how
    # the per-action-layer video K/V is sourced (selected/mixed stack vs
    # kv_fusion output).
    # ------------------------------------------------------------------
    def _validate_decoupled_forward_inputs(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_masks: Dict[str, torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        t_mod_all: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate per-expert forward inputs; return (video_mask, action_mask)."""
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
        expected_action_mask_shape = (action_seq_len, video_seq_len + action_seq_len)
        if action_mask.shape != torch.Size(expected_action_mask_shape):
            raise ValueError(
                f"Action attention mask must have shape {expected_action_mask_shape}, "
                f"got {tuple(action_mask.shape)}"
            )
        return video_mask, action_mask

    def _run_video_layers_raw_kv(
        self,
        x: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context: Optional[dict],
        video_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Run every video layer (3D-RoPE'd self-attn), caching video K and V.

        By default the cached K is pre-RoPE (raw / pre-norm, depending on the
        projection mode). ``legacy_3d`` instead caches the post-3D-RoPE K used
        in video self-attention, so fusion sees already-positioned keys.

        The full post-block MUST run each layer because layer i+1's K depends
        on layer i's post-block output.

        Returns:
            (final video tokens, per-layer K list, per-layer V list);
            cached tensors are flattened ``[B, Sv, H*Dh]``.
        """
        video_expert = self.mixtures[VIDEO_EXPERT_KEY]
        raw_k_per_layer: list[torch.Tensor] = []
        raw_v_per_layer: list[torch.Tensor] = []
        cache_post_rope_k = self._cache_post_3d_rope_video_k
        for video_layer_idx in range(self.video_num_layers):
            video_block = video_expert.blocks[video_layer_idx]
            build_attention_io = (
                self._build_expert_attention_io_with_raw_k
                if (
                    self.kv_source_mode != "new_fused_kv"
                    or self.new_fused_kv_projection_mode in {
                        "HeadFusedKV",
                        "HeadFusedKV+Sin2DPE",
                        "MLPMixerFusedKV",
                    }
                )
                else self._build_expert_attention_io_with_pre_norm_k
            )
            (
                q_video,
                k_video_rope,
                k_video_raw,
                v_video,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = build_attention_io(
                expert=video_expert,
                block=video_block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )

            mixed = self._mixed_attention(
                q_cat=q_video,
                k_cat=k_video_rope,
                v_cat=v_video,
                attention_mask=video_mask,
            )

            x = self._apply_post_with_optional_checkpoint(
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

            raw_k_per_layer.append(
                k_video_rope if cache_post_rope_k else k_video_raw
            )
            raw_v_per_layer.append(v_video)

        return x, raw_k_per_layer, raw_v_per_layer

    def _validate_cached_action_mask(
        self,
        *,
        action_mask: torch.Tensor,
        video_seq_len: int,
        action_seq_len: int,
        use_new_fused_kv: bool,
    ) -> None:
        """Check the action mask against the sourced video K on the cache path.

        Only the KV-cache inference path calls this: training validates its
        square joint mask upfront. Subclasses whose mask is request-dependent
        override this to admit their own batched form.
        """
        expected_mask_shape = (action_seq_len, video_seq_len + action_seq_len)
        # exclusive_ee_rope and aligned_3dp refine the 2D mask into a
        # head-aware [num_heads, Sa, Sk] mask above.
        head_aware_ok = (
            use_new_fused_kv
            and self.new_fused_kv_rope_mode in {"exclusive_ee_rope", "aligned_3dp"}
        )
        if head_aware_ok:
            expected_head_shape = (self.num_heads, *expected_mask_shape)
            if action_mask.ndim != 3:
                raise ValueError(
                    f"{self.new_fused_kv_rope_mode} "
                    "action_attention_mask must be 3D "
                    f"{expected_head_shape}, got ndim={action_mask.ndim}"
                )
            if action_mask.shape != torch.Size(expected_head_shape):
                raise ValueError(
                    f"{self.new_fused_kv_rope_mode} "
                    "action_attention_mask shape must be "
                    f"{expected_head_shape}, got {tuple(action_mask.shape)}"
                )
        else:
            if action_mask.ndim != 2:
                raise ValueError(
                    f"action_attention_mask must be 2D, "
                    f"got ndim={action_mask.ndim}"
                )
            if action_mask.shape != torch.Size(expected_mask_shape):
                raise ValueError(
                    f"action_attention_mask shape must be {expected_mask_shape}, "
                    f"got {tuple(action_mask.shape)}"
                )

    def _run_action_layers_with_video_kv_source(
        self,
        x: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context: Optional[dict],
        action_mask: torch.Tensor,
        video_kv_for_layer: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
        validate_mask_at_layer0: bool = False,
        video_freqs: Optional[torch.Tensor] = None,
        eef_anchor_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run every action layer against per-layer sourced raw video K/V.

        ``video_kv_for_layer(action_layer_idx)`` returns the cached video K and
        the video V for that action layer. For most modes that K is RAW
        (un-RoPE'd); for ``legacy_3d`` it is already post-3D-RoPE. For
        ``new_fused_kv``, ``aligned_3d`` / ``aligned_3dp`` apply video-style 3D
        RoPE to both video K and action Q/K (``aligned_3dp`` additionally
        restricts each head to its camera-region video tokens), ``aligned_1d``
        applies temporal-only 1D RoPE to both fused video K and action Q/K,
        ``original_3d`` applies the original video 3D RoPE to fused video K and
        keeps action 1D RoPE, while ``video_zero_1d`` / ``legacy_3d`` apply no
        extra video RoPE in this action path. V is never RoPE'd.

        Args:
            validate_mask_at_layer0: When True (KV-cache inference path, where
                no square video mask was validated upfront), check the action
                mask shape against the sourced video K at the first layer.
        """
        action_expert = self.mixtures[ACTION_EXPERT_KEY]
        use_new_fused_kv = self.kv_source_mode == "new_fused_kv"
        use_aligned_new_fused_kv_rope = (
            use_new_fused_kv
            and self.new_fused_kv_rope_mode in self._aligned_3d_family_rope_modes
        )
        use_aligned_3dp_camera_mask = (
            use_new_fused_kv and self.new_fused_kv_rope_mode == "aligned_3dp"
        )
        # Deliberately NOT part of `_aligned_3d_family_rope_modes`: it reuses the
        # family's static anchors but folds both camera charts into one shared
        # frame, and it never refines the action mask.
        use_aligned_3d_overlap_rope = (
            use_new_fused_kv
            and self.new_fused_kv_rope_mode == "aligned_3d_overlap"
        )
        use_aligned_1d_new_fused_kv_rope = (
            use_new_fused_kv and self.new_fused_kv_rope_mode == "aligned_1d"
        )
        use_original_new_fused_kv_rope = (
            use_new_fused_kv and self.new_fused_kv_rope_mode == "original_3d"
        )
        # video_zero_1d: fuse pre-RoPE K, skip action-path video RoPE.
        # legacy_3d: fuse post-3D-RoPE K, skip action-path video RoPE (same skip).
        use_skip_action_path_video_rope = (
            use_new_fused_kv
            and self.new_fused_kv_rope_mode in {"video_zero_1d", "legacy_3d"}
        )
        use_ee_rope = (
            use_new_fused_kv and self.new_fused_kv_rope_mode in self.ee_rope_modes
        )
        if use_ee_rope:
            if video_freqs is None:
                raise ValueError(
                    f"new_fused_kv {self.new_fused_kv_rope_mode!r} action path "
                    "requires video_freqs to build EEF-relative RoPE."
                )
            if eef_anchor_token is None:
                raise ValueError(
                    f"new_fused_kv {self.new_fused_kv_rope_mode!r} requires "
                    "eef_anchor_token [B, 2, 2]; it must be threaded from "
                    "build_inputs() and never defaulted."
                )
            video_rope_freqs, action_rope_freqs = self._build_ee_rope_freqs(
                video_freqs=video_freqs,
                action_seq_len=x.shape[1],
                eef_anchor_token=eef_anchor_token,
            )
            # THE single divergence point between the two sibling modes.
            if self.new_fused_kv_rope_mode == "exclusive_ee_rope":
                action_mask = self._build_exclusive_ee_rope_action_mask(
                    base_action_mask=action_mask,
                    video_freqs=video_freqs,
                    video_seq_len=video_freqs.shape[0],
                    action_seq_len=x.shape[1],
                )
        elif use_aligned_3d_overlap_rope:
            if video_freqs is None:
                raise ValueError(
                    "new_fused_kv aligned_3d_overlap action path requires "
                    "video_freqs to build the shared camera-local anchor frame."
                )
            video_rope_freqs, action_rope_freqs = (
                self._build_aligned_3d_overlap_rope_freqs(
                    video_freqs=video_freqs,
                    action_seq_len=x.shape[1],
                )
            )
        elif use_aligned_new_fused_kv_rope:
            if video_freqs is None:
                raise ValueError(
                    "new_fused_kv fixed-RoPE action path requires video_freqs "
                    "to build video-style 3D RoPE for action attention."
                )
            video_rope_freqs, action_rope_freqs = self._build_new_fused_kv_3d_rope_freqs(
                video_freqs=video_freqs,
                action_seq_len=x.shape[1],
            )
            if use_aligned_3dp_camera_mask:
                action_mask = self._build_aligned_3dp_per_head_action_mask(
                    base_action_mask=action_mask,
                    video_freqs=video_freqs,
                )
        elif use_original_new_fused_kv_rope:
            if video_freqs is None:
                raise ValueError(
                    "new_fused_kv original-3D-RoPE action path requires video_freqs "
                    "to apply the original video 3D RoPE to fused video K."
                )
            # Rank/rope-dim validation for the original_3d branch. Unlike the
            # aligned_3d path, original_3d uses video_freqs verbatim as the RoPE
            # basis for rope_apply(k_video_raw, video_rope_freqs, num_heads), so a
            # malformed freqs tensor (wrong rank, or a last dim that does not match
            # attn_head_dim // 2) would silently mis-rotate every video token. This
            # mirrors the shape contract asserted by the aligned-path builder
            # _build_new_fused_kv_3d_rope_freqs (t_dim + h_dim + w_dim). It is
            # complementary to the 5fb5881 length guard (freqs.shape[0] vs Sv),
            # which is enforced separately at the per-layer rope_apply site.
            t_dim, h_dim, w_dim = self._split_3d_rope_dims()
            expected_rope_dim = t_dim + h_dim + w_dim
            if video_freqs.ndim != 3 or video_freqs.shape[1] != 1:
                raise ValueError(
                    "new_fused_kv original_3d video_freqs must be "
                    "[Sv, 1, rope_dim], got "
                    f"{tuple(video_freqs.shape)}"
                )
            if video_freqs.shape[-1] != expected_rope_dim:
                raise ValueError(
                    "new_fused_kv original_3d video_freqs rope dim "
                    f"{video_freqs.shape[-1]} != expected {expected_rope_dim} "
                    f"for attn_head_dim={self.attn_head_dim}"
                )
            video_rope_freqs = video_freqs
            action_rope_freqs = action_freqs
        elif use_aligned_1d_new_fused_kv_rope:
            video_rope_freqs = action_freqs[:1]
            action_rope_freqs = action_freqs
        elif use_skip_action_path_video_rope:
            video_rope_freqs = None
            action_rope_freqs = action_freqs
        else:
            video_rope_freqs = action_freqs[:1]
            action_rope_freqs = action_freqs

        for action_layer_idx in range(self.action_num_layers):
            k_video_raw, v_video = video_kv_for_layer(action_layer_idx)
            if (
                use_new_fused_kv
                and self.new_fused_kv_projection_mode
                in {"simple+PE", "simple+PE-postnorm"}
            ):
                if video_freqs is None:
                    raise ValueError(
                        f"{self.new_fused_kv_projection_mode} new_fused_kv "
                        "requires video_freqs to build spatial 2D sin/cos PE."
                    )
                if self.new_fused_kv_projection_mode == "simple+PE":
                    k_video_raw = self._apply_simple_pe_to_video_k(
                        k_video=k_video_raw,
                        action_layer_idx=action_layer_idx,
                        video_freqs=video_freqs,
                    )
                    if self.k_fused_norm is not None:
                        k_video_raw = self.k_fused_norm[action_layer_idx](k_video_raw)
                else:
                    k_video_raw = k_video_raw + self._build_simple_pe_for_video_k(
                        k_video=k_video_raw,
                        action_layer_idx=action_layer_idx,
                        video_freqs=video_freqs,
                    )
            if (
                use_new_fused_kv
                and self.new_fused_kv_projection_mode == "HeadFusedKV+Sin2DPE"
            ):
                if video_freqs is None:
                    raise ValueError(
                        "HeadFusedKV+Sin2DPE requires video_freqs to build spatial 2D sin/cos PE."
                    )
                k_video_raw = k_video_raw + self._build_head_fused_kv_sin2d_pe(
                    k_video=k_video_raw,
                    action_layer_idx=action_layer_idx,
                    video_freqs=video_freqs,
                )
                if self.k_fused_norm is not None:
                    k_video_raw = self.k_fused_norm[action_layer_idx](k_video_raw)
            if use_new_fused_kv:
                if use_skip_action_path_video_rope:
                    k_video = k_video_raw
                else:
                    assert video_rope_freqs is not None
                    k_video = rope_apply(k_video_raw, video_rope_freqs, self.num_heads)
            else:
                k_video = self._apply_action_zero_rope_to_video_k(
                    k_video_raw=k_video_raw,
                    action_freqs=video_rope_freqs,
                )

            if validate_mask_at_layer0 and action_layer_idx == 0:
                self._validate_cached_action_mask(
                    action_mask=action_mask,
                    video_seq_len=k_video.shape[1],
                    action_seq_len=x.shape[1],
                    use_new_fused_kv=use_new_fused_kv,
                )

            action_block = action_expert.blocks[action_layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=action_expert,
                block=action_block,
                x=x,
                freqs=action_rope_freqs,
                t_mod=action_t_mod,
            )

            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)

            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_mask,
            )

            x = self._apply_post_with_optional_checkpoint(
                block=action_block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context,
            )

        return x

    def forward_decoupled(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_masks: Dict[str, torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        eef_anchor_token: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward with raw video-K fusion and action-aligned RoPE.

        ``eef_anchor_token`` is ``[B, 2, 2]`` and required only by the
        ``ee_rope`` / ``exclusive_ee_rope`` modes; every other mode ignores it.
        """
        # Fused-MLP mode routes through a dedicated fuse-then-RoPE path. Selected
        # modes (final_only / uniform_end / fused_kv) keep the body below.
        if self.kv_fusion is not None:
            return self._forward_decoupled_fused_rope(
                embeds_all=embeds_all,
                attention_masks=attention_masks,
                freqs_all=freqs_all,
                context_all=context_all,
                t_mod_all=t_mod_all,
                eef_anchor_token=eef_anchor_token,
            )

        video_mask, action_mask = self._validate_decoupled_forward_inputs(
            embeds_all=embeds_all,
            attention_masks=attention_masks,
            freqs_all=freqs_all,
            t_mod_all=t_mod_all,
        )

        x_video, raw_k_per_layer, raw_v_per_layer = self._run_video_layers_raw_kv(
            x=embeds_all[VIDEO_EXPERT_KEY],
            video_freqs=freqs_all[VIDEO_EXPERT_KEY],
            video_t_mod=t_mod_all[VIDEO_EXPERT_KEY],
            video_context=context_all.get(VIDEO_EXPERT_KEY),
            video_mask=video_mask,
        )

        video_kv_cache = [
            {"k": k, "v": v} for k, v in zip(raw_k_per_layer, raw_v_per_layer)
        ]
        stacked_k, stacked_v = self._stack_video_kv(video_kv_cache)

        x_action = self._run_action_layers_with_video_kv_source(
            x=embeds_all[ACTION_EXPERT_KEY],
            action_freqs=freqs_all[ACTION_EXPERT_KEY],
            action_t_mod=t_mod_all[ACTION_EXPERT_KEY],
            action_context=context_all.get(ACTION_EXPERT_KEY),
            action_mask=action_mask,
            video_kv_for_layer=lambda idx: self._select_or_mix_stacked_video_kv(
                stacked_k=stacked_k,
                stacked_v=stacked_v,
                action_layer_idx=idx,
            ),
            video_freqs=freqs_all[VIDEO_EXPERT_KEY],
            eef_anchor_token=eef_anchor_token,
        )

        return {
            VIDEO_EXPERT_KEY: x_video,
            ACTION_EXPERT_KEY: x_action,
        }

    def _forward_decoupled_fused_rope(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_masks: Dict[str, torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        eef_anchor_token: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward for ``fused_mlp`` mode with fuse-then-RoPE.

        Video self-attention is identical to the selected-mode path: it uses the
        video's native 3D RoPE. The raw (un-RoPE'd) video K and V of every video
        layer are cached, stacked into ``[B, Sv, N, D]``, and fused by
        ``self.kv_fusion`` into one K/V per action layer. Each fused, still-raw K
        is then re-RoPE'd right before it is concatenated with the action K
        (fuse-then-RoPE, per action layer). This
        mirrors the ``_select_or_mix_stacked_video_kv`` call in the selected-mode
        path, but swaps selection/mixing for the learned MLP fusion.

        Args:
            embeds_all: Per-expert input tokens (``video`` / ``action``).
            attention_masks: Per-expert attention masks (video square, action
                rectangular ``[Sa, Sv + Sa]``).
            freqs_all: Per-expert RoPE frequencies.
            context_all: Per-expert optional cross-attention context payloads.
            t_mod_all: Per-expert AdaLN time-modulation tensors.

        Returns:
            Dict with denoised ``video`` and ``action`` tokens.
        """
        video_mask, action_mask = self._validate_decoupled_forward_inputs(
            embeds_all=embeds_all,
            attention_masks=attention_masks,
            freqs_all=freqs_all,
            t_mod_all=t_mod_all,
        )

        action_freqs = freqs_all[ACTION_EXPERT_KEY]

        # ---- Video loop: run self-attn (3D RoPE), cache RAW K + V per layer. ----
        x_video, raw_k_per_layer, raw_v_per_layer = self._run_video_layers_raw_kv(
            x=embeds_all[VIDEO_EXPERT_KEY],
            video_freqs=freqs_all[VIDEO_EXPERT_KEY],
            video_t_mod=t_mod_all[VIDEO_EXPERT_KEY],
            video_context=context_all.get(VIDEO_EXPERT_KEY),
            video_mask=video_mask,
        )

        # Stack raw K/V over video layers -> [B, Sv, N, D] for the fusion module.
        all_k = torch.stack(raw_k_per_layer, dim=2)
        all_v = torch.stack(raw_v_per_layer, dim=2)
        fused_kv = self.kv_fusion(all_k, all_v)  # list length M of {"k","v"} [B,Sv,D]

        # ---- Action loop: fuse-then-RoPE per action layer (the RoPE rotation
        # of the fused raw K happens inside the shared action-loop helper). ----
        x_action = self._run_action_layers_with_video_kv_source(
            x=embeds_all[ACTION_EXPERT_KEY],
            action_freqs=action_freqs,
            action_t_mod=t_mod_all[ACTION_EXPERT_KEY],
            action_context=context_all.get(ACTION_EXPERT_KEY),
            action_mask=action_mask,
            video_kv_for_layer=lambda idx: (
                fused_kv[idx]["k"],
                fused_kv[idx]["v"],
            ),
            video_freqs=freqs_all[VIDEO_EXPERT_KEY],
            eef_anchor_token=eef_anchor_token,
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
        """Run video layers and cache video K plus normal V for action.

        The return contract is mode-dependent, mirroring the two branches below:

        - ``kv_fusion is not None`` (fused_mlp): the K/V of every video layer
          are stacked into ``[B, Sv, N, D]`` and fused by ``self.kv_fusion`` into
          one K/V per action layer. Returns the length ``action_num_layers`` (M)
          fused list of ``{"k","v"}`` tensors ``[B, Sv, D]``, exactly what
          ``forward_action_with_video_kv`` consumes in fused mode.
        - ``kv_fusion is None`` (selected modes): returns the length
          ``video_num_layers`` (N) list of per-layer ``{"k","v"}`` caches, to
          be stacked/selected downstream by ``forward_action_with_video_kv``.

        By default the cached K is RAW (un-RoPE'd) and action attention applies
        mode-specific RoPE later in ``forward_action_with_video_kv``.
        ``legacy_3d`` instead caches post-3D-RoPE K and skips extra video RoPE
        in the action path. V is never RoPE'd.

        Args:
            video_tokens: Video expert input tokens ``[B, Sv, D_video]``.
            video_freqs: Video 3D RoPE frequencies for the self-attention loop.
            video_t_mod: Video AdaLN time-modulation tensor.
            video_context_payload: Optional video cross-attention context payload.
            video_attention_mask: Square video self-attention mask ``[Sv, Sv]``.

        Returns:
            list of ``{"k","v"}`` dicts. Length is ``action_num_layers`` (M) when
            ``kv_fusion is not None`` (fused list), else ``video_num_layers`` (N)
            (raw per-layer list). Each tensor has shape ``[B, Sv, D]``.
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

        # Run the full video block per layer (self-attn uses 3D-RoPE'd K),
        # caching K/V of every layer via the shared training helper. Default
        # modes cache RAW K; ``legacy_3d`` caches post-3D-RoPE K.
        _, raw_k_per_layer, raw_v_per_layer = self._run_video_layers_raw_kv(
            x=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context_payload,
            video_mask=video_attention_mask,
        )

        if self.kv_fusion is not None:
            # Stack raw K/V over video layers -> [B, Sv, N, D] for the fusion
            # module. dtype invariant: fused K stays in the model dtype; the
            # action-token-0 RoPE applied later in forward returns .to(x.dtype)
            # (wan_video_dit.py rope_apply), so no cast is needed here.
            all_k = torch.stack(raw_k_per_layer, dim=2)
            all_v = torch.stack(raw_v_per_layer, dim=2)
            # Length == action_num_layers list of {"k","v"} tensors [B, Sv, D].
            return self.kv_fusion(all_k, all_v)

        return [
            {"k": k, "v": v, "freqs": video_freqs}
            if self.kv_source_mode == "new_fused_kv"
            else {"k": k, "v": v}
            for k, v in zip(raw_k_per_layer, raw_v_per_layer)
        ]

    def forward_action_with_video_kv(
        self,
        video_kv_per_layer: list[dict[str, torch.Tensor]],
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        action_attention_mask: torch.Tensor,
        eef_anchor_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run action using raw cached video K rotated by action token-0 RoPE.

        The expected length of ``video_kv_per_layer`` is mode-dependent and must
        match what ``prefill_video_kv`` produced:

        - ``kv_fusion is not None`` (fused_mlp): ``video_kv_per_layer`` is the
          length ``action_num_layers`` (M) FUSED list from ``prefill_video_kv``
          (already one raw K/V per action layer). ``_stack_video_kv`` is NOT
          called (it asserts len == ``video_num_layers``); each action layer
          re-RoPEs its own fused raw K and uses V raw.
        - ``kv_fusion is None`` (selected modes): ``video_kv_per_layer`` is the
          length ``video_num_layers`` (N) raw per-layer list; it is stacked and
          the selected/mixed K/V per action layer is re-RoPE'd.

        Mirrors the training action loop in ``_forward_decoupled_fused_rope``
        (fused mode) / ``forward_decoupled`` (selected mode).

        Args:
            video_kv_per_layer: Cached video K/V from ``prefill_video_kv``. Length
                ``action_num_layers`` (M) when ``kv_fusion is not None`` (fused
                list), else ``video_num_layers`` (N) raw per-layer list. Each
                ``{"k","v"}`` tensor has shape ``[B, Sv, D]``.
            action_tokens: Action expert input tokens ``[B, Sa, D_action]``.
            action_freqs: Action RoPE frequencies for the legacy fixed-RoPE path.
                ``new_fused_kv`` uses action length plus cached ``video_freqs``
                to build video-style 3D RoPE instead.
            action_t_mod: Action AdaLN time-modulation tensor.
            action_context_payload: Optional action cross-attention context.
            action_attention_mask: Rectangular action mask ``[Sa, Sv + Sa]``.

        Returns:
            Denoised action tokens ``[B, Sa, D_action]``.
        """
        if self.kv_fusion is not None:
            # Fused-MLP mode: video_kv_per_layer is the length-M fused list from
            # prefill_video_kv (already one raw K/V per action layer). Do NOT
            # call _stack_video_kv (it asserts len == video_num_layers). Per
            # action layer, the shared helper re-RoPEs the fused raw K with
                # mode-specific fixed RoPE and uses V raw. Mirrors the training
                # action loop in _forward_decoupled_fused_rope.
            if len(video_kv_per_layer) != self.action_num_layers:
                raise ValueError(
                    f"video_kv_per_layer length {len(video_kv_per_layer)} != "
                    f"action_num_layers {self.action_num_layers}"
                )
            return self._run_action_layers_with_video_kv_source(
                x=action_tokens,
                action_freqs=action_freqs,
                action_t_mod=action_t_mod,
                action_context=action_context_payload,
                action_mask=action_attention_mask,
                video_kv_for_layer=lambda idx: (
                    video_kv_per_layer[idx]["k"],
                    video_kv_per_layer[idx]["v"],
                ),
                validate_mask_at_layer0=True,
                eef_anchor_token=eef_anchor_token,
            )

        stacked_k, stacked_v = self._stack_video_kv(video_kv_per_layer)
        video_freqs = None
        if self.kv_source_mode == "new_fused_kv":
            video_freqs = video_kv_per_layer[0].get("freqs")
            if video_freqs is None:
                raise ValueError(
                    "new_fused_kv fixed-RoPE inference cache must include "
                    "`freqs`; use prefill_video_kv() from the same model."
                )
            # Sequence-axis derivation: the cached sidecar is consumed by
            # rope_apply(k_video_raw, video_rope_freqs, num_heads) at
            # _run_action_layers_with_video_kv_source line ~414. There
            # k_video_raw is [B, Sv, D] -> rearranged to [B, Sv, n, d] and the
            # freqs (video_rope_freqs = video_freqs.clone(), so same axis-0
            # length) broadcast as [Sv, 1, rope_dim/2] against [B, Sv, n, d/2].
            # rope_apply broadcasts positionally, so a shorter/longer freqs
            # axis-0 would either error or (when equal by coincidence) silently
            # mis-rotate; axis 0 of video_freqs MUST equal the cached video K
            # sequence length Sv. _build_new_fused_kv_3d_rope_freqs asserts
            # video_freqs is [Sv, 1, rope_dim] (line ~217), fixing Sv on axis 0.
            # stacked_k is [N, B, Sv, num_heads, attn_head_dim] from
            # _stack_video_kv, so Sv = stacked_k.shape[2].
            cached_video_seq_len = stacked_k.shape[2]
            freqs_seq_len = video_freqs.shape[0]
            if freqs_seq_len != cached_video_seq_len:
                raise ValueError(
                    "new_fused_kv cached `freqs` sequence length "
                    f"{freqs_seq_len} != cached video K sequence length "
                    f"{cached_video_seq_len}; the freqs sidecar is malformed or "
                    "stale. Regenerate the cache with prefill_video_kv() from "
                    "the same model."
                )
        return self._run_action_layers_with_video_kv_source(
            x=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=action_t_mod,
            action_context=action_context_payload,
            action_mask=action_attention_mask,
            video_kv_for_layer=lambda idx: self._select_or_mix_stacked_video_kv(
                stacked_k=stacked_k,
                stacked_v=stacked_v,
                action_layer_idx=idx,
            ),
            validate_mask_at_layer0=True,
            video_freqs=video_freqs,
            eef_anchor_token=eef_anchor_token,
        )


class FasterWAMDecoupled(FastWAMDecoupled):
    """FastWAMDecoupled variant using action-aligned video K RoPE."""

    def build_inputs(self, sample, tiled: bool = False):
        """Extend the base inputs with the EEF anchor when a mode needs it.

        Overridden here rather than in ``FastWAM.build_inputs`` (upstream) or in
        ``FastWAMDecoupled`` (which never runs these modes): the EEF-relative
        RoPE modes are hosted by this class, so the input contract belongs with
        them. Adds one key and changes nothing else.

        ``eef_anchor_token`` is ``[B, 2, 2]`` float32, cameras ordered
        ``(main, wrist)`` and coordinates ``(y, x)`` -- row first. The
        ``eef_anchor_observed`` flag is deliberately NOT threaded here: nothing
        in attention, masking, or frequency construction reads it, and an unread
        tensor in the model contract is exactly the field a later reader
        re-interprets as a rejection gate (plan Sections 16.2, 19.2).
        """
        inputs = super().build_inputs(sample, tiled=tiled)
        anchor = sample.get("eef_anchor_token")
        if anchor is None:
            return inputs
        if not torch.is_tensor(anchor):
            anchor = torch.as_tensor(anchor)
        if anchor.ndim != 3 or anchor.shape[1:] != (2, 2):
            raise ValueError(
                "`sample['eef_anchor_token']` must be [B, 2, 2] with cameras "
                f"(main, wrist) and coords (y, x); got {tuple(anchor.shape)}"
            )
        if not torch.isfinite(anchor).all():
            raise ValueError(
                "`sample['eef_anchor_token']` contains nonfinite values; an "
                "invalid projection must fail during load-time resolution, "
                "never reach the model (plan Section 19.1 Decision 1)."
            )
        inputs["eef_anchor_token"] = anchor.to(
            device=self.device, dtype=torch.float32, non_blocking=True
        )
        return inputs

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        decoupled: bool = True,
        kv_source_mapping: list[int] | None = None,
        kv_source_mode: str = "final_only",
        fixed_rope: bool = True,
        new_fused_kv_rope_mode: str = "aligned_3d",
        aligned_3d_action_spatial_anchor_layout: str | None = "center",
        eef_calibration_path: str | None = None,
        eef_raw_source_resolution: int | None = None,
        eef_processed_video_size: list[int] | tuple[int, int] | None = None,
        new_fused_kv_projection_mode: str = "full",
        new_fused_kv_pos_embed_max_tokens: int = 4096,
        new_fused_kv_pos_embed_dim: int = 128,
        new_fused_kv_mlp_mixer_num_blocks: int = 1,
        new_fused_kv_mlp_mixer_token_mlp_ratio: float = 4.0,
        new_fused_kv_mlp_mixer_channel_mlp_ratio: float = 4.0,
        new_fused_kv_head_fused_kv_sin2d_pe_mlp_hidden_dim: int = 512,
        new_fused_kv_simple_head_softmax_fuse_mode: str = "all",
        new_fused_kv_head_fused_kv_fuse_mode: str = "all",
        kv_fusion: "torch.nn.Module | None" = None,
    ):
        """Load Wan components and build a validated decoupled FasterWAM model."""
        if video_dit_config is None:
            raise ValueError(
                "`video_dit_config` is required for "
                "FasterWAMDecoupled.from_wan22_pretrained()."
            )
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required.")
        if not decoupled:
            raise ValueError("FasterWAMDecoupled requires decoupled=True.")
        new_fused_kv_projection_mode = _validate_new_fused_kv_projection_mode(
            kv_source_mode,
            new_fused_kv_projection_mode,
        )

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        layer_init_mapping = kv_source_mapping
        # Both "fused_kv" and "fused_mlp" fuse across all/many video layers, so
        # neither has a 1:1 action->video mapping usable for selective init. When
        # action_dit_pretrained_path is None (the fused_mlp training case), this
        # mapping is ignored by ActionDiT.from_pretrained (random init) anyway.
        if kv_source_mode in ("fused_kv", "new_fused_kv", "fused_mlp"):
            from .mot_decoupled import compute_kv_source_mapping

            action_config_for_init = action_dit_config or {}
            action_num_layers = int(action_config_for_init.get("num_layers", 5))
            if kv_source_mapping is not None and len(kv_source_mapping) == action_num_layers:
                layer_init_mapping = kv_source_mapping
            else:
                layer_init_mapping = compute_kv_source_mapping(
                    mode="uniform_end",
                    video_num_layers=len(video_expert.blocks),
                    action_num_layers=action_num_layers,
                )

        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
            layer_init_mapping=layer_init_mapping,
        )

        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError(
                "ActionDiT `num_heads` must match video expert for mixed attention."
            )
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError(
                "ActionDiT `attn_head_dim` must match video expert for mixed attention."
            )

        mot_cls = (
            MoTDecoupledActionAlignedVideoRoPE
            if fixed_rope or kv_source_mode == "new_fused_kv"
            else MoTDecoupled
        )
        logger.info(
            "FasterWAMDecoupled fixed_rope=%s, kv_source_mode=%s, "
            "new_fused_kv_rope_mode=%s, aligned_3d_anchor_layout=%s, "
            "new_fused_kv_projection_mode=%s, "
            "simple_head_softmax_fuse_mode=%s, "
            "head_fused_kv_fuse_mode=%s, "
            "using MoT class %s",
            fixed_rope,
            kv_source_mode,
            new_fused_kv_rope_mode,
            aligned_3d_action_spatial_anchor_layout,
            new_fused_kv_projection_mode,
            new_fused_kv_simple_head_softmax_fuse_mode,
            new_fused_kv_head_fused_kv_fuse_mode,
            mot_cls.__name__,
        )
        eef_geometry_identity = None
        if new_fused_kv_rope_mode in MoTDecoupledActionAlignedVideoRoPE.ee_rope_modes:
            if eef_calibration_path is None:
                raise ValueError(
                    f"new_fused_kv_rope_mode={new_fused_kv_rope_mode!r} requires "
                    "eef_calibration_path"
                )
            from fastwam.geometry import (
                CAMERA_ORDER,
                EEF_PROJECTION_VERSION,
                VAE_SPATIAL_FACTOR,
                calibration_digest,
            )

            if eef_raw_source_resolution is None:
                raise ValueError(
                    f"{new_fused_kv_rope_mode} requires "
                    "eef_raw_source_resolution derived from data.shape_meta"
                )
            if eef_processed_video_size is None or len(eef_processed_video_size) != 2:
                raise ValueError(
                    f"{new_fused_kv_rope_mode} requires eef_processed_video_size "
                    "[H,W] derived from data.video_size"
                )
            processed_h, processed_w = map(int, eef_processed_video_size)
            patch = tuple(int(v) for v in video_dit_config.get("patch_size", ()))
            if len(patch) != 3:
                raise ValueError(
                    "video_dit_config.patch_size must be [T,H,W] to derive the "
                    f"EEF token grid, got {patch}"
                )
            divisor_h = VAE_SPATIAL_FACTOR * patch[1]
            divisor_w = VAE_SPATIAL_FACTOR * patch[2]
            if processed_h % divisor_h or processed_w % divisor_w:
                raise ValueError(
                    f"processed video {processed_h}x{processed_w} is not divisible "
                    f"by VAE*patch {divisor_h}x{divisor_w}"
                )
            eef_geometry_identity = {
                "calibration_digest": calibration_digest(eef_calibration_path),
                "raw_source_resolution": int(eef_raw_source_resolution),
                "token_grid_h": processed_h // divisor_h,
                "token_grid_w": processed_w // divisor_w,
                "camera_order": list(CAMERA_ORDER),
                "projection_version": EEF_PROJECTION_VERSION,
            }

        mot_kwargs = {}
        if mot_cls is MoTDecoupledActionAlignedVideoRoPE:
            mot_kwargs["new_fused_kv_rope_mode"] = new_fused_kv_rope_mode
            mot_kwargs["eef_geometry_identity"] = eef_geometry_identity
            mot_kwargs["aligned_3d_action_spatial_anchor_layout"] = (
                aligned_3d_action_spatial_anchor_layout
            )
        mot_kwargs["new_fused_kv_projection_mode"] = new_fused_kv_projection_mode
        if kv_source_mode == "new_fused_kv":
            mot_kwargs["new_fused_kv_pos_embed_max_tokens"] = new_fused_kv_pos_embed_max_tokens
            mot_kwargs["new_fused_kv_pos_embed_dim"] = new_fused_kv_pos_embed_dim
            mot_kwargs["new_fused_kv_mlp_mixer_num_blocks"] = new_fused_kv_mlp_mixer_num_blocks
            mot_kwargs["new_fused_kv_mlp_mixer_token_mlp_ratio"] = new_fused_kv_mlp_mixer_token_mlp_ratio
            mot_kwargs["new_fused_kv_mlp_mixer_channel_mlp_ratio"] = new_fused_kv_mlp_mixer_channel_mlp_ratio
            mot_kwargs["new_fused_kv_head_fused_kv_sin2d_pe_mlp_hidden_dim"] = new_fused_kv_head_fused_kv_sin2d_pe_mlp_hidden_dim
            mot_kwargs["new_fused_kv_simple_head_softmax_fuse_mode"] = (
                new_fused_kv_simple_head_softmax_fuse_mode
            )
            mot_kwargs["new_fused_kv_head_fused_kv_fuse_mode"] = (
                new_fused_kv_head_fused_kv_fuse_mode
            )
        mot = mot_cls(
            mixtures={"video": video_expert, "action": action_expert},
            video_num_layers=len(video_expert.blocks),
            action_num_layers=len(action_expert.blocks),
            num_heads=int(video_expert.num_heads),
            attn_head_dim=int(video_expert.attn_head_dim),
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            kv_source_mapping=kv_source_mapping,
            kv_source_mode=kv_source_mode,
            kv_fusion=kv_fusion,
            **mot_kwargs,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        model.eef_calibration_path = eef_calibration_path
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model


def create_fasterwam_decoupled(
    model_id: str,
    tokenizer_model_id: str,
    tokenizer_max_len: int,
    load_text_encoder: bool,
    proprio_dim: int | None,
    redirect_common_files: bool,
    mot_checkpoint_mixed_attn: bool,
    action_dit_pretrained_path: str | None,
    skip_dit_load_from_pretrain: bool,
    decoupled: bool,
    kv_source_mode: str,
    video_dit_config: dict,
    action_dit_config: dict,
    video_scheduler: dict,
    action_scheduler: dict,
    loss: dict | None = None,
    device: str = "cuda",
    model_dtype: torch.dtype = torch.bfloat16,
    fixed_rope: bool = True,
    new_fused_kv_rope_mode: str = "aligned_3d",
    aligned_3d_action_spatial_anchor_layout: str | None = "center",
    eef_calibration_path: str | None = None,
    eef_raw_source_resolution: int | None = None,
    eef_processed_video_size: list[int] | tuple[int, int] | None = None,
    new_fused_kv_projection_mode: str = "full",
    new_fused_kv_pos_embed_max_tokens: int = 4096,
    new_fused_kv_pos_embed_dim: int = 128,
    new_fused_kv_mlp_mixer_num_blocks: int = 1,
    new_fused_kv_mlp_mixer_token_mlp_ratio: float = 4.0,
    new_fused_kv_mlp_mixer_channel_mlp_ratio: float = 4.0,
    new_fused_kv_head_fused_kv_sin2d_pe_mlp_hidden_dim: int = 512,
    new_fused_kv_simple_head_softmax_fuse_mode: str = "all",
    new_fused_kv_head_fused_kv_fuse_mode: str = "all",
    fusion_hidden_dim: int = 64,
    fusion_use_norm: bool = True,
) -> FasterWAMDecoupled:
    """Hydra factory for ``FasterWAMDecoupled``."""
    from omegaconf import DictConfig, OmegaConf
    from .mot_decoupled import compute_kv_source_mapping

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)

    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to dict, got {type(video_dit_config)}")
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to dict, got {type(action_dit_config)}")
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must resolve to dict, got {type(video_scheduler)}")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must resolve to dict, got {type(action_scheduler)}")
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must resolve to dict, got {type(loss)}")
    if not decoupled:
        raise ValueError("create_fasterwam_decoupled requires decoupled=True.")
    new_fused_kv_projection_mode = _validate_new_fused_kv_projection_mode(
        kv_source_mode,
        new_fused_kv_projection_mode,
    )

    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}"
        )

    action_dit_config = dict(action_dit_config)
    kv_fusion = None  # set by the fused_mlp branch below
    layer_selected = action_dit_config.pop("layer_selected", None)
    if layer_selected is not None:
        # layer_selected directly selects video KV sources, which is mutually
        # exclusive with the learned MLP fusion of fused_mlp mode.
        if kv_source_mode == "fused_mlp":
            raise ValueError(
                "layer_selected and kv_source_mode='fused_mlp' are incompatible"
            )
        action_dit_config["num_layers"] = len(layer_selected)
        kv_source_mapping = list(layer_selected)
        logger.info(
            "layer_selected=%s, overriding num_layers=%d, kv_source_mapping=%s",
            layer_selected,
            len(layer_selected),
            kv_source_mapping,
        )
    elif kv_source_mode == "fused_mlp":
        # Build a runtime KVFusionModule that fuses all N video layers' K/V into
        # per-action-layer K/V via a learned MLP. Runtime routing is handled by
        # this module; the mapping below is only used for weight init selection.
        from .kv_fusion import KVFusionModule

        num_video_layers = int(video_dit_config.get("num_layers", 30))
        num_action_layers = int(action_dit_config.get("num_layers", 5))
        num_heads = int(video_dit_config.get("num_heads", 24))
        attn_head_dim = int(video_dit_config.get("attn_head_dim", 128))
        init_mode = "uniform_end" if num_action_layers <= num_video_layers else "final_only"
        kv_source_mapping = compute_kv_source_mapping(
            mode=init_mode,
            video_num_layers=num_video_layers,
            action_num_layers=num_action_layers,
        )
        kv_fusion = KVFusionModule(
            num_action_layers=num_action_layers,
            num_video_layers=num_video_layers,
            attn_hidden_dim=num_heads * attn_head_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            fusion_use_norm=fusion_use_norm,
            dtype=model_dtype,
        )
        logger.info(
            "KV source mode: fused_mlp (all %d video layers fused via MLP), "
            "fusion params: %.1fK",
            num_video_layers,
            sum(p.numel() for p in kv_fusion.parameters()) / 1e3,
        )
    else:
        kv_source_mapping = compute_kv_source_mapping(
            mode=kv_source_mode,
            video_num_layers=int(video_dit_config.get("num_layers", 30)),
            action_num_layers=int(action_dit_config.get("num_layers", 5)),
        )
        logger.info("KV source mode: %s, mapping: %s", kv_source_mode, kv_source_mapping)

    return FasterWAMDecoupled.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        decoupled=True,
        kv_source_mapping=kv_source_mapping,
        kv_source_mode=kv_source_mode,
        fixed_rope=bool(fixed_rope),
        new_fused_kv_rope_mode=str(new_fused_kv_rope_mode),
        aligned_3d_action_spatial_anchor_layout=(
            None
            if aligned_3d_action_spatial_anchor_layout is None
            else str(aligned_3d_action_spatial_anchor_layout)
        ),
        eef_calibration_path=eef_calibration_path,
        eef_raw_source_resolution=(
            None if eef_raw_source_resolution is None else int(eef_raw_source_resolution)
        ),
        eef_processed_video_size=eef_processed_video_size,
        new_fused_kv_projection_mode=new_fused_kv_projection_mode,
        new_fused_kv_pos_embed_max_tokens=int(new_fused_kv_pos_embed_max_tokens),
        new_fused_kv_pos_embed_dim=int(new_fused_kv_pos_embed_dim),
        new_fused_kv_mlp_mixer_num_blocks=int(new_fused_kv_mlp_mixer_num_blocks),
        new_fused_kv_mlp_mixer_token_mlp_ratio=float(new_fused_kv_mlp_mixer_token_mlp_ratio),
        new_fused_kv_mlp_mixer_channel_mlp_ratio=float(new_fused_kv_mlp_mixer_channel_mlp_ratio),
        new_fused_kv_head_fused_kv_sin2d_pe_mlp_hidden_dim=int(
            new_fused_kv_head_fused_kv_sin2d_pe_mlp_hidden_dim
        ),
        new_fused_kv_simple_head_softmax_fuse_mode=str(
            new_fused_kv_simple_head_softmax_fuse_mode
        ),
        new_fused_kv_head_fused_kv_fuse_mode=str(
            new_fused_kv_head_fused_kv_fuse_mode
        ),
        kv_fusion=kv_fusion,
    )
