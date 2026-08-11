"""RoboTwin dual-arm aligned_3dpft integration for FasterWAM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from fastwam.geometry.robotwin_flow import (
    RobotWinAlohaKinematics,
    build_robotwin_aligned_3dpft_geometry,
)

from .fasterwam_decoupled import (
    FasterWAMDecoupled,
    MoTDecoupledActionAlignedVideoRoPE,
    create_fasterwam_decoupled,
)


class MoTRobotWinAligned3DPFT(MoTDecoupledActionAlignedVideoRoPE):
    """Dynamic four-group dual-arm trajectory RoPE over the RoboTwin mosaic."""

    supported_new_fused_kv_rope_modes = (
        MoTDecoupledActionAlignedVideoRoPE.supported_new_fused_kv_rope_modes
        | {"aligned_3dpft"}
    )
    _aligned_3d_family_rope_modes = frozenset(
        {*MoTDecoupledActionAlignedVideoRoPE._aligned_3d_family_rope_modes, "aligned_3dpft"}
    )

    def set_flow_trajectory_context(
        self, context: Optional[Mapping[str, torch.Tensor]]
    ) -> None:
        self._aligned_3dpft_context = context

    def prepare_flow_trajectory(self, noisy_action: torch.Tensor) -> None:
        context = getattr(self, "_aligned_3dpft_context", None)
        if context is None:
            raise ValueError(
                "RoboTwin aligned_3dpft requires context for every train/infer call"
            )
        kinematics = getattr(self, "_robotwin_kinematics", None)
        if kinematics is None:
            raise RuntimeError("RoboTwin aligned_3dpft kinematics was not initialized")
        anchors, visible = build_robotwin_aligned_3dpft_geometry(
            noisy_action, context, kinematics
        )
        self._aligned_3dpft_anchors = anchors
        self._aligned_3dpft_visible = visible

    def clear_flow_trajectory(self) -> None:
        self._aligned_3dpft_context = None
        self._aligned_3dpft_anchors = None
        self._aligned_3dpft_visible = None

    def _robotwin_head_anchor_indices(self, device: torch.device) -> torch.Tensor:
        if self.num_heads % 4 != 0:
            raise ValueError(
                "RoboTwin aligned_3dpft requires num_heads divisible by four, "
                f"got {self.num_heads}"
            )
        per_group = self.num_heads // 4
        return torch.arange(4, device=device, dtype=torch.long).repeat_interleave(
            per_group
        )

    def _build_new_fused_kv_3d_rope_freqs(
        self,
        video_freqs: torch.Tensor,
        action_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_rope, static_action_rope = super()._build_new_fused_kv_3d_rope_freqs(
            video_freqs, action_seq_len
        )
        if not bool(getattr(self, "_aligned_3dpft_active", False)):
            return video_rope, static_action_rope

        anchors = getattr(self, "_aligned_3dpft_anchors", None)
        if anchors is None:
            raise ValueError("RoboTwin aligned_3dpft action hook did not produce anchors")
        if anchors.ndim != 4 or anchors.shape[1:] != (action_seq_len, 4, 2):
            raise ValueError(
                "RoboTwin aligned_3dpft anchors must be [B,Sa,4,2], got "
                f"{tuple(anchors.shape)}"
            )

        t_dim, h_dim, _ = self._split_3d_rope_dims()
        tokens_per_frame = self._infer_video_tokens_per_frame(video_freqs)
        height, width = self._infer_video_spatial_grid_size(
            video_freqs, tokens_per_frame
        )
        if height <= 1 or width <= 1:
            raise ValueError(
                f"RoboTwin aligned_3dpft requires a spatial grid, got {(height, width)}"
            )
        h_unit = video_freqs[width, 0, t_dim:t_dim + h_dim]
        w_unit = video_freqs[1, 0, t_dim + h_dim:]

        anchors = anchors.to(device=video_freqs.device, dtype=torch.float32)
        head_anchor = self._robotwin_head_anchor_indices(anchors.device)
        gather_index = head_anchor.view(1, 1, -1, 1).expand(
            anchors.shape[0], action_seq_len, -1, 2
        )
        head_coordinates = torch.gather(anchors, 2, gather_index)
        h_angles = head_coordinates[..., 0, None] * torch.angle(h_unit).view(
            1, 1, 1, -1
        )
        w_angles = head_coordinates[..., 1, None] * torch.angle(w_unit).view(
            1, 1, 1, -1
        )
        spatial = torch.cat(
            [
                torch.polar(torch.ones_like(h_angles), h_angles),
                torch.polar(torch.ones_like(w_angles), w_angles),
            ],
            dim=-1,
        )
        temporal = static_action_rope[..., :t_dim].unsqueeze(0).expand(
            anchors.shape[0], -1, -1, -1
        )
        return video_rope, torch.cat([temporal, spatial], dim=-1)

    def _build_aligned_3dp_per_head_action_mask(
        self,
        base_action_mask: torch.Tensor,
        video_freqs: torch.Tensor,
    ) -> torch.Tensor:
        # The established robotwin aligned_3dp mask already allocates heads as
        # 12 main, 6 left wrist, 6 right wrist. The four dynamic anchor groups
        # split those 12 main heads into left/right trajectories without changing
        # their visual region.
        mask = super()._build_aligned_3dp_per_head_action_mask(
            base_action_mask, video_freqs
        )
        if not bool(getattr(self, "_aligned_3dpft_active", False)):
            return mask
        visible = getattr(self, "_aligned_3dpft_visible", None)
        if visible is None:
            raise ValueError("RoboTwin aligned_3dpft has no visibility mask")
        head_anchor = self._robotwin_head_anchor_indices(visible.device)
        head_visible = visible.index_select(2, head_anchor).permute(0, 2, 1)
        video_seq_len = mask.shape[-1] - mask.shape[-2]
        dynamic = mask.unsqueeze(0).expand(visible.shape[0], -1, -1, -1).clone()
        dynamic[..., :video_seq_len] &= head_visible.unsqueeze(-1)
        return dynamic[0] if dynamic.shape[0] == 1 else dynamic

    def _run_action_layers_with_video_kv_source(self, *args: Any, **kwargs: Any):
        if self.new_fused_kv_rope_mode != "aligned_3dpft":
            return super()._run_action_layers_with_video_kv_source(*args, **kwargs)
        self._aligned_3dpft_active = True
        self.new_fused_kv_rope_mode = "aligned_3dp"
        try:
            return super()._run_action_layers_with_video_kv_source(*args, **kwargs)
        finally:
            self.new_fused_kv_rope_mode = "aligned_3dpft"
            self._aligned_3dpft_active = False


class RobotWinAligned3DPFTFasterWAMDecoupled(FasterWAMDecoupled):
    """FasterWAM whose noisy dual-arm qpos drives RoboTwin action RoPE."""

    def _install_aligned_3dpft_hook(self) -> None:
        if getattr(self, "_aligned_3dpft_hook_handle", None) is not None:
            return

        def _capture_noisy_action(_module, args):
            if not args:
                raise RuntimeError("ActionDiT action_encoder hook received no input")
            self.mot.prepare_flow_trajectory(args[0])

        self._aligned_3dpft_hook_handle = self.action_expert.action_encoder.register_forward_pre_hook(
            _capture_noisy_action
        )

    def _set_flow_context(self, context: Optional[Mapping[str, torch.Tensor]]) -> None:
        self.mot.set_flow_trajectory_context(context)

    def training_loss(self, sample: dict, tiled: bool = False):
        self._set_flow_context(sample.get("aligned_3dpft_context"))
        try:
            return super().training_loss(sample, tiled=tiled)
        finally:
            self.mot.clear_flow_trajectory()

    @torch.no_grad()
    def infer_action(
        self,
        *args: Any,
        aligned_3dpft_context: Optional[Mapping[str, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._set_flow_context(aligned_3dpft_context)
        try:
            return super().infer_action(*args, **kwargs)
        finally:
            self.mot.clear_flow_trajectory()


def create_robotwin_aligned_3dpft(
    *,
    new_fused_kv_rope_mode: str = "aligned_3dpft",
    robotwin_urdf_path: str,
    robotwin_root_pose: Sequence[float] = (0.0, -0.65, 0.0, 0.707, 0.0, 0.0, 0.707),
    robotwin_tcp_offset: float = 0.12,
    **kwargs: Any,
) -> RobotWinAligned3DPFTFasterWAMDecoupled:
    """Build a checkpoint-compatible opt-in RoboTwin aligned_3dpft model."""
    if new_fused_kv_rope_mode != "aligned_3dpft":
        raise ValueError("RoboTwin factory only accepts aligned_3dpft RoPE")
    if kwargs.get("kv_source_mode") != "new_fused_kv":
        raise ValueError("RoboTwin aligned_3dpft requires kv_source_mode=new_fused_kv")
    if kwargs.get("new_fused_kv_projection_mode") != "simple_head_softmax":
        raise ValueError(
            "RoboTwin aligned_3dpft requires simple_head_softmax projection"
        )
    if str(kwargs.get("aligned_3d_action_spatial_anchor_layout")) != "robotwin":
        raise ValueError("RoboTwin aligned_3dpft requires layout=robotwin")
    action_dim = int(kwargs.get("action_dit_config", {}).get("action_dim", 0))
    if action_dim != 14:
        raise ValueError(f"RoboTwin aligned_3dpft requires action_dim=14, got {action_dim}")

    model = create_fasterwam_decoupled(
        new_fused_kv_rope_mode="aligned_3dp",
        **kwargs,
    )
    resolved_urdf = Path(robotwin_urdf_path).expanduser()
    if not resolved_urdf.is_absolute():
        resolved_urdf = Path.cwd() / resolved_urdf
    kinematics = RobotWinAlohaKinematics(
        resolved_urdf,
        root_pose=robotwin_root_pose,
        tcp_offset=robotwin_tcp_offset,
    )
    model.mot.__class__ = MoTRobotWinAligned3DPFT
    model.mot.new_fused_kv_rope_mode = "aligned_3dpft"
    model.mot._robotwin_kinematics = kinematics
    model.mot._aligned_3dpft_context = None
    model.mot._aligned_3dpft_anchors = None
    model.mot._aligned_3dpft_visible = None
    model.__class__ = RobotWinAligned3DPFTFasterWAMDecoupled
    model.robotwin_urdf_path = str(resolved_urdf.resolve())
    model.robotwin_root_pose = tuple(float(value) for value in robotwin_root_pose)
    model.robotwin_tcp_offset = float(robotwin_tcp_offset)
    model._aligned_3dpft_hook_handle = None
    model._install_aligned_3dpft_hook()
    return model
