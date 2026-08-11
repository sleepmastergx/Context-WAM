"""Opt-in RoboTwin v3 dataset context for dual-arm aligned_3dpft."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from fastwam.geometry.robotwin_flow import (
    RobotWinAlohaKinematics,
    build_robotwin_training_context,
)

from .lerobot.robot_video_dataset import RobotVideoDataset


class RobotWinAligned3DPFTRobotVideoDataset(RobotVideoDataset):
    """Attach observed cameras, qpos, and normalization to each RoboTwin sample."""

    def __init__(
        self,
        *args,
        robotwin_urdf_path: str,
        robotwin_root_pose: Sequence[float] = (0.0, -0.65, 0.0, 0.707, 0.0, 0.0, 0.707),
        robotwin_tcp_offset: float = 0.12,
        robotwin_source_size: Sequence[int] = (480, 640),
        robotwin_camera_fovy_deg: float = 37.0,
        robotwin_head_camera_position: Sequence[float] = (-0.032, -0.45, 1.35),
        robotwin_head_camera_forward: Sequence[float] = (0.0, 0.6, -0.8),
        robotwin_head_camera_left: Sequence[float] = (-1.0, 0.0, 0.0),
        robotwin_token_stride: float = 16.0,
        **kwargs,
    ) -> None:
        resolved_urdf = Path(robotwin_urdf_path).expanduser()
        if not resolved_urdf.is_absolute():
            resolved_urdf = Path.cwd() / resolved_urdf
        self.robotwin_kinematics = RobotWinAlohaKinematics(
            resolved_urdf,
            root_pose=robotwin_root_pose,
            tcp_offset=robotwin_tcp_offset,
        )
        self.robotwin_source_size = tuple(int(value) for value in robotwin_source_size)
        self.robotwin_camera_fovy_deg = float(robotwin_camera_fovy_deg)
        self.robotwin_head_camera_position = tuple(
            float(value) for value in robotwin_head_camera_position
        )
        self.robotwin_head_camera_forward = tuple(
            float(value) for value in robotwin_head_camera_forward
        )
        self.robotwin_head_camera_left = tuple(
            float(value) for value in robotwin_head_camera_left
        )
        self.robotwin_token_stride = float(robotwin_token_stride)
        super().__init__(*args, **kwargs)
        if self.concat_multi_camera != "robotwin":
            raise ValueError("RoboTwin aligned_3dpft requires concat_multi_camera=robotwin")
        if tuple(self.video_size) != (384, 320):
            raise ValueError("RoboTwin aligned_3dpft requires final video_size=[384,320]")

    @staticmethod
    def _merged_normalization(processor, field: str) -> tuple[torch.Tensor, torch.Tensor]:
        scales = []
        offsets = []
        for meta in processor.shape_meta[field]:
            normalizer = processor.normalizer.normalizers[field][meta["key"]]
            scales.append(torch.as_tensor(normalizer.scale, dtype=torch.float32))
            offsets.append(torch.as_tensor(normalizer.offset, dtype=torch.float32))
        return torch.cat(scales, dim=-1), torch.cat(offsets, dim=-1)

    def _get(self, idx):
        data = super()._get(idx)
        processor = self.lerobot_dataset.processor
        if processor is None:
            raise RuntimeError("RoboTwin aligned_3dpft requires a configured processor")
        state_scale, state_offset = self._merged_normalization(processor, "state")
        normalized_state = torch.as_tensor(data["proprio"][0], dtype=torch.float32)
        if normalized_state.numel() != 14:
            raise ValueError(
                "RoboTwin aligned_3dpft expects 14D observed qpos, got "
                f"{normalized_state.numel()}"
            )
        if bool((state_scale == 0).any()):
            raise ValueError("RoboTwin state normalization scale contains zero")
        raw_qpos = (normalized_state - state_offset) / state_scale
        context = build_robotwin_training_context(
            self.robotwin_kinematics,
            raw_qpos,
            source_size=self.robotwin_source_size,
            fovy_deg=self.robotwin_camera_fovy_deg,
            head_position=self.robotwin_head_camera_position,
            head_forward=self.robotwin_head_camera_forward,
            head_left=self.robotwin_head_camera_left,
            token_stride=self.robotwin_token_stride,
        )
        action_scale, action_offset = self._merged_normalization(processor, "action")
        context["action_scale"] = action_scale.numpy()
        context["action_offset"] = action_offset.numpy()
        data["aligned_3dpft_context"] = {
            key: torch.from_numpy(value.copy()) for key, value in context.items()
        }
        return data
