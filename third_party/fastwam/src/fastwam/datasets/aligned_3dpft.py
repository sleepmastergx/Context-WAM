"""Opt-in LIBERO dataset context for ``aligned_3dpft``."""

from __future__ import annotations

import torch

from fastwam.geometry.flow_context import build_flow_trajectory_context

from .lerobot.robot_video_dataset import DEFAULT_PROMPT, RobotVideoDataset


class Aligned3DPFTRobotVideoDataset(RobotVideoDataset):
    """Attach observed cameras and normalization affine to each training sample."""

    def __init__(self, *args, **kwargs):
        if not kwargs.get("eef_anchor_calibration_path"):
            raise ValueError(
                "Aligned3DPFTRobotVideoDataset requires eef_anchor_calibration_path"
            )
        super().__init__(*args, **kwargs)

    @staticmethod
    def _merged_normalization(processor, field: str) -> tuple[torch.Tensor, torch.Tensor]:
        meta_entries = processor.shape_meta[field]
        scales = []
        offsets = []
        for meta in meta_entries:
            normalizer = processor.normalizer.normalizers[field][meta["key"]]
            scales.append(torch.as_tensor(normalizer.scale, dtype=torch.float32))
            offsets.append(torch.as_tensor(normalizer.offset, dtype=torch.float32))
        return torch.cat(scales, dim=-1), torch.cat(offsets, dim=-1)

    def _get(self, idx):
        data = super()._get(idx)
        if self.precompute_video_only:
            return data
        if self.eef_anchor_index is None:
            raise RuntimeError("aligned_3dpft EEF projector was not initialized")
        processor = self.lerobot_dataset.processor
        if processor is None:
            raise RuntimeError("aligned_3dpft requires a configured processor")

        state_scale, state_offset = self._merged_normalization(processor, "state")
        normalized_state = torch.as_tensor(data["proprio"][0], dtype=torch.float32)
        raw_state = (normalized_state[: state_scale.numel()] - state_offset) / state_scale
        if raw_state.numel() < 6:
            raise ValueError("aligned_3dpft requires at least six EEF state dimensions")

        prefix = DEFAULT_PROMPT.split("{task}", 1)[0]
        prompt = str(data["prompt"])
        if not prompt.startswith(prefix):
            raise ValueError(f"unexpected training prompt format: {prompt!r}")
        task_language = prompt[len(prefix):]
        context = build_flow_trajectory_context(
            self.eef_anchor_index.projector,
            task_language,
            raw_state[:6].numpy(),
        )
        action_scale, action_offset = self._merged_normalization(processor, "action")
        context["action_scale"] = action_scale.numpy()
        context["action_offset"] = action_offset.numpy()
        data["aligned_3dpft_context"] = {
            key: torch.from_numpy(value.copy()) for key, value in context.items()
        }
        return data
