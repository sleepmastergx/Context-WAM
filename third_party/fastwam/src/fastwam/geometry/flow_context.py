"""Observed-camera context construction for flow-coupled trajectory RoPE."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .calibration import TaskCalibration
from .eef_projector import EEFProjector, _reject_nonfinite, axis_angle_to_matrix, make_pose


def build_flow_trajectory_context(
    projector: EEFProjector,
    task_language: str | Sequence[str],
    eef_state: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build fixed t0 main/wrist camera matrices from a raw observed EEF pose."""
    state = np.asarray(eef_state, dtype=np.float64)
    if state.ndim == 0 or state.shape[-1] < 6:
        raise ValueError(
            "eef_state must have trailing dim >= 6 "
            f"(position 0:3, axis-angle 3:6), got shape {state.shape}"
        )
    unbatched = state.ndim == 1
    batch_shape = state.shape[:-1]
    flat = state.reshape(-1, state.shape[-1])
    _reject_nonfinite(
        flat[:, :6], "EEF pose (observation.state[0:6])", f"task={task_language!r}"
    )
    entries: list[TaskCalibration] = projector._resolve_entries(
        task_language, flat.shape[0]
    )
    position = flat[:, :3]
    world_from_eef = make_pose(position, axis_angle_to_matrix(flat[:, 3:6]))
    if any(entry.wrist_mount_T_eef_from_C is not None for entry in entries):
        mount = np.stack(
            [
                entry.wrist_mount_T_eef_from_C
                if entry.wrist_mount_T_eef_from_C is not None
                else projector.calibration.wrist_mount_T_eef_from_C
                for entry in entries
            ]
        )
    else:
        mount = projector.calibration.wrist_mount_T_eef_from_C
    world_from_wrist = world_from_eef @ mount
    world_from_main = np.stack([entry.agentview_T_W_from_C for entry in entries])
    poses = np.stack([world_from_main, world_from_wrist], axis=1)
    intrinsics = np.stack(
        [
            np.stack(
                [projector._intrinsics(entry.agentview_fovy_deg) for entry in entries]
            ),
            np.stack([projector._intrinsics(entry.wrist_fovy_deg) for entry in entries]),
        ],
        axis=1,
    )

    def _shape(array: np.ndarray) -> np.ndarray:
        target = array.shape[1:]
        return array.reshape(target) if unbatched else array.reshape(batch_shape + target)

    geometry = np.asarray(
        [
            projector.geometry.raw_width,
            projector.geometry.raw_height,
            projector.geometry.scale,
            projector.geometry.crop_left,
            projector.geometry.crop_top,
            projector.geometry.token_stride,
        ],
        dtype=np.float32,
    )
    if not unbatched:
        geometry = np.broadcast_to(geometry, batch_shape + geometry.shape).copy()
    return {
        "eef_position": _shape(position).astype(np.float32),
        "camera_world_from_camera": _shape(poses).astype(np.float32),
        "camera_intrinsics": _shape(intrinsics).astype(np.float32),
        "image_geometry": geometry,
    }
