"""Flow-coupled action trajectory projection for ``aligned_3dpft`` RoPE."""

from __future__ import annotations

from typing import Mapping

import torch


REQUIRED_CONTEXT_KEYS = (
    "eef_position",
    "camera_world_from_camera",
    "camera_intrinsics",
    "image_geometry",
    "action_scale",
    "action_offset",
)


def _batched(value: torch.Tensor, rank: int, name: str) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.ndim == rank - 1:
        value = value.unsqueeze(0)
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank - 1} or {rank}, got {tuple(value.shape)}")
    return value


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"aligned_3dpft received nonfinite {name}")


def _expand_action_affine(
    value: torch.Tensor,
    *,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    name: str,
) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.ndim == 1:
        value = value.view(1, 1, -1)
    elif value.ndim == 2:
        if value.shape == (action_horizon, action_dim):
            value = value.unsqueeze(0)
        elif value.shape[-1] == action_dim:
            value = value.unsqueeze(1)
        else:
            raise ValueError(f"{name} has incompatible shape {tuple(value.shape)}")
    if value.ndim != 3 or value.shape[-1] != action_dim:
        raise ValueError(
            f"{name} must broadcast to [B, Sa, D]={batch_size, action_horizon, action_dim}, "
            f"got {tuple(value.shape)}"
        )
    try:
        return value.expand(batch_size, action_horizon, action_dim)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} cannot broadcast to [B, Sa, D]={batch_size, action_horizon, action_dim}"
        ) from exc


def _project_points(
    points_world: torch.Tensor,
    camera_world_from_camera: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    image_geometry: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project ``[B, S, 3]`` points into camera-local ``(y, x)`` token units."""
    batch_size, sequence_length, _ = points_world.shape
    num_cameras = camera_world_from_camera.shape[1]
    camera_from_world = torch.linalg.inv(camera_world_from_camera)
    homogeneous = torch.cat(
        [points_world, torch.ones_like(points_world[..., :1])], dim=-1
    )
    camera_point = torch.einsum(
        "bcij,bsj->bsci", camera_from_world, homogeneous
    )[..., :3]
    _require_finite(camera_point, "camera-frame trajectory")

    depth = camera_point[..., 2]
    positive_depth = depth > 0
    # Invalid depths never enter the pinhole division. Their coordinates are
    # replaced by the causal hold below and their visual keys are masked.
    divisor = torch.where(positive_depth, depth, torch.ones_like(depth))
    u = (
        camera_intrinsics[:, None, :, 0, 0] * camera_point[..., 0] / divisor
        + camera_intrinsics[:, None, :, 0, 2]
    )
    v = (
        camera_intrinsics[:, None, :, 1, 1] * camera_point[..., 1] / divisor
        + camera_intrinsics[:, None, :, 1, 2]
    )

    geometry = image_geometry[:, None, None, :]
    raw_width = geometry[..., 0]
    resize_scale = geometry[..., 2]
    crop_left = geometry[..., 3]
    crop_top = geometry[..., 4]
    token_stride = geometry[..., 5]
    u_consumed = raw_width - 1.0 - u
    u_processed = (u_consumed + 0.5) * resize_scale - 0.5 - crop_left
    v_processed = (v + 0.5) * resize_scale - 0.5 - crop_top
    x_token = (u_processed + 0.5) / token_stride - 0.5
    y_token = (v_processed + 0.5) / token_stride - 0.5
    anchors = torch.stack([y_token, x_token], dim=-1)
    _require_finite(anchors, "projected trajectory anchors")
    if anchors.shape != (batch_size, sequence_length, num_cameras, 2):
        raise RuntimeError(f"unexpected aligned_3dpft anchor shape {tuple(anchors.shape)}")
    return anchors, positive_depth


def build_aligned_3dpft_geometry(
    noisy_action: torch.Tensor,
    context: Mapping[str, torch.Tensor],
    *,
    strict_batch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the current normalized flow state to dynamic camera anchors.

    Token ``j`` is the inclusive trajectory point
    ``eef_position + cumsum(denormalize(noisy_action)[..., :3])``. No token is
    assigned the observed EEF position itself.

    Args:
        strict_batch: Require each observed-geometry tensor to carry exactly
            ``B`` entries instead of allowing a single entry to expand across
            the batch. A server that packs independent requests into one
            forward pass must set this: silently broadcasting one request's
            camera pose over the other slots would denoise them against the
            wrong geometry and still return plausible actions.

    Returns:
        ``anchors`` as ``[B, Sa, C, 2]`` in camera-local ``(y, x)`` token units,
        and ``visible`` as ``[B, Sa, C]`` for strictly positive camera depth.
    """
    missing = [key for key in REQUIRED_CONTEXT_KEYS if key not in context]
    if missing:
        raise ValueError(f"aligned_3dpft context is missing keys: {missing}")
    if noisy_action.ndim != 3 or noisy_action.shape[-1] < 3:
        raise ValueError(
            f"noisy_action must be [B, Sa, D] with D >= 3, got {tuple(noisy_action.shape)}"
        )

    batch_size, action_horizon, action_dim = noisy_action.shape
    device = noisy_action.device
    geometry_dtype = torch.float32
    action = noisy_action.to(dtype=geometry_dtype)
    _require_finite(action, "noisy action")

    scale = _expand_action_affine(
        torch.as_tensor(context["action_scale"], device=device, dtype=geometry_dtype),
        batch_size=batch_size,
        action_horizon=action_horizon,
        action_dim=action_dim,
        name="action_scale",
    )
    offset = _expand_action_affine(
        torch.as_tensor(context["action_offset"], device=device, dtype=geometry_dtype),
        batch_size=batch_size,
        action_horizon=action_horizon,
        action_dim=action_dim,
        name="action_offset",
    )
    _require_finite(scale, "action normalization scale")
    _require_finite(offset, "action normalization offset")
    if bool((scale == 0).any()):
        raise ValueError("aligned_3dpft action normalization scale contains zero")
    raw_action = (action - offset) / scale

    eef_position = _batched(
        torch.as_tensor(context["eef_position"], device=device, dtype=geometry_dtype),
        2,
        "eef_position",
    )
    camera_poses = _batched(
        torch.as_tensor(
            context["camera_world_from_camera"], device=device, dtype=geometry_dtype
        ),
        4,
        "camera_world_from_camera",
    )
    intrinsics = _batched(
        torch.as_tensor(context["camera_intrinsics"], device=device, dtype=geometry_dtype),
        4,
        "camera_intrinsics",
    )
    image_geometry = _batched(
        torch.as_tensor(context["image_geometry"], device=device, dtype=geometry_dtype),
        2,
        "image_geometry",
    )
    for value, name in (
        (eef_position, "eef_position"),
        (camera_poses, "camera poses"),
        (intrinsics, "camera intrinsics"),
        (image_geometry, "image geometry"),
    ):
        _require_finite(value, name)
        if strict_batch:
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"{name} batch {value.shape[0]} must equal action batch "
                    f"{batch_size} under strict_batch; one request's observed "
                    "geometry must never be reused for another batch slot"
                )
        elif value.shape[0] not in (1, batch_size):
            raise ValueError(
                f"{name} batch {value.shape[0]} does not match action batch {batch_size}"
            )
    eef_position = eef_position.expand(batch_size, -1)
    camera_poses = camera_poses.expand(batch_size, -1, -1, -1)
    intrinsics = intrinsics.expand(batch_size, -1, -1, -1)
    image_geometry = image_geometry.expand(batch_size, -1)
    if camera_poses.shape[1] != 2 or intrinsics.shape[1] != 2:
        raise ValueError("aligned_3dpft currently requires exactly main and wrist cameras")
    if image_geometry.shape[1] != 6:
        raise ValueError(
            "image_geometry must contain [raw_w, raw_h, scale, crop_left, crop_top, token_stride]"
        )

    trajectory = eef_position[:, None, :] + torch.cumsum(raw_action[..., :3], dim=1)
    anchors, visible = _project_points(
        trajectory, camera_poses, intrinsics, image_geometry
    )

    origin_anchor, origin_visible = _project_points(
        eef_position[:, None, :], camera_poses, intrinsics, image_geometry
    )
    if not bool(origin_visible.all()):
        raise ValueError(
            "aligned_3dpft observed EEF origin must have positive depth in both cameras"
        )
    last = origin_anchor[:, 0]
    held = []
    for step in range(action_horizon):
        last = torch.where(visible[:, step, :, None], anchors[:, step], last)
        held.append(last)
    anchors = torch.stack(held, dim=1)
    return anchors, visible
