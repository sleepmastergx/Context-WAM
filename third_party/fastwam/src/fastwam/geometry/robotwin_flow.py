"""RoboTwin dual-arm FK and observed-camera projection for aligned_3dpft."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import torch


ROBOTWIN_CONTEXT_KEYS = (
    "observed_qpos",
    "camera_from_world",
    "camera_intrinsics",
    "mosaic_affine",
    "action_scale",
    "action_offset",
)


def _parse_vector(text: str | None, default: Sequence[float]) -> tuple[float, ...]:
    if text is None:
        return tuple(float(value) for value in default)
    values = tuple(float(value) for value in text.split())
    if len(values) != len(default):
        raise ValueError(f"expected {len(default)} values, got {values}")
    return values


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _pose_matrix(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _rpy_matrix(rpy)
    matrix[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return matrix


def _quat_wxyz_matrix(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        raise ValueError("zero-norm root quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _root_pose_matrix(pose: Sequence[float]) -> np.ndarray:
    if len(pose) != 7:
        raise ValueError(f"RoboTwin root pose must be [x,y,z,qw,qx,qy,qz], got {pose}")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_wxyz_matrix(pose[3:])
    matrix[:3, 3] = np.asarray(pose[:3], dtype=np.float64)
    return matrix


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class RobotWinAlohaKinematics:
    """Small torch FK implementation for the RoboTwin ALOHA front arms."""

    left_joint_names = tuple(f"fl_joint{index}" for index in range(1, 7))
    right_joint_names = tuple(f"fr_joint{index}" for index in range(1, 7))
    left_action_indices = (0, 1, 2, 3, 4, 5)
    right_action_indices = (7, 8, 9, 10, 11, 12)

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        root_pose: Sequence[float] = (0.0, -0.65, 0.0, 0.707, 0.0, 0.0, 0.707),
        tcp_offset: float = 0.12,
    ) -> None:
        path = Path(urdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RoboTwin URDF not found: {path}")
        root = ET.parse(path).getroot()
        joints: dict[str, _Joint] = {}
        child_to_joint: dict[str, _Joint] = {}
        for element in root.findall("joint"):
            parent_element = element.find("parent")
            child_element = element.find("child")
            if parent_element is None or child_element is None:
                continue
            origin_element = element.find("origin")
            axis_element = element.find("axis")
            xyz = _parse_vector(
                None if origin_element is None else origin_element.get("xyz"),
                (0.0, 0.0, 0.0),
            )
            rpy = _parse_vector(
                None if origin_element is None else origin_element.get("rpy"),
                (0.0, 0.0, 0.0),
            )
            axis = np.asarray(
                _parse_vector(
                    None if axis_element is None else axis_element.get("xyz"),
                    (1.0, 0.0, 0.0),
                ),
                dtype=np.float64,
            )
            joint = _Joint(
                name=str(element.get("name")),
                kind=str(element.get("type", "fixed")),
                parent=str(parent_element.get("link")),
                child=str(child_element.get("link")),
                origin=_pose_matrix(xyz, rpy),
                axis=axis,
            )
            joints[joint.name] = joint
            child_to_joint[joint.child] = joint

        self.urdf_path = str(path)
        self._joints = joints
        self._child_to_joint = child_to_joint
        self._root_pose = _root_pose_matrix(root_pose)
        self.tcp_offset = float(tcp_offset)
        self._joint_action_index = {
            **dict(zip(self.left_joint_names, self.left_action_indices)),
            **dict(zip(self.right_joint_names, self.right_action_indices)),
        }
        self._chains = {
            target: self._chain_to(target)
            for target in ("fl_link6", "fr_link6", "left_camera", "right_camera")
        }

    def _chain_to(self, target: str) -> tuple[_Joint, ...]:
        chain = []
        child = target
        seen = set()
        while child in self._child_to_joint:
            if child in seen:
                raise ValueError(f"cycle in URDF chain ending at {target!r}")
            seen.add(child)
            joint = self._child_to_joint[child]
            chain.append(joint)
            child = joint.parent
        chain.reverse()
        if not chain:
            raise ValueError(f"URDF has no chain to {target!r}")
        return tuple(chain)

    @staticmethod
    def _axis_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        axis = axis / torch.linalg.vector_norm(axis)
        x, y, z = axis.unbind()
        zero = torch.zeros_like(x)
        skew = torch.stack(
            [zero, -z, y, z, zero, -x, -y, x, zero], dim=0
        ).reshape(3, 3)
        eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
        outer = axis[:, None] * axis[None, :]
        cos = torch.cos(angle)[..., None, None]
        sin = torch.sin(angle)[..., None, None]
        return cos * eye + (1.0 - cos) * outer + sin * skew

    def link_pose(self, qpos: torch.Tensor, target: str) -> torch.Tensor:
        if qpos.shape[-1] != 14:
            raise ValueError(f"RoboTwin qpos must end in 14 dimensions, got {tuple(qpos.shape)}")
        if target not in self._chains:
            raise ValueError(f"unsupported RoboTwin FK target {target!r}")
        qpos = torch.as_tensor(qpos)
        leading = qpos.shape[:-1]
        root = torch.as_tensor(self._root_pose, device=qpos.device, dtype=qpos.dtype)
        transform = root.expand(leading + (4, 4)).clone()
        for joint in self._chains[target]:
            origin = torch.as_tensor(joint.origin, device=qpos.device, dtype=qpos.dtype)
            transform = transform @ origin
            if joint.kind in {"revolute", "continuous"}:
                if joint.name not in self._joint_action_index:
                    raise ValueError(f"no action index for moving joint {joint.name!r}")
                angle = qpos[..., self._joint_action_index[joint.name]]
                axis = torch.as_tensor(joint.axis, device=qpos.device, dtype=qpos.dtype)
                rotation = self._axis_rotation(axis, angle)
                motion = torch.eye(4, device=qpos.device, dtype=qpos.dtype).expand(
                    leading + (4, 4)
                ).clone()
                motion[..., :3, :3] = rotation
                transform = transform @ motion
            elif joint.kind != "fixed":
                raise ValueError(
                    f"unsupported joint type {joint.kind!r} in chain to {target!r}"
                )
        return transform

    def tcp_positions(self, qpos: torch.Tensor) -> torch.Tensor:
        qpos = torch.as_tensor(qpos)
        point = torch.tensor(
            [self.tcp_offset, 0.0, 0.0, 1.0], device=qpos.device, dtype=qpos.dtype
        )
        left = (self.link_pose(qpos, "fl_link6") @ point)[..., :3]
        right = (self.link_pose(qpos, "fr_link6") @ point)[..., :3]
        return torch.stack([left, right], dim=-2)

    def wrist_camera_world_poses(self, qpos: torch.Tensor) -> torch.Tensor:
        left = self.link_pose(qpos, "left_camera")
        right = self.link_pose(qpos, "right_camera")
        return torch.stack([left, right], dim=-3)


def sapien_world_pose_to_cv_extrinsic(world_from_camera: torch.Tensor) -> torch.Tensor:
    """Convert SAPIEN's x-forward/y-left/z-up camera pose to OpenCV extrinsics."""
    pose = torch.as_tensor(world_from_camera)
    sapien_to_cv = torch.tensor(
        [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        device=pose.device,
        dtype=pose.dtype,
    )
    return sapien_to_cv @ torch.linalg.inv(pose)


def intrinsics_from_fovy(
    fovy_deg: float, width: int, height: int, *, dtype: np.dtype = np.float32
) -> np.ndarray:
    focal = 0.5 * float(height) / np.tan(float(fovy_deg) * np.pi / 360.0)
    return np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )


def _head_world_pose(
    position: Sequence[float],
    forward: Sequence[float],
    left: Sequence[float],
) -> np.ndarray:
    forward_vector = np.asarray(forward, dtype=np.float64)
    forward_vector /= np.linalg.norm(forward_vector)
    left_vector = np.asarray(left, dtype=np.float64)
    left_vector /= np.linalg.norm(left_vector)
    up_vector = np.cross(forward_vector, left_vector)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.stack([forward_vector, left_vector, up_vector], axis=1)
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def robotwin_mosaic_affine(
    source_sizes: Sequence[Sequence[int]],
    *,
    token_stride: float = 16.0,
) -> np.ndarray:
    """Return per-camera [scale_x, scale_y, offset_x, offset_y, stride]."""
    if len(source_sizes) != 3:
        raise ValueError(f"RoboTwin requires three source sizes, got {source_sizes}")
    targets = ((256, 320, 0, 0), (128, 160, 0, 256), (128, 160, 160, 256))
    affine = []
    for source, target in zip(source_sizes, targets):
        source_h, source_w = (int(value) for value in source)
        target_h, target_w, offset_x, offset_y = target
        affine.append(
            [
                target_w / source_w,
                target_h / source_h,
                float(offset_x),
                float(offset_y),
                float(token_stride),
            ]
        )
    return np.asarray(affine, dtype=np.float32)


def build_robotwin_training_context(
    kinematics: RobotWinAlohaKinematics,
    observed_qpos: np.ndarray | torch.Tensor,
    *,
    source_size: Sequence[int] = (480, 640),
    fovy_deg: float = 37.0,
    head_position: Sequence[float] = (-0.032, -0.45, 1.35),
    head_forward: Sequence[float] = (0.0, 0.6, -0.8),
    head_left: Sequence[float] = (-1.0, 0.0, 0.0),
    token_stride: float = 16.0,
) -> dict[str, np.ndarray]:
    qpos = torch.as_tensor(observed_qpos, dtype=torch.float64)
    unbatched = qpos.ndim == 1
    if unbatched:
        qpos = qpos.unsqueeze(0)
    if qpos.ndim != 2 or qpos.shape[-1] != 14:
        raise ValueError(f"observed RoboTwin qpos must be [14] or [B,14], got {tuple(qpos.shape)}")
    wrist_world = kinematics.wrist_camera_world_poses(qpos)
    head_world = torch.as_tensor(
        _head_world_pose(head_position, head_forward, head_left), dtype=qpos.dtype
    ).expand(qpos.shape[0], 4, 4)
    world_poses = torch.cat([head_world[:, None], wrist_world], dim=1)
    camera_from_world = sapien_world_pose_to_cv_extrinsic(world_poses)

    source_h, source_w = (int(value) for value in source_size)
    intrinsics = np.stack(
        [intrinsics_from_fovy(fovy_deg, source_w, source_h) for _ in range(3)]
    )
    intrinsics = np.broadcast_to(intrinsics, (qpos.shape[0], 3, 3, 3)).copy()
    affine = robotwin_mosaic_affine([source_size] * 3, token_stride=token_stride)
    affine = np.broadcast_to(affine, (qpos.shape[0], 3, 5)).copy()
    result = {
        "observed_qpos": qpos.numpy().astype(np.float32),
        "camera_from_world": camera_from_world.numpy().astype(np.float32),
        "camera_intrinsics": intrinsics.astype(np.float32),
        "mosaic_affine": affine.astype(np.float32),
    }
    if unbatched:
        return {key: value[0] for key, value in result.items()}
    return result


def _batched(value: torch.Tensor, rank: int, name: str) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.ndim == rank - 1:
        value = value.unsqueeze(0)
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank - 1} or {rank}, got {tuple(value.shape)}")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"RoboTwin aligned_3dpft received nonfinite {name}")


def _expand_affine(
    value: torch.Tensor, batch: int, horizon: int, action_dim: int, name: str
) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.ndim == 1:
        value = value.view(1, 1, -1)
    elif value.ndim == 2:
        value = value.unsqueeze(1)
    if value.ndim != 3 or value.shape[-1] != action_dim:
        raise ValueError(f"{name} cannot broadcast to [B,S,D]={batch,horizon,action_dim}")
    return value.expand(batch, horizon, action_dim)


def _project_all(
    points_world: torch.Tensor,
    camera_from_world: torch.Tensor,
    intrinsics: torch.Tensor,
    mosaic_affine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    homogeneous = torch.cat([points_world, torch.ones_like(points_world[..., :1])], dim=-1)
    camera_points = torch.einsum("bcij,bsej->bseci", camera_from_world, homogeneous)[..., :3]
    depth = camera_points[..., 2]
    visible = depth > 0
    divisor = torch.where(visible, depth, torch.ones_like(depth))
    u = intrinsics[:, None, None, :, 0, 0] * camera_points[..., 0] / divisor
    u = u + intrinsics[:, None, None, :, 0, 2]
    v = intrinsics[:, None, None, :, 1, 1] * camera_points[..., 1] / divisor
    v = v + intrinsics[:, None, None, :, 1, 2]
    affine = mosaic_affine[:, None, None]
    x_pixel = (u + 0.5) * affine[..., 0] - 0.5 + affine[..., 2]
    y_pixel = (v + 0.5) * affine[..., 1] - 0.5 + affine[..., 3]
    x_token = (x_pixel + 0.5) / affine[..., 4] - 0.5
    y_token = (y_pixel + 0.5) / affine[..., 4] - 0.5
    anchors = torch.stack([y_token, x_token], dim=-1)
    _finite(anchors, "projected anchors")
    return anchors, visible


def build_robotwin_aligned_3dpft_geometry(
    noisy_action: torch.Tensor,
    context: Mapping[str, torch.Tensor],
    kinematics: RobotWinAlohaKinematics,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map current noisy absolute qpos to four dual-arm RoboTwin anchor groups."""
    missing = [key for key in ROBOTWIN_CONTEXT_KEYS if key not in context]
    if missing:
        raise ValueError(f"RoboTwin aligned_3dpft context is missing keys: {missing}")
    if noisy_action.ndim != 3 or noisy_action.shape[-1] != 14:
        raise ValueError(f"RoboTwin noisy action must be [B,S,14], got {tuple(noisy_action.shape)}")
    batch, horizon, action_dim = noisy_action.shape
    device = noisy_action.device
    dtype = torch.float32
    action = noisy_action.to(dtype=dtype)
    scale = _expand_affine(
        torch.as_tensor(context["action_scale"], device=device, dtype=dtype),
        batch,
        horizon,
        action_dim,
        "action_scale",
    )
    offset = _expand_affine(
        torch.as_tensor(context["action_offset"], device=device, dtype=dtype),
        batch,
        horizon,
        action_dim,
        "action_offset",
    )
    _finite(action, "noisy action")
    _finite(scale, "action scale")
    _finite(offset, "action offset")
    if bool((scale == 0).any()):
        raise ValueError("RoboTwin aligned_3dpft action scale contains zero")
    raw_qpos = (action - offset) / scale
    tcp = kinematics.tcp_positions(raw_qpos)

    observed_qpos = _batched(
        torch.as_tensor(context["observed_qpos"], device=device, dtype=dtype), 2, "observed_qpos"
    ).expand(batch, -1)
    camera_from_world = _batched(
        torch.as_tensor(context["camera_from_world"], device=device, dtype=dtype),
        4,
        "camera_from_world",
    ).expand(batch, -1, -1, -1)
    intrinsics = _batched(
        torch.as_tensor(context["camera_intrinsics"], device=device, dtype=dtype),
        4,
        "camera_intrinsics",
    ).expand(batch, -1, -1, -1)
    mosaic_affine = _batched(
        torch.as_tensor(context["mosaic_affine"], device=device, dtype=dtype),
        3,
        "mosaic_affine",
    ).expand(batch, -1, -1)
    for value, name in (
        (observed_qpos, "observed qpos"),
        (camera_from_world, "camera extrinsics"),
        (intrinsics, "camera intrinsics"),
        (mosaic_affine, "mosaic affine"),
    ):
        _finite(value, name)
    if camera_from_world.shape[1:] != (3, 4, 4):
        raise ValueError("RoboTwin camera_from_world must be [B,3,4,4]")
    if intrinsics.shape[1:] != (3, 3, 3):
        raise ValueError("RoboTwin camera_intrinsics must be [B,3,3,3]")
    if mosaic_affine.shape[1:] != (3, 5):
        raise ValueError("RoboTwin mosaic_affine must be [B,3,5]")

    all_anchors, all_visible = _project_all(
        tcp, camera_from_world, intrinsics, mosaic_affine
    )
    # Four head groups: main-left, main-right, left-wrist, right-wrist.
    entity_index = torch.tensor([0, 1, 0, 1], device=device)
    camera_index = torch.tensor([0, 0, 1, 2], device=device)
    anchors = all_anchors[:, :, entity_index, camera_index]
    visible = all_visible[:, :, entity_index, camera_index]

    observed_tcp = kinematics.tcp_positions(observed_qpos)[:, None]
    origin_all, origin_visible_all = _project_all(
        observed_tcp, camera_from_world, intrinsics, mosaic_affine
    )
    origin_anchors = origin_all[:, 0, entity_index, camera_index]
    origin_visible = origin_visible_all[:, 0, entity_index, camera_index]
    if not bool(origin_visible.all()):
        bad = (~origin_visible).nonzero(as_tuple=False).tolist()
        raise ValueError(
            "RoboTwin observed TCP origin has non-positive depth for anchor groups "
            f"{bad}"
        )
    last = origin_anchors
    held = []
    for step in range(horizon):
        last = torch.where(visible[:, step, :, None], anchors[:, step], last)
        held.append(last)
    return torch.stack(held, dim=1), visible
