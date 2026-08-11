"""Shared EEF projector: raw world pose -> continuous camera-local token anchor.

Author: Rui Heng Yang

One implementation of the geometry, used by **both** offline training-time anchor
resolution and live inference. Two paths that agree only by inspection do not
agree; the anchor a checkpoint was trained against must be the anchor it is
evaluated against.
(Plan: /home/ruiheng/.claude/plans/fastwam/200731_RoPE_Anchor.md, Sections 5, 6.1, 7, 8, 14, 19.1)

Pipeline::

    observation.state (raw metric EEF pose, pre-normalization)
      -> T_C_from_W = inverse(T_W_from_C), continuous pinhole projection   6, 7
      -> horizontal flip ONLY:  u_img = W-1-u, v_img = v                   7.1
      -> resize / centre crop, half pixel:  dst = (src+0.5)*scale - 0.5    7.2
      -> continuous processed pixel -> camera-local token coordinate       7.3
      -> anchor (y, x), depth, on-screen flag

Three traps this module exists to hold fixed:

* **The flip is horizontal only.** An earlier revision specified a 180-degree
  rotation, which would have placed every main-camera anchor at the wrong row.
  The vertical component cancels: the raw MuJoCo render is already vertically
  flipped relative to the pinhole convention the camera matrices assume, and the
  capture code applies ``[::-1, ::-1]``, so the two vertical flips compose away
  and a net horizontal mirror remains (plan Section 7.1, finding F2). An
  orientation check using a point on either centre line cannot see the
  difference -- that is exactly how the error survived its first test.
* **Anchors are ordered (y, x)**, matching the model input contract
  ``eef_anchor_token [B, 2, 2]`` of plan Section 5.2. The diagnostic tooling's
  ``token_coords()`` returns ``(x, y)``. Swapping them is silent and puts every
  anchor on the wrong axis.
* **Raw render resolution differs by path**: 512 offline, 256 live. It enters
  only through the ``W-1`` flip term, so the wrong value biases the anchor by
  ``3.5/W`` token on both axes without any other symptom (plan Section 7.4).

Nothing here rounds or clips. A finite off-screen projection stays finite and
off-screen so the caller can classify it; only a nonfinite or non-positive-depth
projection is an error, and it is raised rather than substituted
(plan Section 19.1 Decision 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Sequence

import numpy as np

from .calibration import CalibrationTable, TaskCalibration

# --- fixed geometry constants (plan Sections 7.3, 7.4, 8, 19.1) -------------

#: Raw render resolution of the stored training videos (dataset meta/info.json).
TRAINING_RAW_RESOLUTION = 512
#: Raw render resolution of the live evaluator (LIBERO_ENV_RESOLUTION).
LIVE_RAW_RESOLUTION = 256

#: Per-camera processed image consumed by the model (configs/data/libero_2cam.yaml).
PROCESSED_WIDTH = 224
PROCESSED_HEIGHT = 224

#: Token lattice stride inside one processed camera image.
VAE_SPATIAL_FACTOR = 16
DIT_PATCH_SIZE = 2

#: Camera order is LOCKED: index 0 is main/agentview, index 1 is wrist/eye-in-hand,
#: matching the left-to-right composite layout (plan Section 2.2).
CAMERA_ORDER: tuple[str, str] = ("main", "wrist")

#: A camera-frame depth at or below this is at/behind the camera plane: invalid.
DEPTH_EPSILON = 1e-6

#: Version of the complete raw-state -> token-anchor transform contract.
EEF_PROJECTION_VERSION = 1

#: Image-centre fallback for a leading gap, in camera-local token units. Equal to
#: the `horizontal` preset `aligned_3d` already uses, so the fallback degrades to
#: today's fixed-anchor behaviour (plan Section 19.1 Decision 2). Derived, not
#: assumed: `centre_anchor_token()` reproduces it at raw 512 and raw 256.
CENTRE_ANCHOR_TOKEN: tuple[float, float] = (3.0, 3.0)

#: Rotation magnitude below which an axis-angle vector is treated as identity.
_AXIS_ANGLE_EPSILON = 1e-12

#: How many offending entries an invalid-projection message lists before eliding.
_MAX_REPORTED_INVALID = 5


class InvalidProjectionError(ValueError):
    """A projection was nonfinite or at/behind the camera plane.

    Never substituted, never downgraded to "off-screen". The distinction is
    load-bearing: an earlier projector tested only ``Z <= eps``, so a NaN depth
    (``NaN <= eps`` is False) produced NaN pixels that then failed the on-screen
    bounds test and were recorded as an ordinary out-of-frame gripper. A broken
    calculation was filed as expected data (plan Section 19.1 Decision 1).
    """


# --- image geometry ---------------------------------------------------------


@dataclass(frozen=True)
class ProcessedCameraGeometry:
    """Raw -> processed -> token geometry for one camera.

    Holds the resize/crop constants so the projector and any overlay audit read
    the same numbers instead of recomputing them from separate formulas
    (plan Section 7.2 requires one shared implementation).
    """

    raw_width: int
    raw_height: int
    processed_width: int = PROCESSED_WIDTH
    processed_height: int = PROCESSED_HEIGHT
    vae_spatial_factor: int = VAE_SPATIAL_FACTOR
    patch_size: int = DIT_PATCH_SIZE

    def __post_init__(self) -> None:
        for name in ("raw_width", "raw_height", "processed_width", "processed_height",
                     "vae_spatial_factor", "patch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        stride = self.token_stride
        # A processed image that is not a whole number of tokens wide has no
        # camera-local grid, so every token coordinate below would be a fiction.
        if self.processed_width % stride or self.processed_height % stride:
            raise ValueError(
                f"processed image {self.processed_width}x{self.processed_height} is not a whole "
                f"number of token strides ({stride}); the camera-local token grid is undefined"
            )

    @property
    def token_stride(self) -> int:
        """Processed pixels per visual token: VAE spatial factor x DiT patch size."""
        return self.vae_spatial_factor * self.patch_size

    @property
    def local_grid_width(self) -> int:
        """Token columns in one camera's processed image (7 for LIBERO)."""
        return self.processed_width // self.token_stride

    @property
    def local_grid_height(self) -> int:
        """Token rows in one camera's processed image (7 for LIBERO)."""
        return self.processed_height // self.token_stride

    @cached_property
    def scale(self) -> float:
        """Resize factor: ``max(W_t/W_s, H_t/H_s)``, matching production preprocessing."""
        return max(self.processed_width / self.raw_width, self.processed_height / self.raw_height)

    @cached_property
    def resized_width(self) -> int:
        """Width after resize and before centre crop."""
        return round(self.raw_width * self.scale)

    @cached_property
    def resized_height(self) -> int:
        """Height after resize and before centre crop."""
        return round(self.raw_height * self.scale)

    @cached_property
    def crop_left(self) -> int:
        """Centre-crop offset in resized columns (zero for square LIBERO input)."""
        return max((self.resized_width - self.processed_width) // 2, 0)

    @cached_property
    def crop_top(self) -> int:
        """Centre-crop offset in resized rows (zero for square LIBERO input)."""
        return max((self.resized_height - self.processed_height) // 2, 0)


def composite_token_grid(
    geometry: ProcessedCameraGeometry, num_cameras: int = len(CAMERA_ORDER)
) -> tuple[int, int]:
    """Composite ``(grid_h, grid_w)`` for ``num_cameras`` concatenated horizontally.

    Args:
        geometry: Per-camera processed geometry.
        num_cameras: Cameras concatenated along the width axis.

    Returns:
        ``(grid_h, grid_w)``; ``(7, 14)`` for the resolved LIBERO two-camera setup.

    Raises:
        ValueError: If ``num_cameras`` is not positive.
    """
    if num_cameras <= 0:
        raise ValueError(f"num_cameras must be positive, got {num_cameras}")
    return geometry.local_grid_height, geometry.local_grid_width * num_cameras


# --- primitive transforms ---------------------------------------------------


def intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics for a MuJoCo camera, at a given raw render size.

    Mirrors robosuite's ``get_camera_intrinsic_matrix()``: the focal length comes
    from the vertical field of view and the raw render *height*, and the
    principal point is ``(width/2, height/2)``. ``fovy`` is resolution
    independent, so intrinsics are always derived, never stored
    (plan Section 6.1).

    Args:
        fovy_deg: Vertical field of view in degrees.
        width: Raw render width in pixels.
        height: Raw render height in pixels.

    Returns:
        ``[3, 3]`` float64 intrinsic matrix.
    """
    focal = 0.5 * height / np.tan(fovy_deg * np.pi / 360.0)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation from an axis-angle vector, vectorized over leading dims.

    ``observation.state[3:6]`` is **axis-angle**, despite the dataset metadata
    labelling it ``roll, pitch, yaw``: the production evaluator builds it with
    ``quat2axisangle(obs["robot0_eef_quat"])``. Applying an Euler conversion here
    corrupts the rotation and therefore the whole wrist-camera reconstruction
    (plan Section 2.1).

    Args:
        axis_angle: ``[..., 3]`` rotation vector; magnitude is the angle in radians.

    Returns:
        ``[..., 3, 3]`` float64 rotation matrices.

    Raises:
        ValueError: If the trailing dimension is not 3.
    """
    vector = np.asarray(axis_angle, dtype=np.float64)
    if vector.shape[-1] != 3:
        raise ValueError(f"axis-angle must have trailing dim 3, got shape {vector.shape}")

    angle = np.linalg.norm(vector, axis=-1)
    # Guard the division only; the near-zero rows are overwritten with identity
    # below, so the placeholder denominator never reaches the result.
    safe_angle = np.where(angle < _AXIS_ANGLE_EPSILON, 1.0, angle)
    axis = vector / safe_angle[..., None]

    zero = np.zeros_like(axis[..., 0])
    skew = np.stack(
        [
            zero, -axis[..., 2], axis[..., 1],
            axis[..., 2], zero, -axis[..., 0],
            -axis[..., 1], axis[..., 0], zero,
        ],
        axis=-1,
    ).reshape(vector.shape[:-1] + (3, 3))

    identity = np.broadcast_to(np.eye(3), vector.shape[:-1] + (3, 3))
    sin = np.sin(angle)[..., None, None]
    cos = np.cos(angle)[..., None, None]
    rotation = identity + sin * skew + (1.0 - cos) * (skew @ skew)
    return np.where((angle < _AXIS_ANGLE_EPSILON)[..., None, None], identity, rotation)


def make_pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Assemble homogeneous ``[..., 4, 4]`` poses from positions and rotations."""
    position = np.asarray(position, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    batch_shape = np.broadcast_shapes(position.shape[:-1], rotation.shape[:-2])
    pose = np.zeros(batch_shape + (4, 4), dtype=np.float64)
    pose[..., :3, :3] = rotation
    pose[..., :3, 3] = position
    pose[..., 3, 3] = 1.0
    return pose


def project_to_pinhole(
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    T_W_from_C: np.ndarray,
    *,
    depth_epsilon: float = DEPTH_EPSILON,
    context: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into continuous pinhole pixels.

    ``T_W_from_C`` is the camera pose **in the world frame**, as robosuite's
    ``get_camera_extrinsic_matrix()`` returns it; the projection needs its
    inverse. Treating it directly as world-to-camera is the standard way to get
    a plausible-looking but wrong anchor (plan Section 6).

    Nothing is rounded, swapped to row/column order, or clipped to the image --
    the robosuite point helper does all three, which is why it is not usable
    here.

    Args:
        points_world: ``[..., 3]`` world positions.
        intrinsics: ``[..., 3, 3]`` pinhole matrices, broadcastable.
        T_W_from_C: ``[..., 4, 4]`` camera-in-world poses, broadcastable.
        depth_epsilon: Depths at or below this are at/behind the camera plane.
        context: Text appended to error messages to identify the caller's frame.

    Returns:
        ``(u, v, depth)`` float64 arrays with the broadcast leading shape.

    Raises:
        InvalidProjectionError: If any input or output is nonfinite, if a camera
            pose is singular, or if any depth is at/behind the camera plane.
    """
    points = np.asarray(points_world, dtype=np.float64)
    matrices = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(T_W_from_C, dtype=np.float64)

    _reject_nonfinite(points, "world point", context)
    _reject_nonfinite(poses, "camera pose", context)
    _reject_nonfinite(matrices, "intrinsic matrix", context)

    try:
        T_C_from_W = np.linalg.inv(poses)
    except np.linalg.LinAlgError as exc:
        raise InvalidProjectionError(
            f"camera pose is singular and cannot be inverted{_suffix(context)}: {exc}"
        ) from exc
    _reject_nonfinite(T_C_from_W, "inverted camera pose", context)

    homogeneous = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    camera_point = np.einsum("...ij,...j->...i", T_C_from_W, homogeneous)[..., :3]
    _reject_nonfinite(camera_point, "camera-frame point", context)

    depth = camera_point[..., 2]
    behind = depth <= depth_epsilon
    if np.any(behind):
        raise InvalidProjectionError(
            f"projection at/behind the camera plane (depth <= {depth_epsilon}) at "
            f"{_format_indices(behind)}{_suffix(context)}; this is an error, not an "
            "off-screen anchor (plan Section 19.1 Decision 1)"
        )

    u = matrices[..., 0, 0] * camera_point[..., 0] / depth + matrices[..., 0, 2]
    v = matrices[..., 1, 1] * camera_point[..., 1] / depth + matrices[..., 1, 2]
    _reject_nonfinite(u, "pinhole u", context)
    _reject_nonfinite(v, "pinhole v", context)
    return u, v, depth


def pinhole_to_consumed(
    u_pinhole: np.ndarray, v_pinhole: np.ndarray, raw_width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map pinhole pixels to the consumed image: a **horizontal flip only**.

    ``u_img = W - 1 - u``; ``v_img = v``, unchanged. This is not a 180-degree
    rotation. Both the stored training video and the live evaluator image are
    ``hflip(pinhole)`` (plan Section 7.1).

    Args:
        u_pinhole: Continuous pinhole column coordinate(s).
        v_pinhole: Continuous pinhole row coordinate(s).
        raw_width: Raw render width in pixels.

    Returns:
        ``(u_img, v_img)`` in consumed-image pixels.
    """
    u_img = (raw_width - 1) - np.asarray(u_pinhole, dtype=np.float64)
    v_img = np.asarray(v_pinhole, dtype=np.float64)
    return u_img, v_img


def resize_crop_point(
    u_img: np.ndarray, v_img: np.ndarray, geometry: ProcessedCameraGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """Map consumed-image pixels through the production resize and centre crop.

    Uses the **half-pixel-centre** mapping ``dst = (src + 0.5) * scale - 0.5``
    that both resamplers in production implement -- PIL ``Image.BILINEAR`` at
    evaluation and ``torchvision.transforms.Resize`` at training. The naive
    ``dst = src * scale`` is wrong and shifts the anchor by roughly 0.01 token
    (plan Section 7.2).

    Args:
        u_img: Consumed-image column coordinate(s).
        v_img: Consumed-image row coordinate(s).
        geometry: Raw/processed geometry for this camera.

    Returns:
        ``(u_processed, v_processed)`` continuous processed-image pixels.
    """
    scale = geometry.scale
    u_processed = (np.asarray(u_img, dtype=np.float64) + 0.5) * scale - 0.5 - geometry.crop_left
    v_processed = (np.asarray(v_img, dtype=np.float64) + 0.5) * scale - 0.5 - geometry.crop_top
    return u_processed, v_processed


def inverse_resize_crop_point(
    u_processed: np.ndarray, v_processed: np.ndarray, geometry: ProcessedCameraGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """Invert :func:`resize_crop_point`, for overlay audits in raw-image space."""
    scale = geometry.scale
    u_img = (np.asarray(u_processed, dtype=np.float64) + geometry.crop_left + 0.5) / scale - 0.5
    v_img = (np.asarray(v_processed, dtype=np.float64) + geometry.crop_top + 0.5) / scale - 0.5
    return u_img, v_img


def consumed_pixel_to_token(
    u_img: np.ndarray, v_img: np.ndarray, geometry: ProcessedCameraGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """Consumed-image pixel -> continuous camera-local token coordinate ``(y, x)``.

    Returned **row first**, matching the ``eef_anchor_token`` model contract of
    plan Section 5.2. The diagnostic tooling's ``token_coords()`` returns
    ``(x, y)``; do not copy its ordering.

    Never rounded, never clipped: a point outside the image yields a token
    coordinate outside ``[0, grid)``, which is what lets the caller classify it
    as off-screen instead of quietly relocating the anchor (plan Section 7.3).

    Args:
        u_img: Consumed-image column coordinate(s).
        v_img: Consumed-image row coordinate(s).
        geometry: Raw/processed geometry for this camera.

    Returns:
        ``(y_token, x_token)``, continuous.
    """
    u_processed, v_processed = resize_crop_point(u_img, v_img, geometry)
    stride = geometry.token_stride
    x_token = (u_processed + 0.5) / stride - 0.5
    y_token = (v_processed + 0.5) / stride - 0.5
    return y_token, x_token


def token_to_consumed_pixel(
    y_token: np.ndarray, x_token: np.ndarray, geometry: ProcessedCameraGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """Invert :func:`consumed_pixel_to_token`, for overlay audits.

    Args:
        y_token: Camera-local token row(s).
        x_token: Camera-local token column(s).
        geometry: Raw/processed geometry for this camera.

    Returns:
        ``(u_img, v_img)`` consumed-image pixels.
    """
    stride = geometry.token_stride
    u_processed = (np.asarray(x_token, dtype=np.float64) + 0.5) * stride - 0.5
    v_processed = (np.asarray(y_token, dtype=np.float64) + 0.5) * stride - 0.5
    return inverse_resize_crop_point(u_processed, v_processed, geometry)


def centre_anchor_pixel(geometry: ProcessedCameraGeometry) -> tuple[float, float]:
    """Image-centre fallback anchor in consumed-image pixels, zero-based centres."""
    return (geometry.raw_width - 1) / 2.0, (geometry.raw_height - 1) / 2.0


def centre_anchor_token(geometry: ProcessedCameraGeometry) -> tuple[float, float]:
    """Image-centre fallback anchor as a camera-local token ``(y, x)``.

    Derived, never transcribed. It resolves to ``(3.0, 3.0)`` at both raw 512 and
    raw 256 -- exactly the ``horizontal`` preset `aligned_3d` already uses, so a
    leading gap degrades to today's fixed-anchor behaviour rather than to an
    invented value (plan Section 19.1 Decision 2).
    """
    u_img, v_img = centre_anchor_pixel(geometry)
    y_token, x_token = consumed_pixel_to_token(u_img, v_img, geometry)
    return float(y_token), float(x_token)


def camera_token_indices(
    camera: str, grid_h: int = 7, grid_w: int = 14, latent_frame: int = 0
) -> np.ndarray:
    """Flattened video-token indices belonging to one camera, for one latent frame.

    Video tokens flatten as ``(frame, row, col)`` with **column changing
    fastest**, so a camera's tokens are **interleaved, not contiguous**::

        main  -> 0..6, 14..20, 28..34, ...
        wrist -> 7..13, 21..27, 35..41, ...

    The intuitive ``arange(49)`` split is wrong by 21 of 49 tokens in each
    direction: it hands the main group 21 wrist tokens and drops 21 real main
    tokens. Nothing crashes and loss still falls (plan Section 8).

    Args:
        camera: ``"main"`` (left half) or ``"wrist"`` (right half).
        grid_h: Composite token-grid height.
        grid_w: Composite token-grid width; must be even.
        latent_frame: Frame index, offsetting by ``grid_h * grid_w``.

    Returns:
        Sorted ``np.ndarray`` of flattened token indices.

    Raises:
        ValueError: If ``grid_w`` is odd or ``camera`` is unknown.
    """
    if grid_w % 2 != 0:
        raise ValueError(f"composite grid width must be even, got {grid_w}")
    if camera not in CAMERA_ORDER:
        raise ValueError(f"camera must be one of {CAMERA_ORDER}, got {camera!r}")

    local_w = grid_w // 2
    tokens_per_frame = grid_h * grid_w
    columns = np.arange(tokens_per_frame) % grid_w
    selected = columns < local_w if camera == "main" else columns >= local_w
    return np.flatnonzero(selected) + latent_frame * tokens_per_frame


# --- anchor resolution policy (plan Section 19.1) ---------------------------


def resolve_anchor_series(
    anchor_token: np.ndarray,
    observed: np.ndarray,
    *,
    centre_anchor: tuple[float, float] = CENTRE_ANCHOR_TOKEN,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply the causal hold-last-observed policy over one episode, in token space.

    Policy (plan Section 19.1)::

        observed                -> use this frame's own anchor
        finite but off-screen   -> hold the last observed anchor, no time limit
        off-screen with no prior-> image centre, the aligned_3d default anchor
        invalid projection      -> ERROR (raised upstream by the projector)

    The policy is **causal**: it reads only frames at or before the current one,
    so training and live inference implement it identically. A leading gap must
    never be back-filled from a later observation -- that reads a future frame,
    which plan Section 2.1 forbids and live inference cannot do at all.

    Args:
        anchor_token: ``[N, 2]`` per-frame anchors as ``(y, x)``.
        observed: ``[N]`` boolean; True when this frame's own projection was
            finite and landed inside the image.
        centre_anchor: Token-space image centre used for a leading gap.

    Returns:
        ``(resolved, stats)`` where ``resolved`` is ``[N, 2]`` float64 and
        ``stats`` carries the plan Section 19.2 substitution counts, so no
        substitution is ever silent.

    Raises:
        InvalidProjectionError: If any anchor is nonfinite. Defence in depth: the
            projector already raises, but ``0 <= nan < W`` is False, so a NaN
            reaching here would be classified off-screen and then *held* --
            precisely the defect Decision 1 exists to prevent.
        ValueError: If the array shapes disagree.
    """
    anchors = np.asarray(anchor_token, dtype=np.float64)
    flags = np.asarray(observed, dtype=bool)
    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError(f"anchor_token must be [N, 2], got shape {anchors.shape}")
    if flags.shape != (anchors.shape[0],):
        raise ValueError(
            f"observed must be [N] matching anchor_token, got {flags.shape} vs {anchors.shape}"
        )
    if not np.isfinite(anchors).all():
        bad = np.flatnonzero(~np.isfinite(anchors).all(axis=1))
        raise InvalidProjectionError(
            f"nonfinite anchor at frame(s) {bad[:_MAX_REPORTED_INVALID].tolist()}"
            f"{'...' if bad.size > _MAX_REPORTED_INVALID else ''}: invalid projections are a "
            "hard error, never held or substituted (plan Section 19.1 Decision 1)"
        )

    resolved = np.empty_like(anchors)
    last: np.ndarray | None = None
    held_offscreen = 0
    centre_default = 0
    run = 0
    max_run = 0
    for index in range(anchors.shape[0]):
        if flags[index]:
            last = anchors[index]
            resolved[index] = anchors[index]
            run = 0
            continue
        run += 1
        max_run = max(max_run, run)
        if last is None:
            # No prior observation to hold: fall back to the aligned_3d centre.
            resolved[index] = centre_anchor
            centre_default += 1
        else:
            resolved[index] = last
            held_offscreen += 1

    stats = {
        "frames": int(anchors.shape[0]),
        "observed_frames": int(flags.sum()),
        "held_offscreen_frames": held_offscreen,
        "centre_default_frames": centre_default,
        "substituted_frames": held_offscreen + centre_default,
        "max_consecutive_held": max_run,
    }
    return resolved, stats


# --- projector --------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class EEFProjection:
    """Result of projecting one or more EEF poses into both cameras.

    Leading dimensions mirror the caller's input. The camera axis follows
    :data:`CAMERA_ORDER`.
    """

    anchor_token: np.ndarray
    """float32 ``[..., 2, 2]`` camera-local anchors, last axis ``(y, x)``."""

    depth: np.ndarray
    """float32 ``[..., 2]`` camera-frame depth in metres; diagnostic only."""

    onscreen: np.ndarray
    """bool ``[..., 2]``; True when the projection landed inside the raw image."""

    anchor_token_precise: np.ndarray
    """float64 ``[..., 2, 2]`` anchors before the float32 model-contract cast."""

    consumed_pixel: np.ndarray
    """float64 ``[..., 2, 2]`` post-flip pixels ``(u, v)``, for overlay audits."""

    pinhole_pixel: np.ndarray
    """float64 ``[..., 2, 2]`` pre-flip pixels ``(u, v)``, for overlay audits."""

    camera_order: tuple[str, ...]
    geometry: ProcessedCameraGeometry


class LiveAnchorResolver:
    """Causal per-actor hold state for live inference.

    One instance belongs to one actor and one episode. Call :meth:`update` on
    **every returned simulator state**, including intermediate action steps
    between replans; call :meth:`reset` at the episode boundary. The current
    resolved anchor is then reused for every Euler denoising step of that
    observation and discarded at the next replan (plan Sections 17, 19.1).
    """

    def __init__(self, projector: "EEFProjector") -> None:
        self.projector = projector
        self._last_observed = np.full((len(CAMERA_ORDER), 2), np.nan, dtype=np.float64)
        self._current = np.tile(
            np.asarray(CENTRE_ANCHOR_TOKEN, dtype=np.float64),
            (len(CAMERA_ORDER), 1),
        )
        self._observed = np.zeros(len(CAMERA_ORDER), dtype=bool)
        self.observed_frames = 0
        self.held_offscreen_frames = 0
        self.centre_default_frames = 0

    def reset(self) -> None:
        """Reset hold history at an episode boundary -- never leak across episodes."""
        self._last_observed.fill(np.nan)
        self._current[:] = np.asarray(CENTRE_ANCHOR_TOKEN, dtype=np.float64)
        self._observed.fill(False)
        self.observed_frames = 0
        self.held_offscreen_frames = 0
        self.centre_default_frames = 0

    @property
    def current(self) -> np.ndarray:
        """Current resolved ``[2, 2]`` float32 anchor, cameras (main,wrist)."""
        return self._current.astype(np.float32, copy=True)

    @property
    def observed(self) -> np.ndarray:
        """Diagnostics-only ``[2]`` flag: measured this step vs substituted."""
        return self._observed.copy()

    def update(self, task_language: str, eef_state: np.ndarray) -> np.ndarray:
        """Project one simulator state and update causal hold state.

        Nonfinite or at/behind-camera projections raise in :class:`EEFProjector`
        and are never held. Finite off-screen positions hold the last observed
        value indefinitely; a leading gap uses local token centre ``(3, 3)``.
        """
        projection = self.projector.project(task_language, eef_state)
        anchor = np.asarray(projection.anchor_token_precise, dtype=np.float64)
        onscreen = np.asarray(projection.onscreen, dtype=bool)
        if anchor.shape != (len(CAMERA_ORDER), 2) or onscreen.shape != (len(CAMERA_ORDER),):
            raise ValueError(
                f"live projection must be [2,2]/[2], got {anchor.shape}/{onscreen.shape}"
            )
        self._observed[:] = onscreen
        for camera in range(len(CAMERA_ORDER)):
            if onscreen[camera]:
                self._last_observed[camera] = anchor[camera]
                self._current[camera] = anchor[camera]
                self.observed_frames += 1
            elif np.isfinite(self._last_observed[camera]).all():
                self._current[camera] = self._last_observed[camera]
                self.held_offscreen_frames += 1
            else:
                self._current[camera] = np.asarray(
                    CENTRE_ANCHOR_TOKEN, dtype=np.float64
                )
                self.centre_default_frames += 1
        return self.current


class EEFProjector:
    """Projects raw metric EEF poses into camera-local visual-token anchors.

    One instance serves one *path*, because the two paths render at different raw
    resolutions: 512 offline, 256 live (plan Section 7.4). Use
    :meth:`for_training` and :meth:`for_live_evaluation` rather than passing the
    resolution by hand.
    """

    def __init__(
        self,
        calibration: CalibrationTable,
        raw_resolution: int,
        *,
        geometry: ProcessedCameraGeometry | None = None,
        camera_order: Sequence[str] = CAMERA_ORDER,
    ) -> None:
        """Build a projector bound to one calibration table and raw resolution.

        Args:
            calibration: Loaded calibration table, keyed by task language string.
            raw_resolution: Square raw render size of the path being served.
            geometry: Processed geometry override; defaults to the LIBERO
                224x224 per-camera configuration at ``raw_resolution``.
            camera_order: Must equal ``("main", "wrist")``; the composite places
                the main camera in the left columns (plan Section 2.2).

        Raises:
            ValueError: If the camera order or the resolved token grid is wrong.
        """
        order = tuple(camera_order)
        if order != CAMERA_ORDER:
            raise ValueError(
                f"camera order is LOCKED to {CAMERA_ORDER} (main occupies the left composite "
                f"columns); got {order}"
            )
        if geometry is None:
            geometry = ProcessedCameraGeometry(raw_width=raw_resolution, raw_height=raw_resolution)
        elif (geometry.raw_width, geometry.raw_height) != (raw_resolution, raw_resolution):
            raise ValueError(
                f"geometry raw size {geometry.raw_width}x{geometry.raw_height} disagrees with "
                f"raw_resolution {raw_resolution}; the flip term W-1 depends on it "
                "(plan Section 7.4)"
            )

        self.calibration = calibration
        self.raw_resolution = int(raw_resolution)
        self.geometry = geometry
        self.camera_order = order
        self._intrinsics_cache: dict[float, np.ndarray] = {}

    @classmethod
    def for_training(
        cls, calibration: CalibrationTable, **kwargs: Any
    ) -> "EEFProjector":
        """Projector for the offline path: stored videos render at raw 512."""
        return cls(calibration, TRAINING_RAW_RESOLUTION, **kwargs)

    @classmethod
    def for_live_evaluation(
        cls, calibration: CalibrationTable, **kwargs: Any
    ) -> "EEFProjector":
        """Projector for the live path: the evaluator renders at raw 256."""
        return cls(calibration, LIVE_RAW_RESOLUTION, **kwargs)

    @property
    def composite_token_grid(self) -> tuple[int, int]:
        """Composite ``(grid_h, grid_w)`` for this projector's camera count."""
        return composite_token_grid(self.geometry, len(self.camera_order))

    def assert_libero_two_camera_grid(self) -> None:
        """Verify the configuration resolves to two local 7x7 grids in a 7x14 composite.

        Raises:
            ValueError: If the local or composite grid is not the resolved LIBERO
                geometry of plan Section 8.
        """
        local = (self.geometry.local_grid_height, self.geometry.local_grid_width)
        if local != (7, 7):
            raise ValueError(f"expected two local 7x7 token grids, resolved {local}")
        if self.composite_token_grid != (7, 14):
            raise ValueError(
                f"expected a 7x14 composite token grid, resolved {self.composite_token_grid}"
            )

    def centre_anchor_token(self) -> tuple[float, float]:
        """Image-centre fallback anchor for this projector, as a token ``(y, x)``."""
        return centre_anchor_token(self.geometry)

    def _intrinsics(self, fovy_deg: float) -> np.ndarray:
        """Cached intrinsics for one ``fovy`` at this projector's raw resolution."""
        matrix = self._intrinsics_cache.get(fovy_deg)
        if matrix is None:
            matrix = intrinsics_from_fovy(fovy_deg, self.raw_resolution, self.raw_resolution)
            self._intrinsics_cache[fovy_deg] = matrix
        return matrix

    def project(
        self, task_language: str | Sequence[str], eef_state: np.ndarray
    ) -> EEFProjection:
        """Project raw metric EEF poses into both cameras' token grids.

        Args:
            task_language: The task's language string, or one string per frame.
                Never a ``task_index`` (plan Section 6.1).
            eef_state: ``[..., D]`` raw, **pre-normalization** ``observation.state``
                with ``D >= 6``: dims 0-2 are the metric EEF position and dims 3-5
                the axis-angle orientation. A single ``[D]`` pose is accepted and
                returns unbatched arrays.

        Returns:
            An :class:`EEFProjection` whose anchors are float32 ``(y, x)`` in
            camera-local token units, plus depth, on-screen flags, and the
            intermediate pixels an overlay audit needs.

        Raises:
            InvalidProjectionError: If any projection is nonfinite or at/behind
                the camera plane. Never downgraded to off-screen.
            ValueError: If shapes or the per-frame language count disagree.
            TypeError: If ``task_language`` is not a string or sequence of strings.
            KeyError: If a language string has no calibration entry.
        """
        state = np.asarray(eef_state, dtype=np.float64)
        if state.ndim == 0 or state.shape[-1] < 6:
            raise ValueError(
                f"eef_state must have trailing dim >= 6 (position 0:3, axis-angle 3:6), "
                f"got shape {state.shape}"
            )
        unbatched = state.ndim == 1
        batch_shape = state.shape[:-1]
        flat = state.reshape(-1, state.shape[-1])
        num_frames = flat.shape[0]

        entries = self._resolve_entries(task_language, num_frames)

        # Checked before any arithmetic: a nonfinite pose otherwise propagates
        # through the matrix products as inf*0 -> NaN, which reports a less
        # useful location and emits numpy warnings on the way.
        _reject_nonfinite(
            flat[:, 0:6], "EEF pose (observation.state[0:6])", f"task={task_language!r}"
        )

        position = flat[:, 0:3]
        T_W_from_eef = make_pose(position, axis_angle_to_matrix(flat[:, 3:6]))
        # The wrist camera is rigidly mounted, so its world pose is derived from
        # the same stored EEF pose the main camera already needs -- no queried
        # matrices (plan Section 6.1). Standard LIBERO shares one universal
        # mount across all 40 tasks; LIBERO-Plus does not, because its
        # robot-state variants swap the robot model, so entries may carry their
        # own. Broadcasting [4,4] and stacking [N,4,4] both compose correctly
        # against T_W_from_eef.
        if any(entry.wrist_mount_T_eef_from_C is not None for entry in entries):
            mount = np.stack(
                [
                    entry.wrist_mount_T_eef_from_C
                    if entry.wrist_mount_T_eef_from_C is not None
                    else self.calibration.wrist_mount_T_eef_from_C
                    for entry in entries
                ]
            )
        else:
            mount = self.calibration.wrist_mount_T_eef_from_C
        T_W_from_wrist = T_W_from_eef @ mount
        T_W_from_main = np.stack([entry.agentview_T_W_from_C for entry in entries])

        poses = np.stack([T_W_from_main, T_W_from_wrist], axis=1)          # [N, 2, 4, 4]
        intrinsics = np.stack(
            [
                np.stack([self._intrinsics(entry.agentview_fovy_deg) for entry in entries]),
                np.stack([self._intrinsics(entry.wrist_fovy_deg) for entry in entries]),
            ],
            axis=1,
        )                                                                   # [N, 2, 3, 3]
        points = np.repeat(position[:, None, :], len(self.camera_order), axis=1)

        u_pinhole, v_pinhole, depth = project_to_pinhole(
            points,
            intrinsics,
            poses,
            context=f"cameras={self.camera_order}, raw_resolution={self.raw_resolution}",
        )
        u_img, v_img = pinhole_to_consumed(u_pinhole, v_pinhole, self.geometry.raw_width)
        y_token, x_token = consumed_pixel_to_token(u_img, v_img, self.geometry)

        onscreen = (
            (u_img >= 0.0)
            & (u_img < self.geometry.raw_width)
            & (v_img >= 0.0)
            & (v_img < self.geometry.raw_height)
        )
        anchors = np.stack([y_token, x_token], axis=-1)                     # [N, 2, 2]

        def _shape(array: np.ndarray) -> np.ndarray:
            """Restore the caller's leading dims, dropping them for a single pose."""
            target = array.shape[1:]
            return array.reshape(target) if unbatched else array.reshape(batch_shape + target)

        return EEFProjection(
            anchor_token=_shape(anchors.astype(np.float32)),
            depth=_shape(depth.astype(np.float32)),
            onscreen=_shape(onscreen),
            anchor_token_precise=_shape(anchors),
            consumed_pixel=_shape(np.stack([u_img, v_img], axis=-1)),
            pinhole_pixel=_shape(np.stack([u_pinhole, v_pinhole], axis=-1)),
            camera_order=self.camera_order,
            geometry=self.geometry,
        )

    def _resolve_entries(
        self, task_language: str | Sequence[str], num_frames: int
    ) -> list[TaskCalibration]:
        """Resolve one calibration entry per frame, keyed by language string."""
        if isinstance(task_language, str):
            return [self.calibration.for_task(task_language)] * num_frames
        if isinstance(task_language, (bytes, bytearray)) or not isinstance(
            task_language, Sequence
        ):
            raise TypeError(
                "task_language must be a string or a sequence of strings, never a task_index "
                f"(got {type(task_language).__name__}); plan Section 6.1"
            )
        languages = list(task_language)
        if len(languages) != num_frames:
            raise ValueError(
                f"got {len(languages)} task language strings for {num_frames} frames"
            )
        cache: dict[str, TaskCalibration] = {}
        entries: list[TaskCalibration] = []
        for language in languages:
            entry = cache.get(language)
            if entry is None:
                entry = self.calibration.for_task(language)
                cache[language] = entry
            entries.append(entry)
        return entries


# --- error-reporting helpers ------------------------------------------------


def _suffix(context: str) -> str:
    """Format an optional context string for an error message."""
    return f" [{context}]" if context else ""


def _format_indices(mask: np.ndarray) -> str:
    """Render the first few offending index tuples of a boolean mask."""
    offenders = np.argwhere(mask)
    shown = ", ".join(
        str(tuple(int(axis) for axis in index)) for index in offenders[:_MAX_REPORTED_INVALID]
    )
    return shown + ("..." if offenders.shape[0] > _MAX_REPORTED_INVALID else "")


def _reject_nonfinite(array: np.ndarray, label: str, context: str) -> None:
    """Raise if ``array`` holds any nonfinite value.

    Checked explicitly and early because the comparison-based tests downstream
    silently misclassify NaN: ``NaN <= eps`` and ``0 <= NaN < W`` are both False,
    so a NaN would sail past the depth test and then be recorded as off-screen.
    """
    finite = np.isfinite(array)
    if not finite.all():
        raise InvalidProjectionError(
            f"nonfinite {label} at {_format_indices(~finite)}{_suffix(context)}; "
            "a nonfinite projection is an error, never an off-screen anchor "
            "(plan Section 19.1 Decision 1)"
        )
