"""LIBERO camera calibration: canonical serialization, content digest, and lookup.

Author: Rui Heng Yang

The calibration artifact *is* the geometry. If an extrinsic, a ``fovy``, or the
wrist mount changes, every EEF anchor changes, and any checkpoint trained
against the old values is invalid. Its bytes are therefore part of checkpoint
identity, which is why this module defines one canonical byte form and one
digest rather than leaving serialization to the caller.
(Plan: /home/ruiheng/.claude/plans/fastwam/200731_RoPE_Anchor.md, Sections 15.3, 20.1)

Canonical form -- DECIDED 2026-08-03, do not vary::

    path       configs/calibration/libero_camera_calibration_v1.json
    precision  every float rounded to 12 decimals BEFORE serialization
    canonical  json.dumps(obj, sort_keys=True, separators=(',', ':'))
    digest     sha256 over the canonical UTF-8 bytes

Rounding happens before serialization so the digest is invariant to platform
float-repr differences rather than merely usually equal.

Lookup is keyed by the task **language string**, never by ``task_index``. The
dataset's task order does not match the LIBERO benchmark's: ``libero_10``
dataset task 0 is benchmark task 2, which uses a different agentview camera, so
an index join silently mis-calibrates six of that suite's ten tasks.
(Plan Section 6.1, finding F5)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

# --- canonical-form constants (plan Section 15.3; do not vary) --------------

CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_FLOAT_DECIMALS = 12
CALIBRATION_JSON_SEPARATORS = (",", ":")

#: Production artifact location, relative to the repository root.
CALIBRATION_RELATIVE_PATH = Path("configs/calibration/libero_camera_calibration_v1.json")

#: Simulator camera names behind the two LIBERO observation keys.
CAMERA_NAMES = {"main": "agentview", "wrist": "robot0_eye_in_hand"}

#: Rigid transforms must end in this row; a violation means the artifact does
#: not hold homogeneous 4x4 poses and every downstream projection is nonsense.
_HOMOGENEOUS_BOTTOM_ROW = (0.0, 0.0, 0.0, 1.0)

_REQUIRED_TASK_FIELDS = (
    "suite",
    "task_index_in_suite",
    "agentview_T_W_from_C",
    "agentview_fovy_deg",
    "wrist_fovy_deg",
)


class CalibrationError(ValueError):
    """Raised when a calibration artifact is missing, malformed, or non-canonical."""


# --- canonical serialization ------------------------------------------------


def round_floats(value: Any, decimals: int = CALIBRATION_FLOAT_DECIMALS) -> Any:
    """Recursively round every float in ``value`` to ``decimals`` decimal places.

    Applied *before* serialization so the JSON text -- and therefore the digest
    -- cannot vary with a platform's shortest-round-trip float repr.

    Two normalizations beyond plain ``round()``:

    * negative zero collapses to ``0.0``. ``round(-1e-15, 12)`` yields ``-0.0``,
      which is numerically equal to ``0.0`` but serializes as a different
      string; picking one representative among numerically equal values is what
      "canonical" means and changes no geometry.
    * nonfinite floats are rejected. ``json.dumps`` would emit ``NaN`` /
      ``Infinity``, which are not valid JSON and would poison every projection
      that reads them.

    Args:
        value: Any JSON-compatible structure, optionally containing numpy scalars
            or arrays.
        decimals: Decimal places to keep.

    Returns:
        The same structure with floats rounded and numpy types converted to
        built-ins.

    Raises:
        CalibrationError: If a nonfinite float is encountered.
        TypeError: If a value is not JSON-representable.
    """
    if isinstance(value, Mapping):
        return {str(key): round_floats(item, decimals) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return round_floats(value.tolist(), decimals)
    if isinstance(value, (list, tuple)):
        return [round_floats(item, decimals) for item in value]
    # bool is a subclass of int, so it must be tested before the numeric cases.
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise CalibrationError(
                f"calibration contains a nonfinite float ({number!r}); "
                "a nonfinite extrinsic, fovy, or mount entry corrupts every projection"
            )
        rounded = round(number, decimals)
        return 0.0 if rounded == 0.0 else rounded  # collapse -0.0 to 0.0
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"calibration payload holds a non-serializable value of type {type(value)!r}")


def canonical_calibration_json(payload: Mapping[str, Any]) -> str:
    """Return the canonical JSON text for ``payload`` (rounded, sorted, compact)."""
    return json.dumps(
        round_floats(payload),
        sort_keys=True,
        separators=CALIBRATION_JSON_SEPARATORS,
        allow_nan=False,
    )


def canonical_calibration_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return the canonical UTF-8 bytes that the artifact file must contain."""
    return canonical_calibration_json(payload).encode("utf-8")


def calibration_digest_from_payload(payload: Mapping[str, Any]) -> str:
    """Return the sha256 hex digest of ``payload``'s canonical bytes."""
    return hashlib.sha256(canonical_calibration_bytes(payload)).hexdigest()


def calibration_digest(path: str | Path, *, require_canonical: bool = True) -> str:
    """Return the content digest of the calibration artifact at ``path``.

    This is the value Stage 7 persists in checkpoint metadata as
    ``eef_relative_camera_calibration_digest`` and the live projector asserts at
    inference. A mismatch is a hard error (plan Section 20.1).

    Args:
        path: Artifact path.
        require_canonical: When True (default), the file's bytes must already be
            the canonical form. A file that parses to the right content but was
            hand-edited or re-indented did not come from the exporter, and
            silently normalizing it would hide that.

    Returns:
        Lowercase sha256 hex digest over the canonical UTF-8 bytes.

    Raises:
        CalibrationError: If the file is missing, unparsable, or non-canonical.
    """
    artifact = Path(path)
    try:
        raw = artifact.read_bytes()
    except OSError as exc:
        raise CalibrationError(f"cannot read calibration artifact {artifact}: {exc}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"calibration artifact {artifact} is not valid JSON: {exc}") from exc

    canonical = canonical_calibration_bytes(payload)
    if require_canonical and raw != canonical:
        raise CalibrationError(
            f"calibration artifact {artifact} is not in canonical form "
            f"({len(raw)} bytes on disk, {len(canonical)} canonical); "
            "regenerate it with scripts_ruiheng/eef_anchor/export_calibration.py"
        )
    return hashlib.sha256(canonical).hexdigest()


# --- typed access -----------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TaskCalibration:
    """Calibration for one LIBERO task, resolved by its language string.

    ``agentview_T_W_from_C`` is the camera pose **in the world frame**
    (``T_W_from_C``), exactly as robosuite's ``get_camera_extrinsic_matrix()``
    returns it. The projector needs its inverse; never treat it as
    world-to-camera (plan Section 6).
    """

    task_language: str
    suite: str
    task_index_in_suite: int
    bddl_file: str
    agentview_T_W_from_C: np.ndarray
    agentview_fovy_deg: float
    wrist_fovy_deg: float
    #: Per-task wrist mount, or ``None`` to use the table's universal mount.
    #: Only LIBERO-Plus populates this: its robot-state variants swap the robot
    #: model, so the single-mount premise of plan Section 6.1 does not hold
    #: there (see :mod:`fastwam.geometry.plus_anchors`). Standard LIBERO leaves
    #: it ``None`` and the table-level constant applies unchanged.
    wrist_mount_T_eef_from_C: np.ndarray | None = None


class CalibrationTable:
    """All 40 LIBERO tasks' camera calibration, keyed by task language string.

    The wrist camera is rigidly mounted to the end effector, so it needs no
    per-task entry: its world pose is ``T_W_from_eef @ wrist_mount_T_eef_from_C``
    with one universal mount constant (plan Section 6.1).
    """

    def __init__(
        self,
        *,
        digest: str,
        wrist_mount_T_eef_from_C: np.ndarray,
        tasks: Mapping[str, TaskCalibration],
        camera_names: Mapping[str, str],
        schema_version: int,
        source_path: Path | None = None,
    ) -> None:
        self.digest = digest
        self.wrist_mount_T_eef_from_C = wrist_mount_T_eef_from_C
        self.camera_names = dict(camera_names)
        self.schema_version = schema_version
        self.source_path = source_path
        self._tasks = dict(tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tasks)

    @property
    def task_languages(self) -> tuple[str, ...]:
        """Normalized language keys, sorted, for diagnostics and error messages."""
        return tuple(sorted(self._tasks))

    def for_task(self, task_language: str) -> TaskCalibration:
        """Look up one task's calibration by its language string.

        Args:
            task_language: The task's natural-language description. Matching is
                case-insensitive and strips surrounding whitespace, mirroring how
                the exporter and the LeRobot metadata readers normalize it.

        Returns:
            The task's calibration.

        Raises:
            TypeError: If an integer is passed. The dataset's ``task_index``
                does not agree with the LIBERO benchmark's task order, so an
                index join silently selects the wrong agentview camera for six
                ``libero_10`` tasks (plan Section 6.1). Failing loudly on the
                type is the only cheap way to catch that.
            KeyError: If no entry exists for the language string.
        """
        if not isinstance(task_language, str):
            raise TypeError(
                "calibration is keyed by the task LANGUAGE STRING, never by task_index: "
                f"got {type(task_language).__name__}. The dataset's task order differs "
                "from the LIBERO benchmark's, so an index join mis-calibrates tasks "
                "without failing (plan Section 6.1)."
            )
        key = task_language.strip().lower()
        if key not in self._tasks:
            raise KeyError(
                f"no calibration entry for task language {task_language!r} "
                f"(normalized {key!r}); the table holds {len(self._tasks)} tasks"
            )
        return self._tasks[key]

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        digest: str | None = None,
        source_path: Path | None = None,
    ) -> "CalibrationTable":
        """Build a validated table from a parsed calibration payload.

        Args:
            payload: The parsed artifact contents.
            digest: Precomputed content digest; recomputed from the payload when
                omitted.
            source_path: Origin of the payload, for error messages.

        Returns:
            A validated ``CalibrationTable``.

        Raises:
            CalibrationError: If the payload is malformed or the schema version
                is unsupported.
        """
        schema_version = payload.get("schema_version")
        if schema_version != CALIBRATION_SCHEMA_VERSION:
            raise CalibrationError(
                f"unsupported calibration schema_version {schema_version!r}; "
                f"this code reads version {CALIBRATION_SCHEMA_VERSION}"
            )
        for field in ("wrist_mount_T_eef_from_C", "tasks", "camera_names"):
            if field not in payload:
                raise CalibrationError(f"calibration payload is missing {field!r}")

        mount = _as_rigid_transform(payload["wrist_mount_T_eef_from_C"], "wrist_mount_T_eef_from_C")

        raw_tasks = payload["tasks"]
        if not isinstance(raw_tasks, Mapping) or not raw_tasks:
            raise CalibrationError("calibration payload holds no tasks")

        tasks: dict[str, TaskCalibration] = {}
        for language, entry in raw_tasks.items():
            for field in _REQUIRED_TASK_FIELDS:
                if field not in entry:
                    raise CalibrationError(f"task {language!r} is missing {field!r}")
            key = str(language).strip().lower()
            if key in tasks:
                raise CalibrationError(f"duplicate task language key {key!r} after normalization")
            tasks[key] = TaskCalibration(
                task_language=key,
                suite=str(entry["suite"]),
                task_index_in_suite=int(entry["task_index_in_suite"]),
                bddl_file=str(entry.get("bddl_file", "")),
                agentview_T_W_from_C=_as_rigid_transform(
                    entry["agentview_T_W_from_C"], f"{key}:agentview_T_W_from_C"
                ),
                agentview_fovy_deg=_as_fovy(
                    entry["agentview_fovy_deg"], f"{key}:agentview_fovy_deg"
                ),
                wrist_fovy_deg=_as_fovy(entry["wrist_fovy_deg"], f"{key}:wrist_fovy_deg"),
            )

        return cls(
            digest=digest if digest is not None else calibration_digest_from_payload(payload),
            wrist_mount_T_eef_from_C=mount,
            tasks=tasks,
            camera_names=payload["camera_names"],
            schema_version=int(schema_version),
            source_path=source_path,
        )


def load_calibration(path: str | Path, *, require_canonical: bool = True) -> CalibrationTable:
    """Load and validate the calibration artifact at ``path``.

    Args:
        path: Artifact path.
        require_canonical: Enforce the canonical byte form (see
            :func:`calibration_digest`).

    Returns:
        A validated table carrying its own content digest.

    Raises:
        CalibrationError: If the artifact is missing, unparsable, non-canonical,
            or malformed.
    """
    artifact = Path(path)
    digest = calibration_digest(artifact, require_canonical=require_canonical)
    payload = json.loads(artifact.read_bytes().decode("utf-8"))
    return CalibrationTable.from_payload(payload, digest=digest, source_path=artifact)


# --- validation helpers -----------------------------------------------------


def _as_rigid_transform(value: Any, label: str) -> np.ndarray:
    """Validate a 4x4 homogeneous transform and return it as float64."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise CalibrationError(f"{label} must be 4x4, got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise CalibrationError(f"{label} contains nonfinite entries")
    if not np.allclose(matrix[3], _HOMOGENEOUS_BOTTOM_ROW, atol=1e-9):
        raise CalibrationError(f"{label} is not homogeneous; bottom row is {matrix[3].tolist()}")
    return matrix


def _as_fovy(value: Any, label: str) -> float:
    """Validate a vertical field of view in degrees."""
    fovy = float(value)
    if not np.isfinite(fovy) or not 0.0 < fovy < 180.0:
        raise CalibrationError(f"{label} must lie in (0, 180) degrees, got {fovy!r}")
    return fovy
