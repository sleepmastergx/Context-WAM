"""LIBERO-Plus per-task anchor table: loading, validation, and consistency.

Author: Rui Heng Yang

The standard calibration (:mod:`fastwam.geometry.calibration`) is keyed by task
**language** and carries one universal wrist mount. Neither holds for
LIBERO-Plus:

* Plus derives language from the perturbed BDDL filename, so its 10,030 tasks
  share only 10,002 distinct language strings. A language join silently
  collides, so this table is keyed by ``"<suite>/<bddl_file>"``.
* Plus robot-state variants swap the robot model
  (LIBERO-plus/libero/libero/envs/env_wrapper.py:219-220), which moves the
  eye-in-hand mount by up to 4.03e-07 -- ~400x the standard exporter's 1e-9
  tolerance. The mount is therefore stored **per task**.

This artifact is evaluation-only and deliberately outside checkpoint geometry
identity, so an existing checkpoint keeps loading against
``libero_camera_calibration_v1.json``. :func:`assert_plus_anchor_consistency`
is what ties the two together, and it checks the property that actually
matters -- anchor displacement in token space -- rather than raw matrix
equality, which cannot hold: the Plus checkout's agentview sits 4.5e-05..8.2e-05
from the standard checkout's for the same nominal task.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .calibration import (
    CalibrationError,
    CalibrationTable,
    TaskCalibration,
    _as_fovy,
    _as_rigid_transform,
    calibration_digest,
)

PLUS_ANCHOR_SCHEMA_VERSION = 1

#: Production artifact location, relative to the repository root.
PLUS_ANCHOR_RELATIVE_PATH = Path("configs/calibration/libero_plus_anchors_v1.json")

_REQUIRED_TASK_FIELDS = (
    "suite",
    "task_index_in_suite",
    "bddl_file",
    "agentview_T_W_from_C",
    "agentview_fovy_deg",
    "wrist_fovy_deg",
    "wrist_mount_T_eef_from_C",
)

#: Maximum tolerated anchor displacement, in token units on the 7x7 local grid,
#: between a Plus entry and the standard calibration for the same nominal task.
#: The measured worst case across suites is 0.0135 tokens; this bound is ~7x
#: that, and still far below the ~0.5-token quantisation the anchors feed. A
#: violation means the Plus artifact was measured against different geometry
#: than the checkpoint was trained on.
PLUS_ANCHOR_TOKEN_TOLERANCE = 0.1


def plus_task_key(suite: str, bddl_file: str) -> str:
    """Stable anchor key for one Plus task. Must match the exporter."""
    return f"{suite}/{bddl_file}"


@dataclass(frozen=True, eq=False)
class PlusTaskAnchor:
    """Camera geometry for one LIBERO-Plus task variant."""

    key: str
    suite: str
    task_index_in_suite: int
    bddl_file: str
    agentview_T_W_from_C: np.ndarray
    agentview_fovy_deg: float
    wrist_fovy_deg: float
    wrist_mount_T_eef_from_C: np.ndarray


class PlusAnchorTable:
    """Per-task LIBERO-Plus camera geometry, keyed by ``"<suite>/<bddl_file>"``."""

    def __init__(
        self,
        *,
        digest: str,
        base_calibration_digest: str,
        tasks: Mapping[str, PlusTaskAnchor],
        camera_names: Mapping[str, str],
        schema_version: int,
        source_path: Path | None = None,
    ) -> None:
        self.digest = digest
        self.base_calibration_digest = base_calibration_digest
        self.camera_names = dict(camera_names)
        self.schema_version = schema_version
        self.source_path = source_path
        self._tasks = dict(tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tasks)

    def __contains__(self, key: object) -> bool:
        return key in self._tasks

    @property
    def keys_sorted(self) -> tuple[str, ...]:
        """All anchor keys, sorted, for diagnostics and preflight reporting."""
        return tuple(sorted(self._tasks))

    def for_task(self, key: str) -> PlusTaskAnchor:
        """Look up one Plus task's geometry by its ``"<suite>/<bddl_file>"`` key.

        Raises:
            TypeError: If ``key`` is not a string. Passing a ``task_index`` here
                would select an unrelated task's camera without failing.
            KeyError: If no entry exists, with a suite-scoped count so a missing
                artifact is distinguishable from a malformed key.
        """
        if not isinstance(key, str):
            raise TypeError(
                "the Plus anchor table is keyed by '<suite>/<bddl_file>', never by "
                f"task_index: got {type(key).__name__}"
            )
        entry = self._tasks.get(key)
        if entry is None:
            suite = key.split("/", 1)[0]
            in_suite = sum(1 for item in self._tasks if item.startswith(f"{suite}/"))
            raise KeyError(
                f"no Plus anchor entry for {key!r}; the table holds {len(self._tasks)} "
                f"tasks ({in_suite} in suite {suite!r}). Regenerate it with "
                "scripts_ruiheng/eef_anchor/export_plus_calibration.py"
            )
        return entry

    def as_calibration_table(self) -> CalibrationTable:
        """Adapt to the :class:`CalibrationTable` interface the projector consumes.

        Per-task mounts ride along on each :class:`TaskCalibration`; the
        table-level mount is the first entry's and is never used when per-task
        mounts are present (see ``EEFProjector.project``).
        """
        # CalibrationTable.for_task() normalizes its query with strip().lower()
        # but the constructor stores keys verbatim, so keys must be
        # pre-normalized here or every uppercase scene name (KITCHEN_SCENE3_...)
        # would miss.
        tasks: dict[str, TaskCalibration] = {}
        for key, entry in self._tasks.items():
            normalized = key.strip().lower()
            if normalized in tasks:
                raise CalibrationError(
                    f"Plus anchor keys {key!r} and "
                    f"{tasks[normalized].task_language!r} collide after "
                    "case normalization; the projector could not tell them apart"
                )
            tasks[normalized] = TaskCalibration(
                task_language=normalized,
                suite=entry.suite,
                task_index_in_suite=entry.task_index_in_suite,
                bddl_file=entry.bddl_file,
                agentview_T_W_from_C=entry.agentview_T_W_from_C,
                agentview_fovy_deg=entry.agentview_fovy_deg,
                wrist_fovy_deg=entry.wrist_fovy_deg,
                wrist_mount_T_eef_from_C=entry.wrist_mount_T_eef_from_C,
            )
        reference = next(iter(self._tasks.values()))
        return CalibrationTable(
            digest=self.digest,
            wrist_mount_T_eef_from_C=reference.wrist_mount_T_eef_from_C,
            tasks=tasks,
            camera_names=self.camera_names,
            schema_version=self.schema_version,
            source_path=self.source_path,
        )

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        digest: str,
        source_path: Path | None = None,
    ) -> "PlusAnchorTable":
        """Build a validated table from a parsed Plus anchor payload."""
        schema_version = payload.get("schema_version")
        if schema_version != PLUS_ANCHOR_SCHEMA_VERSION:
            raise CalibrationError(
                f"unsupported Plus anchor schema_version {schema_version!r}; "
                f"this code reads version {PLUS_ANCHOR_SCHEMA_VERSION}"
            )
        for field in ("camera_names", "tasks", "base_calibration_digest"):
            if field not in payload:
                raise CalibrationError(f"Plus anchor payload is missing {field!r}")

        raw_tasks = payload["tasks"]
        if not isinstance(raw_tasks, Mapping) or not raw_tasks:
            raise CalibrationError("Plus anchor payload holds no tasks")

        tasks: dict[str, PlusTaskAnchor] = {}
        for key, entry in raw_tasks.items():
            for field in _REQUIRED_TASK_FIELDS:
                if field not in entry:
                    raise CalibrationError(f"Plus anchor task {key!r} is missing {field!r}")
            identity = plus_task_key(str(entry["suite"]), str(entry["bddl_file"]))
            if identity != key:
                raise CalibrationError(
                    f"Plus anchor key {key!r} disagrees with its own "
                    f"(suite, bddl_file) -> {identity!r}; the artifact is inconsistent"
                )
            tasks[key] = PlusTaskAnchor(
                key=key,
                suite=str(entry["suite"]),
                task_index_in_suite=int(entry["task_index_in_suite"]),
                bddl_file=str(entry["bddl_file"]),
                agentview_T_W_from_C=_as_rigid_transform(
                    entry["agentview_T_W_from_C"], f"{key}:agentview_T_W_from_C"
                ),
                agentview_fovy_deg=_as_fovy(
                    entry["agentview_fovy_deg"], f"{key}:agentview_fovy_deg"
                ),
                wrist_fovy_deg=_as_fovy(entry["wrist_fovy_deg"], f"{key}:wrist_fovy_deg"),
                wrist_mount_T_eef_from_C=_as_rigid_transform(
                    entry["wrist_mount_T_eef_from_C"], f"{key}:wrist_mount_T_eef_from_C"
                ),
            )

        return cls(
            digest=digest,
            base_calibration_digest=str(payload["base_calibration_digest"]),
            tasks=tasks,
            camera_names=payload["camera_names"],
            schema_version=int(schema_version),
            source_path=source_path,
        )


def load_plus_anchors(path: str | Path, *, require_canonical: bool = True) -> PlusAnchorTable:
    """Load and validate the Plus anchor artifact at ``path``."""
    artifact = Path(path)
    digest = calibration_digest(artifact, require_canonical=require_canonical)
    payload = json.loads(artifact.read_bytes().decode("utf-8"))
    return PlusAnchorTable.from_payload(payload, digest=digest, source_path=artifact)


# --- consistency with the checkpoint's standard calibration -----------------


#: EEF positions sweeping LIBERO's reachable tabletop workspace. The guard
#: compares anchors over this grid rather than at a single pose, because a
#: rotation-only discrepancy vanishes at the image centre and only shows up off
#: axis.
_WORKSPACE_GRID_BOUNDS = ((-0.30, 0.30, 7), (-0.30, 0.30, 7), (0.90, 1.35, 5))


def parse_plus_bddl(bddl_file: str) -> tuple[str, dict[str, Any]]:
    """Split a Plus BDDL filename into its base file and runtime parameters.

    Mirrors LIBERO-plus/libero/libero/envs/env_wrapper.py:207-221, which is what
    actually determines the camera pose and robot model at runtime. Plus encodes
    those parameters in the filename, and applies the synthetic
    ``_view_..._initstate_...`` suffix even when every parameter is baseline --
    so the raw ``bddl_file`` never equals the standard calibration's name, and
    comparing them directly finds nothing.

    Returns:
        ``(base_bddl, params)`` where ``params`` carries the view parameters,
        ``init_state``, and ``noise``.
    """
    name = str(bddl_file)
    if "_view_" in name and "_initstate_" in name:
        base, angle_view_initstate = name.split("_view_")
        base = base + ".bddl"
        angle_view, init_state = angle_view_initstate.split("_initstate_")
        init_state = init_state.split(".")[0]
        if "_noise_" in init_state:
            init_state, noise = init_state.split("_noise_")
            noise = int(noise)
        else:
            noise = 0
        horizon, vertical, scale, rot, end_vertical = angle_view.split("_")
        return base, {
            "horizon_view": int(horizon),
            "vertical_view": int(vertical),
            "scale_factor": float(int(scale) / 100),
            "end_point_rot": int(rot),
            "end_point_vertical": int(end_vertical),
            "init_state": int(init_state),
            "noise": noise,
        }
    return name, {
        "horizon_view": 0,
        "vertical_view": 0,
        "scale_factor": 1.0,
        "end_point_rot": 0,
        "end_point_vertical": 0,
        "init_state": 0,
        "noise": 0,
    }


def is_geometrically_unperturbed(params: Mapping[str, Any]) -> bool:
    """True when a Plus task keeps its base task's camera and robot.

    Sensor noise is applied to rendered pixels rather than the scene, so a noise
    variant of an unperturbed scene still shares the base geometry.
    """
    return (
        params["horizon_view"] == 0
        and params["vertical_view"] == 0
        and params["end_point_rot"] == 0
        and params["end_point_vertical"] == 0
        and params["scale_factor"] == 1.0
        and params["init_state"] == 0
    )


def _workspace_grid() -> np.ndarray:
    """``[N, 6]`` probe states: gridded positions, identity axis-angle."""
    (x0, x1, nx), (y0, y1, ny), (z0, z1, nz) = _WORKSPACE_GRID_BOUNDS
    xs, ys, zs = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny), np.linspace(z0, z1, nz)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.concatenate([grid, np.zeros_like(grid)], axis=1)


def max_anchor_displacement(
    reference: CalibrationTable,
    reference_key: str,
    candidate: CalibrationTable,
    candidate_key: str,
) -> float:
    """Worst main-camera anchor displacement, in tokens, between two geometries.

    Both tables are projected through the production live-evaluation projector
    over the same workspace grid, so this measures the quantity the model
    actually consumes rather than a matrix norm with no physical meaning.
    """
    from .eef_projector import EEFProjector

    grid = _workspace_grid()
    ref = EEFProjector.for_live_evaluation(reference).project(reference_key, grid)
    cand = EEFProjector.for_live_evaluation(candidate).project(candidate_key, grid)
    delta = np.abs(
        np.asarray(ref.anchor_token_precise)[:, 0, :]
        - np.asarray(cand.anchor_token_precise)[:, 0, :]
    )
    return float(delta.max())


def assert_plus_anchor_consistency(
    plus_table: "PlusAnchorTable",
    base_calibration: CalibrationTable,
    *,
    tolerance: float = PLUS_ANCHOR_TOKEN_TOLERANCE,
    max_tasks: int = 0,
) -> dict[str, Any]:
    """Refuse to serve a Plus table measured against foreign base geometry.

    The Plus artifact sits outside checkpoint geometry identity, so nothing else
    would notice if it had been exported from a different LIBERO checkout, a
    different robot, or a different camera convention. Every such divergence
    shows up as an anchor offset, so that is what this checks.

    Only geometrically *unperturbed* Plus tasks are comparable: a camera-viewpoint
    variant is *supposed* to differ from its base task, and asserting otherwise
    would forbid the very perturbation the table exists to describe. Those are
    identified by their BDDL name matching a standard entry exactly.

    Args:
        plus_table: The loaded Plus anchor table.
        base_calibration: The checkpoint's standard calibration.
        tolerance: Maximum tolerated anchor displacement in token units.
        max_tasks: Cap on comparisons (0 = all comparable tasks).

    Returns:
        A report with the comparison count and the worst displacement.

    Raises:
        CalibrationError: If no task is comparable (the check would otherwise
            pass vacuously), or if any comparison exceeds ``tolerance``.
    """
    base_by_bddl = {
        (entry.suite, entry.bddl_file): language
        for language, entry in (
            (key, base_calibration.for_task(key)) for key in base_calibration.task_languages
        )
        if entry.bddl_file
    }

    comparable = []
    for key, entry in sorted(plus_table._tasks.items()):
        base_bddl, params = parse_plus_bddl(entry.bddl_file)
        if not is_geometrically_unperturbed(params):
            continue
        base_language = base_by_bddl.get((entry.suite, base_bddl))
        if base_language is None:
            continue
        comparable.append((key, entry, base_language))
    if not comparable:
        raise CalibrationError(
            "no LIBERO-Plus task shares a BDDL file with the standard calibration, so "
            "the Plus anchor table cannot be checked against the checkpoint's geometry. "
            "Refusing to report consistency rather than passing on an empty comparison."
        )
    if max_tasks:
        comparable = comparable[:max_tasks]

    plus_as_calibration = plus_table.as_calibration_table()
    worst = 0.0
    worst_key = ""
    for key, _entry, base_language in comparable:
        shift = max_anchor_displacement(
            base_calibration, base_language, plus_as_calibration, key
        )
        if shift > worst:
            worst, worst_key = shift, key

    if worst > tolerance:
        raise CalibrationError(
            f"LIBERO-Plus anchor table diverges from the checkpoint's calibration by "
            f"{worst:.4f} tokens at {worst_key!r}, above the {tolerance} tolerance. The "
            "table was measured against different geometry than the checkpoint was "
            "trained on, so its anchors describe a different spatial origin. "
            f"Plus artifact digest={plus_table.digest[:12]} "
            f"base digest={base_calibration.digest[:12]}"
        )

    return {
        "compared_tasks": len(comparable),
        "max_token_displacement": worst,
        "worst_task": worst_key,
        "tolerance": tolerance,
    }
