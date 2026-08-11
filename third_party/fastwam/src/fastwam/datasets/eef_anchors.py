# Author: Rui Heng Yang
"""Load-time per-episode EEF anchor resolution.

There is no sidecar. Anchors are projected and hold-resolved once per episode at
dataset initialisation and held in RAM (plan Section 15). Projection costs about
35 microseconds per frame, so precomputing to disk bought nothing while adding
stale-artifact and partial-write failure modes.

Two traps this module exists to close:

* **Episode identity is ``(dataset_index, episode_index)``, not
  ``episode_index``.** A ``MultiLeRobotDataset`` concatenates several LeRobot
  directories, each numbering its episodes from zero, so ``episode_index`` alone
  collides across directories and would silently pair one suite's frames with
  another suite's gripper positions.
* **The anchor must key off the RETURNED sample, not the requested index.**
  Both ``RobotVideoDataset._get()`` and ``__getitem__()`` resample to a random
  global index on padding or on exception, so a lookup by requested ``idx``
  attaches geometry from whichever episode was originally asked for -- with no
  error and no log.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from fastwam.geometry import (
    CAMERA_ORDER,
    CalibrationTable,
    EEFProjector,
    load_calibration,
    resolve_anchor_series,
)

logger = logging.getLogger(__name__)

EEF_POSE_DIMS = 6
"""``observation.state[0:3]`` is metric position, ``[3:6]`` axis-angle."""


@dataclass
class AnchorResolutionStats:
    """Substitution accounting, so no substitution is ever silent (Section 19.2)."""

    episodes: int = 0
    frames: int = 0
    observed_frames: int = 0
    held_offscreen_frames: int = 0
    centre_default_frames: int = 0
    max_consecutive_held: int = 0
    per_camera_observed: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "frames": self.frames,
            "observed_frames": self.observed_frames,
            "held_offscreen_frames": self.held_offscreen_frames,
            "centre_default_frames": self.centre_default_frames,
            "max_consecutive_held": self.max_consecutive_held,
            "per_camera_observed": dict(self.per_camera_observed),
        }


class EpisodeAnchorIndex:
    """Resolved per-episode anchors for every episode of every dataset dir.

    Keyed by ``(dataset_index, episode_index)``; ``lookup()`` then indexes the
    episode's series at the sample's own ``frame_index``.
    """

    def __init__(
        self,
        *,
        calibration: CalibrationTable,
        projector: EEFProjector,
        anchors: Mapping[tuple[int, int], np.ndarray],
        stats: AnchorResolutionStats,
    ) -> None:
        self.calibration = calibration
        self.projector = projector
        self._anchors = dict(anchors)
        self.stats = stats

    @property
    def calibration_digest(self) -> str:
        """Content digest of the calibration this index was resolved against."""
        return self.calibration.digest

    def __len__(self) -> int:
        return len(self._anchors)

    def lookup(
        self, dataset_index: int, episode_index: int, frame_index: int
    ) -> np.ndarray:
        """Return the ``[2, 2]`` float32 anchor for one frame, as ``(y, x)``.

        Args:
            dataset_index: Which LeRobot directory the returned sample came from.
            episode_index: Episode index **within that directory**.
            frame_index: Episode-local frame index of the sample's ``t0``.

        Raises:
            KeyError: If the episode was not resolved at initialisation.
            IndexError: If ``frame_index`` lies outside the episode.
        """
        key = (int(dataset_index), int(episode_index))
        try:
            series = self._anchors[key]
        except KeyError as err:
            raise KeyError(
                f"no resolved anchors for (dataset_index={dataset_index}, "
                f"episode_index={episode_index}); the anchor index and the "
                "dataset disagree about which episodes exist"
            ) from err
        frame = int(frame_index)
        if not 0 <= frame < series.shape[0]:
            raise IndexError(
                f"frame_index {frame} outside episode "
                f"(dataset_index={dataset_index}, episode_index={episode_index}) "
                f"of length {series.shape[0]}"
            )
        return series[frame]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        dataset_dirs: Sequence[str],
        *,
        calibration_path: str,
        projector: EEFProjector | None = None,
        episode_filter: Mapping[int, Sequence[int]] | None = None,
    ) -> "EpisodeAnchorIndex":
        """Project and hold-resolve every episode of every directory.

        Args:
            dataset_dirs: LeRobot directories, in the order the dataset stacks
                them -- this ordering defines ``dataset_index``.
            calibration_path: Versioned calibration artifact (Section 15.3).
            projector: Defaults to the training projector (raw 512).
            episode_filter: Optional ``{dataset_index: [episode_index, ...]}``
                restricting work to the episodes actually selected by the split.

        Raises:
            InvalidProjectionError: If any frame projects to a nonfinite
                coordinate or at/behind the camera plane (Section 19.1
                Decision 1). This is a hard error by design: substituting a
                stale anchor would convert a detectable geometry fault into
                silently wrong training data.
        """
        calibration = load_calibration(calibration_path)
        if projector is None:
            projector = EEFProjector.for_training(calibration)
        if tuple(projector.camera_order) != CAMERA_ORDER:
            raise ValueError(
                f"camera order is LOCKED to {CAMERA_ORDER}, got "
                f"{tuple(projector.camera_order)}"
            )

        anchors: dict[tuple[int, int], np.ndarray] = {}
        stats = AnchorResolutionStats(
            per_camera_observed={name: 0 for name in CAMERA_ORDER}
        )

        for dataset_index, root in enumerate(dataset_dirs):
            wanted = None
            if episode_filter is not None:
                wanted = {int(e) for e in episode_filter.get(dataset_index, ())}
            cls._resolve_one_directory(
                root=root,
                dataset_index=dataset_index,
                projector=projector,
                anchors=anchors,
                stats=stats,
                wanted_episodes=wanted,
            )

        logger.info(
            "EEF anchors resolved at load time: %d episodes, %d frames, "
            "%d observed, %d held-offscreen, %d centre-default, "
            "longest hold %d frames (calibration %s)",
            stats.episodes,
            stats.frames,
            stats.observed_frames,
            stats.held_offscreen_frames,
            stats.centre_default_frames,
            stats.max_consecutive_held,
            calibration.digest[:12],
        )
        return cls(
            calibration=calibration,
            projector=projector,
            anchors=anchors,
            stats=stats,
        )

    @staticmethod
    def _episode_task_languages(root: str) -> dict[int, str]:
        """Map episode index to its task language string.

        Calibration is joined by **language string**, never ``task_index``:
        the dataset's task ordering differs from the benchmark's, so an
        index join silently selects another task's camera pose (Section 6.1).
        """
        path = os.path.join(root, "meta", "episodes.jsonl")
        languages: dict[int, str] = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                tasks = record.get("tasks") or []
                if not tasks:
                    raise ValueError(
                        f"episode {record.get('episode_index')} in {root} has no "
                        "task language; calibration cannot be joined"
                    )
                languages[int(record["episode_index"])] = str(tasks[0])
        if not languages:
            raise ValueError(f"no episodes found in {path}")
        return languages

    @classmethod
    def _resolve_one_directory(
        cls,
        *,
        root: str,
        dataset_index: int,
        projector: EEFProjector,
        anchors: dict[tuple[int, int], np.ndarray],
        stats: AnchorResolutionStats,
        wanted_episodes: set[int] | None,
    ) -> None:
        import pandas as pd  # local: keeps the module importable without pandas

        languages = cls._episode_task_languages(root)
        files = sorted(
            glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True)
        )
        if not files:
            raise FileNotFoundError(f"no parquet shards under {root}/data")

        columns = ["episode_index", "frame_index", "observation.state"]
        frames = pd.concat(
            (pd.read_parquet(path, columns=columns) for path in files),
            ignore_index=True,
        )

        for episode_index, group in frames.groupby("episode_index", sort=True):
            episode_index = int(episode_index)
            if wanted_episodes is not None and episode_index not in wanted_episodes:
                continue
            ordered = group.sort_values("frame_index")
            state = np.stack(
                [np.asarray(row, dtype=np.float64) for row in ordered["observation.state"]]
            )
            if state.ndim != 2 or state.shape[1] < EEF_POSE_DIMS:
                raise ValueError(
                    f"observation.state for episode {episode_index} in {root} has "
                    f"shape {state.shape}; expected [N, >= {EEF_POSE_DIMS}]"
                )

            language = languages.get(episode_index)
            if language is None:
                raise KeyError(
                    f"episode {episode_index} present in {root}/data but absent "
                    "from meta/episodes.jsonl; cannot resolve its task language"
                )

            projection = projector.project(language, state)
            resolved = np.empty(projection.anchor_token.shape, dtype=np.float32)
            episode_max_hold = 0
            for camera_index, camera_name in enumerate(CAMERA_ORDER):
                series, camera_stats = resolve_anchor_series(
                    projection.anchor_token_precise[:, camera_index, :],
                    projection.onscreen[:, camera_index],
                )
                resolved[:, camera_index, :] = series.astype(np.float32)
                stats.observed_frames += int(camera_stats.get("observed_frames", 0))
                stats.held_offscreen_frames += int(
                    camera_stats.get("held_offscreen_frames", 0)
                )
                stats.centre_default_frames += int(
                    camera_stats.get("centre_default_frames", 0)
                )
                stats.per_camera_observed[camera_name] += int(
                    camera_stats.get("observed_frames", 0)
                )
                episode_max_hold = max(
                    episode_max_hold, int(camera_stats.get("max_consecutive_held", 0))
                )

            anchors[(dataset_index, episode_index)] = resolved
            stats.episodes += 1
            stats.frames += int(state.shape[0])
            stats.max_consecutive_held = max(
                stats.max_consecutive_held, episode_max_hold
            )
