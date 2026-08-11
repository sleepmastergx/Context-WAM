# Author: Rui Heng Yang
"""Fail-closed EEF geometry-identity comparison between a model and an artifact.

A checkpoint records the calibration it was trained against in
``mot.eef_geometry_identity`` (plan Section 20.1 / C.5). Every consumer that
resolves anchors independently -- the training dataset's anchor index, the
sequential evaluator's live resolver, the shared-model actors -- loads its own
calibration artifact. Nothing forces those to be the same bytes: a relative path
resolves against a different root per process, and an artifact can be
regenerated between training and evaluation.

The mismatch is silent by construction. Anchors resolved from another
calibration are ordinary finite token coordinates; they merely place the spatial
origin somewhere else, so training would stamp checkpoints with calibration A
while consuming anchors from calibration B, and evaluation would load cleanly
against a third.

These helpers are the comparison, and they deliberately import nothing from
``fastwam.models`` -- geometry is imported *by* the models -- so the trainer and
both LIBERO evaluators can share one copy of the rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

EEF_ROPE_MODES = frozenset({"ee_rope", "exclusive_ee_rope"})
"""The RoPE modes that consume projected EEF anchors.

Mirrors ``MoTDecoupledActionAlignedVideoRoPE.ee_rope_modes``; the literal is
repeated rather than imported because importing the model package from
``fastwam.geometry`` would invert the dependency direction.
"""


def eef_rope_mode(model: Any) -> str | None:
    """Return the model's active EEF RoPE mode, or ``None`` for every other mode.

    Args:
        model: A FastWAM model (or any proxy forwarding attribute access to one).

    Returns:
        ``"ee_rope"`` / ``"exclusive_ee_rope"`` when that mode is active,
        otherwise ``None``. ``aligned_3d`` and every legacy mode return ``None``,
        which is what keeps the callers' guards inert for them.
    """
    mot = getattr(model, "mot", None)
    if mot is None:
        return None
    mode = getattr(mot, "new_fused_kv_rope_mode", None)
    if mode is None:
        return None
    mode = str(mode)
    return mode if mode in EEF_ROPE_MODES else None


def model_calibration_digest(model: Any) -> str:
    """Return the calibration digest recorded in the model's geometry identity.

    Args:
        model: A model whose ``mot`` carries ``eef_geometry_identity``.

    Returns:
        The recorded ``calibration_digest``.

    Raises:
        RuntimeError: If the identity or its digest is absent. An EEF-mode model
            cannot be constructed without a complete identity, so absence here
            means the object is not the model the caller believes it to be.
    """
    identity = getattr(getattr(model, "mot", None), "eef_geometry_identity", None)
    if identity is None:
        raise RuntimeError(
            "model records no `eef_geometry_identity`, so the calibration its "
            "weights were trained against cannot be verified. An EEF RoPE mode "
            "cannot be constructed without one (plan Section 20.1)."
        )
    digest = identity.get("calibration_digest")
    if not digest:
        raise RuntimeError(
            "model's `eef_geometry_identity` carries no `calibration_digest`: "
            f"{sorted(identity)}"
        )
    return str(digest)


def assert_calibration_parity(
    model: Any,
    *,
    digest: str,
    source_label: str,
    source_path: str | Path | None = None,
) -> None:
    """Refuse to proceed unless an anchor source uses the model's calibration.

    Call this once per anchor source, after checking that
    :func:`eef_rope_mode` is not ``None`` -- non-EEF modes resolve no anchors and
    must be left untouched.

    Args:
        model: The model that will consume the anchors.
        digest: Content digest of the calibration the source actually loaded.
        source_label: Human-readable name of that source, used in the error.
        source_path: Artifact path the source loaded, when known.

    Raises:
        RuntimeError: If the digests differ, or the model records no identity.
    """
    model_digest = model_calibration_digest(model)
    source_digest = str(digest)
    if source_digest == model_digest:
        logger.info(
            "EEF calibration parity OK: %s (%s) matches the model's geometry "
            "identity, digest %s",
            source_label,
            source_path,
            model_digest[:12],
        )
        return

    mode = eef_rope_mode(model)
    model_path = getattr(model, "eef_calibration_path", None)
    raise RuntimeError(
        f"EEF calibration mismatch: {source_label} resolved anchors from a "
        "different calibration than the model's geometry identity records. "
        f"Model new_fused_kv_rope_mode={mode!r} "
        f"model.eef_calibration_path={model_path!r} "
        f"calibration_digest={model_digest}; {source_label} "
        f"path={str(source_path)!r} calibration_digest={source_digest}. "
        "Anchors from another calibration are ordinary finite token "
        "coordinates, so the run would consume one spatial origin while "
        "recording another, with no other symptom. Point both settings at the "
        "same artifact, or regenerate the checkpoint (plan Section 20.1)."
    )
