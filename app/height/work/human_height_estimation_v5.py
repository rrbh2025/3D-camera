"""Experimental Gemini RGB-D standing-height refinement (v5).

This module deliberately wraps the frozen v4 estimator instead of modifying it.
For an upright person with a reliable floor-plane normal and confirmed soles,
v5 uses the crown-to-sole projection along that normal.  This direct endpoint
measurement is less sensitive to a small frame-local floor-plane offset than
v4's head-to-plane value.  COCO skeletal chains remain diagnostic only: their
numeric output never changes a v5 stature value.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

from work.human_height_estimation import estimate_human_height as _estimate_v4


SCHEMA = "gemini_human_height_estimate.v5"
_DIRECT_MIN_HEIGHT_CM = 65.0
_DIRECT_MAX_HEIGHT_CM = 250.0
_MIN_FLOOR_CONFIDENCE = 0.45
_MIN_NORMAL_Y_COMPONENT = 0.95
_MIN_BODY_AXIS_ALIGNMENT = 0.70
_MAX_FLOOR_RESIDUAL_IQR_M = 0.035
_MAX_SOLE_TO_PLANE_M = 0.08
_MAX_ENDPOINT_TO_PLANE_DIFFERENCE_M = 0.06


def _finite_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _is_standing(posture: str | None) -> bool:
    normalized = str(posture or "").strip().lower()
    return "站立" in normalized or "standing" in normalized or normalized == "stand"


def _direct_geometry(
    quality: Mapping[str, Any] | None,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    """Extract only endpoint and plane evidence from a frozen v4 result."""
    if not isinstance(quality, Mapping):
        return None, {"reason": "missing_measurement_quality"}
    candidates = quality.get("candidate_heights_cm")
    floor = quality.get("ground_plane")
    if not isinstance(candidates, Mapping) or not isinstance(floor, Mapping):
        return None, {"reason": "missing_direct_geometry"}

    projected_cm = _finite_number(candidates.get("top_to_sole_vertical_projection"))
    floor_cm = _finite_number(candidates.get("head_to_ground_plane"))
    confidence = _finite_number(quality.get("direct_metric_floor_confidence"))
    normal_y = _finite_number(floor.get("normal_y_component"))
    axis_alignment = _finite_number(floor.get("body_axis_alignment"))
    residual_iqr_m = _finite_number(floor.get("residual_iqr_m"))
    sole_gap_m = _finite_number(floor.get("median_sole_to_plane_m"))
    if (
        floor.get("status") != "ok"
        or projected_cm is None
        or floor_cm is None
        or confidence is None
        or normal_y is None
        or axis_alignment is None
        or residual_iqr_m is None
        or sole_gap_m is None
    ):
        return None, {"reason": "incomplete_direct_geometry"}

    endpoint_plane_difference_m = abs(projected_cm - floor_cm) / 100.0
    checks = {
        "projected_height_in_range": _DIRECT_MIN_HEIGHT_CM <= projected_cm <= _DIRECT_MAX_HEIGHT_CM,
        "floor_confidence": confidence >= _MIN_FLOOR_CONFIDENCE,
        "floor_normal": normal_y >= _MIN_NORMAL_Y_COMPONENT,
        "body_axis_alignment": axis_alignment >= _MIN_BODY_AXIS_ALIGNMENT,
        "floor_residual": residual_iqr_m <= _MAX_FLOOR_RESIDUAL_IQR_M,
        "sole_plane_proximity": sole_gap_m <= _MAX_SOLE_TO_PLANE_M,
        "endpoint_plane_consistency": (
            endpoint_plane_difference_m <= _MAX_ENDPOINT_TO_PLANE_DIFFERENCE_M
        ),
    }
    diagnostics = {
        "checks": checks,
        "floor_height_cm": floor_cm,
        "endpoint_projected_height_cm": projected_cm,
        "endpoint_plane_difference_cm": endpoint_plane_difference_m * 100.0,
        "floor_confidence": confidence,
        "floor_normal_y_component": normal_y,
        "body_axis_alignment": axis_alignment,
        "floor_residual_iqr_cm": residual_iqr_m * 100.0,
        "median_sole_to_plane_cm": sole_gap_m * 100.0,
    }
    if not all(checks.values()):
        return None, diagnostics
    return {
        "projected_cm": projected_cm,
        "floor_cm": floor_cm,
        "confidence": confidence,
        "endpoint_plane_difference_cm": endpoint_plane_difference_m * 100.0,
    }, diagnostics


def _refine_standing_v4_result(v4_result: Mapping[str, Any]) -> dict[str, Any]:
    """Turn a v4 direct-standing result into a v5 endpoint-only result."""
    output = copy.deepcopy(dict(v4_result))
    output["schema"] = SCHEMA
    quality = output.get("measurement_quality")
    direct, diagnostics = _direct_geometry(quality if isinstance(quality, Mapping) else None)
    if isinstance(quality, dict):
        quality["v5_direct_endpoint_selection"] = {
            "method": "floor_normal_endpoint_projection_no_skeleton_fusion",
            **diagnostics,
        }

    # v4 has already rejected incomplete people, missing feet, and unreliable
    # floor geometry.  The one v4-only rejection we can safely recover from is
    # skeletal disagreement, because the v5 direct measurement does not use a
    # skeletal length in its value.
    if direct is None or output.get("status") not in {"ok", "height_candidates_disagree"}:
        return output

    skeleton = quality.get("skeleton_candidate") if isinstance(quality, Mapping) else None
    skeleton_cm = (
        _finite_number(skeleton.get("height_cm"))
        if isinstance(skeleton, Mapping) and skeleton.get("status") == "ok"
        else None
    )
    skeleton_disagreement_cm = (
        abs(float(direct["projected_cm"]) - skeleton_cm)
        if skeleton_cm is not None
        else None
    )
    if isinstance(quality, dict):
        quality["candidate_heights_cm"]["final_consensus_v4"] = quality[
            "candidate_heights_cm"
        ].get("final_consensus")
        quality["candidate_heights_cm"]["final_v5_endpoint_projection"] = float(
            direct["projected_cm"]
        )
        quality["candidate_heights_cm"]["final_consensus"] = float(
            direct["projected_cm"]
        )
        quality["skeleton_diagnostic"] = {
            "numeric_used_for_height": False,
            "height_cm": skeleton_cm,
            "disagreement_from_direct_endpoint_cm": skeleton_disagreement_cm,
            "status": skeleton.get("status") if isinstance(skeleton, Mapping) else "unavailable",
        }

    base_uncertainty_cm = _finite_number(
        quality.get("direct_metric_floor_uncertainty_cm")
        if isinstance(quality, Mapping)
        else None
    )
    base_uncertainty_cm = max(3.0, base_uncertainty_cm or 8.0)
    endpoint_plane_difference_cm = float(direct["endpoint_plane_difference_cm"])
    uncertainty_cm = float(
        np.sqrt(base_uncertainty_cm**2 + (0.50 * endpoint_plane_difference_cm) ** 2)
    )
    confidence = float(
        np.clip(
            float(direct["confidence"])
            * np.exp(-endpoint_plane_difference_cm / 10.0),
            0.0,
            0.96,
        )
    )
    output.update(
        {
            "status": "ok",
            "height_cm": float(direct["projected_cm"]),
            "height_lower_cm": float(direct["projected_cm"] - uncertainty_cm),
            "height_upper_cm": float(direct["projected_cm"] + uncertainty_cm),
            "confidence": confidence,
            "method": "direct_metric_floor_normal_endpoint_projection",
        }
    )
    return output


def refine_v4_result(
    v4_result: Mapping[str, Any],
    *,
    posture: str | None,
) -> dict[str, Any]:
    """Apply v5 selection to one already-computed v4 result.

    This makes offline ablations fair: v4 and v5 use exactly the same RGB-D
    masks, crown point, feet and floor-plane fit for each frame.
    """
    if not _is_standing(posture):
        output = copy.deepcopy(v4_result)
        output["schema"] = SCHEMA
        return output
    return _refine_standing_v4_result(v4_result)


def estimate_human_height(**kwargs: Any) -> dict[str, Any]:
    """Estimate height with frozen v4 preprocessing and v5 standing selection.

    Non-standing output remains exactly the v4 result apart from its schema,
    because this experiment only evaluates a direct upright measurement.
    """
    return refine_v4_result(_estimate_v4(**kwargs), posture=kwargs.get("posture"))
