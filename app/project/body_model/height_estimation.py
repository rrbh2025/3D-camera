"""Experimental metric height estimation from RGB-D pose observations.

This module deliberately keeps height estimation separate from the trained
body-region checkpoint.  It uses metric keypoints produced from the aligned
depth image and exposes the uncertainty and data quality that support each
estimate.  It is an engineering/research measurement, not a clinical height
measurement.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .body_frame import keypoints_to_camera
from .depth import CameraIntrinsics


HEIGHT_SCHEMA = "human_height_estimate.v1"

_HEAD = (0, 1, 2, 3, 4)
_SHOULDERS = (5, 6)
_HIPS = (11, 12)
_KNEES = (13, 14)
_ANKLES = (15, 16)

_STANDING_TOKENS = (
    "站立",
    "standing",
    "stand",
    "upright",
)
_SEATED_TOKENS = ("坐", "sitting", "seated", "sit")
_LYING_TOKENS = (
    "仰卧",
    "俯卧",
    "侧卧",
    "卧",
    "lying",
    "supine",
    "prone",
    "side",
)


def _as_keypoints(value: Any) -> np.ndarray:
    points = np.asarray(value if value is not None else [], dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("keypoints must have shape [N, 2 or 3]")
    if points.shape[1] == 2:
        points = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1
        )
    return points[:, :3].astype(np.float32, copy=False)


def _finite_vector(value: Any, size: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size != size or not np.all(np.isfinite(array)):
        return None
    return array


def _normalize(value: np.ndarray, eps: float = 1e-6) -> np.ndarray | None:
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < eps:
        return None
    return value / norm


def _posture_group(posture: str | None) -> str:
    text = str(posture or "").strip().lower()
    if any(token in text for token in _STANDING_TOKENS):
        return "standing"
    if any(token in text for token in _SEATED_TOKENS):
        return "seated"
    if any(token in text for token in _LYING_TOKENS):
        return "lying"
    return "unknown"


def _empty_result(
    posture: str | None,
    *,
    status: str = "insufficient_geometry",
    reason: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": HEIGHT_SCHEMA,
        "height_estimate_m": None,
        "height_cm": None,
        "height_lower_m": None,
        "height_upper_m": None,
        "confidence": 0.0,
        "status": status,
        "method": "unavailable",
        "posture": posture,
        "posture_group": _posture_group(posture),
        "vertical_axis_source": "unavailable",
        "head_top_source": "unavailable",
        "foot_ground_source": "unavailable",
        "visible_segment_count": 0,
        "segments_m": {},
        "quality": {
            "depth_valid_fraction": 0.0,
            "depth_median_m": None,
            "depth_iqr_m": None,
            "depth_mad_m": None,
            "metric_keypoint_count": 0,
            "metric_keypoint_coverage": 0.0,
            "segment_coverage": 0.0,
        },
        "prediction_not_ground_truth": True,
        "requires_human_review": True,
    }
    if reason:
        result["reason"] = str(reason)
    return result


def _metric_points_from_body_frame(body_frame: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(body_frame, dict):
        return None
    value = body_frame.get("keypoints_3d_camera")
    try:
        points = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    return points


def _metric_keypoints(
    keypoints: np.ndarray,
    *,
    depth: np.ndarray | None,
    depth_scale: float,
    intrinsics: CameraIntrinsics | dict[str, float] | None,
    body_frame: dict[str, Any] | None,
) -> tuple[np.ndarray, str]:
    points = np.full((len(keypoints), 3), np.nan, dtype=np.float32)
    source = "none"
    frame_points = _metric_points_from_body_frame(body_frame)
    if frame_points is not None:
        count = min(len(points), len(frame_points))
        points[:count] = frame_points[:count]
        source = "body_frame_depth"

    if depth is not None and intrinsics is not None and len(keypoints) > 0:
        try:
            camera_intrinsics = (
                intrinsics
                if isinstance(intrinsics, CameraIntrinsics)
                else CameraIntrinsics.from_mapping(
                    {**intrinsics, "depth_scale": float(depth_scale)}
                )
            )
            projected = keypoints_to_camera(
                keypoints,
                np.asarray(depth),
                camera_intrinsics,
                image_size=(int(np.asarray(depth).shape[1]), int(np.asarray(depth).shape[0])),
                min_depth_m=0.2,
                max_depth_m=10.0,
            )
            missing = ~np.all(np.isfinite(points), axis=1)
            fill = missing & np.all(np.isfinite(projected), axis=1)
            points[fill] = projected[fill]
            if np.any(fill) and source == "body_frame_depth":
                source = "mixed_depth"
            elif np.any(np.all(np.isfinite(projected), axis=1)) and source == "none":
                source = "depth_projected"
        except (TypeError, ValueError, KeyError):
            pass

    # Low-confidence pose points must not contribute metric scale even when a
    # nearby depth pixel happens to be valid.
    if len(keypoints) == len(points):
        low_score = keypoints[:, 2] < 0.2
        points[low_score] = np.nan
    return points, source


def _midpoint(
    points: np.ndarray,
    scores: np.ndarray,
    indices: Iterable[int],
    *,
    min_score: float = 0.2,
) -> tuple[np.ndarray | None, int]:
    selected: list[np.ndarray] = []
    for index in indices:
        if index >= len(points) or index >= len(scores):
            continue
        if scores[index] < min_score or not np.all(np.isfinite(points[index])):
            continue
        selected.append(points[index])
    if not selected:
        return None, 0
    return np.mean(np.stack(selected, axis=0), axis=0).astype(np.float32), len(selected)


def _valid_head_points(points: np.ndarray, scores: np.ndarray) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for index in _HEAD:
        if index < len(points) and index < len(scores):
            if scores[index] >= 0.2 and np.all(np.isfinite(points[index])):
                output.append(points[index])
    return output


def _depth_quality(
    depth: np.ndarray | None,
    depth_scale: float,
    bbox: Iterable[float] | None,
) -> dict[str, Any]:
    empty = {
        "depth_valid_fraction": 0.0,
        "depth_median_m": None,
        "depth_iqr_m": None,
        "depth_mad_m": None,
    }
    if depth is None:
        return empty
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        return empty
    if bbox is None:
        x0, y0, x1, y1 = 0, 0, values.shape[1], values.shape[0]
    else:
        try:
            x0, y0, x1, y1 = [float(item) for item in bbox]
        except (TypeError, ValueError):
            return empty
        x0 = int(np.floor(np.clip(x0, 0, values.shape[1])))
        y0 = int(np.floor(np.clip(y0, 0, values.shape[0])))
        x1 = int(np.ceil(np.clip(x1, 0, values.shape[1])))
        y1 = int(np.ceil(np.clip(y1, 0, values.shape[0])))
    if x1 <= x0 or y1 <= y0:
        return empty
    crop = values[y0:y1, x0:x1] * float(depth_scale)
    finite = np.isfinite(crop)
    valid = finite & (crop >= 0.2) & (crop <= 10.0)
    all_count = int(crop.size)
    valid_values = crop[valid]
    if valid_values.size == 0:
        return empty
    q1, median, q3 = np.percentile(valid_values, [25, 50, 75])
    mad = float(np.median(np.abs(valid_values - median)))
    return {
        "depth_valid_fraction": float(valid_values.size / max(all_count, 1)),
        "depth_median_m": float(median),
        "depth_iqr_m": float(q3 - q1),
        "depth_mad_m": mad,
    }


def _clamp_confidence(value: float) -> float:
    return float(np.clip(value if np.isfinite(value) else 0.0, 0.0, 1.0))


def _finish_result(
    *,
    estimate_m: float,
    uncertainty_m: float,
    confidence: float,
    method: str,
    status: str,
    posture: str | None,
    posture_group: str,
    quality: dict[str, Any],
    segments_m: dict[str, float],
    visible_segment_count: int,
    vertical_axis_source: str,
    head_top_source: str,
    foot_ground_source: str,
    reason: str | None = None,
) -> dict[str, Any]:
    estimate = float(estimate_m)
    uncertainty = float(max(uncertainty_m, 0.005))
    lower = max(0.0, estimate - uncertainty)
    upper = estimate + uncertainty
    output: dict[str, Any] = {
        "schema": HEIGHT_SCHEMA,
        "height_estimate_m": estimate,
        "height_cm": estimate * 100.0,
        "height_lower_m": lower,
        "height_upper_m": upper,
        "height_lower_cm": lower * 100.0,
        "height_upper_cm": upper * 100.0,
        "uncertainty_m": uncertainty,
        "confidence": _clamp_confidence(confidence),
        "status": status,
        "method": method,
        "posture": posture,
        "posture_group": posture_group,
        "vertical_axis_source": vertical_axis_source,
        "head_top_source": head_top_source,
        "foot_ground_source": foot_ground_source,
        "visible_segment_count": int(visible_segment_count),
        "segments_m": {key: float(value) for key, value in segments_m.items()},
        "quality": quality,
        "prediction_not_ground_truth": True,
        "requires_human_review": True,
    }
    if reason:
        output["reason"] = str(reason)
    return output


def estimate_height(
    *,
    keypoints: Any,
    depth: np.ndarray | None = None,
    depth_scale: float = 1.0,
    intrinsics: CameraIntrinsics | dict[str, float] | None = None,
    selected_bbox: Iterable[float] | None = None,
    body_frame: dict[str, Any] | None = None,
    regions: Iterable[dict[str, Any]] | None = None,
    posture: str | None = None,
) -> dict[str, Any]:
    """Estimate a person's height for standing and non-standing poses.

    Standing uses the metric torso axis to estimate head-top to foot-bottom
    extent. Other supported postures sum metric head, torso, and leg segments,
    which is invariant to folding the body into a seated or lying pose. Both
    paths require metric depth geometry; a 2D pose alone cannot determine an
    absolute height.
    """
    try:
        points_2d = _as_keypoints(keypoints)
    except (TypeError, ValueError) as error:
        return _empty_result(posture, reason=f"invalid_keypoints: {error}")
    if len(points_2d) == 0:
        return _empty_result(posture, reason="no_keypoints")

    scores = np.nan_to_num(points_2d[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
    metric, metric_source = _metric_keypoints(
        points_2d,
        depth=depth,
        depth_scale=float(depth_scale),
        intrinsics=intrinsics,
        body_frame=body_frame,
    )
    quality = _depth_quality(depth, float(depth_scale), selected_bbox)
    metric_valid = np.all(np.isfinite(metric), axis=1) & (scores >= 0.2)
    metric_count = int(metric_valid.sum())
    quality["metric_keypoint_count"] = metric_count
    quality["metric_keypoint_coverage"] = float(metric_count / max(min(len(points_2d), 17), 1))

    group = _posture_group(posture)
    shoulder, shoulder_count = _midpoint(metric, scores, _SHOULDERS)
    hip, hip_count = _midpoint(metric, scores, _HIPS)
    head_points = _valid_head_points(metric, scores)
    knee_points = [
        metric[index]
        for index in _KNEES
        if index < len(metric)
        and index < len(scores)
        and scores[index] >= 0.2
        and np.all(np.isfinite(metric[index]))
    ]
    ankle_points = [
        metric[index]
        for index in _ANKLES
        if index < len(metric)
        and index < len(scores)
        and scores[index] >= 0.2
        and np.all(np.isfinite(metric[index]))
    ]
    if shoulder is None or hip is None or not head_points:
        return _empty_result(
            posture,
            reason="missing_head_shoulder_or_hip_metric_geometry",
        ) | {"quality": quality}

    axis = _normalize(shoulder - hip)
    if axis is None:
        return _empty_result(posture, reason="invalid_body_axis") | {"quality": quality}

    # Body-axis direct measurement is only used for an explicitly standing
    # posture. For other postures, a folded skeleton must be measured by path
    # length rather than by its camera-space bounding-box extent.
    if group == "standing" and ankle_points:
        origin = hip
        head_projection = max(float(np.dot(point - origin, axis)) for point in head_points)
        ankle_projection = min(float(np.dot(point - origin, axis)) for point in ankle_points)
        head_extension = 0.08
        foot_extension = 0.08
        estimate = head_projection + head_extension - (ankle_projection - foot_extension)
        if np.isfinite(estimate) and 0.8 <= estimate <= 2.5:
            segment_count = shoulder_count + hip_count + len(head_points) + len(ankle_points)
            quality["segment_coverage"] = float(min(segment_count / 8.0, 1.0))
            depth_fraction = float(quality.get("depth_valid_fraction", 0.0) or 0.0)
            depth_factor = 0.55 + 0.45 * min(depth_fraction / 0.75, 1.0)
            metric_factor = min(metric_count / 10.0, 1.0)
            confidence = 0.82 * depth_factor * (0.55 + 0.45 * metric_factor)
            uncertainty = (
                0.025
                + min(float(quality.get("depth_iqr_m") or 0.0), 0.15) * 0.45
                + (1.0 - min(metric_factor, 1.0)) * 0.06
            )
            return _finish_result(
                estimate_m=estimate,
                uncertainty_m=uncertainty,
                confidence=confidence,
                method="direct_metric",
                status="direct_metric_available",
                posture=posture,
                posture_group=group,
                quality=quality,
                segments_m={
                    "head_top_to_foot_bottom": estimate,
                    "head_extension": head_extension,
                    "foot_extension": foot_extension,
                },
                visible_segment_count=segment_count,
                vertical_axis_source="body_axis_from_metric_keypoints",
                head_top_source="head_keypoint_plus_anatomical_extension",
                foot_ground_source="ankle_keypoint_plus_foot_extension",
                reason=(
                    "body-axis metric estimate; ground plane and shoe height are not calibrated"
                ),
            )

    # Pose-conditioned path. Each leg is measured independently when possible;
    # averaging both sides suppresses a single bad depth hole or occluded limb.
    head_reference = np.mean(np.stack(head_points, axis=0), axis=0)
    head_to_shoulder = float(np.linalg.norm(head_reference - shoulder)) + 0.10
    torso = float(np.linalg.norm(shoulder - hip))
    leg_values: list[float] = []
    leg_names: list[str] = []
    for side, hip_index, knee_index, ankle_index in (
        ("left", 11, 13, 15),
        ("right", 12, 14, 16),
    ):
        if max(hip_index, knee_index, ankle_index) >= len(metric):
            continue
        if not all(
            scores[index] >= 0.2 and np.all(np.isfinite(metric[index]))
            for index in (hip_index, knee_index, ankle_index)
        ):
            continue
        thigh = float(np.linalg.norm(metric[hip_index] - metric[knee_index]))
        shin = float(np.linalg.norm(metric[knee_index] - metric[ankle_index]))
        leg_values.append(thigh + shin + 0.08)
        leg_names.append(f"{side}_leg")
    if not leg_values:
        return _empty_result(
            posture,
            reason="no_complete_metric_leg_segment",
        ) | {"quality": quality}

    leg = float(np.mean(leg_values))
    estimate = head_to_shoulder + torso + leg
    if not np.isfinite(estimate) or not 0.8 <= estimate <= 2.5:
        return _empty_result(
            posture,
            status="invalid_geometry",
            reason=f"height_out_of_expected_range: {estimate!r}",
        ) | {"quality": quality}

    segments = {
        "head_to_shoulder": head_to_shoulder,
        "torso": torso,
        "leg_mean": leg,
        "foot_extension": 0.08,
    }
    segments.update({name: value for name, value in zip(leg_names, leg_values)})
    visible_segment_count = 3 + len(leg_values)
    quality["segment_coverage"] = float(visible_segment_count / 5.0)
    depth_fraction = float(quality.get("depth_valid_fraction", 0.0) or 0.0)
    depth_factor = 0.45 + 0.55 * min(depth_fraction / 0.75, 1.0)
    metric_factor = min(metric_count / 12.0, 1.0)
    confidence = 0.66 * depth_factor * (0.55 + 0.45 * metric_factor)
    uncertainty = (
        0.07
        + min(float(quality.get("depth_iqr_m") or 0.0), 0.20) * 0.65
        + (1.0 - min(metric_factor, 1.0)) * 0.10
        + (0.04 if group == "unknown" else 0.0)
    )
    return _finish_result(
        estimate_m=estimate,
        uncertainty_m=uncertainty,
        confidence=confidence,
        method="pose_conditioned",
        status="pose_conditioned_available",
        posture=posture,
        posture_group=group,
        quality=quality,
        segments_m=segments,
        visible_segment_count=visible_segment_count,
        vertical_axis_source="skeleton_segment_lengths",
        head_top_source="head_keypoint_plus_anatomical_extension",
        foot_ground_source="ankle_keypoint_plus_foot_extension",
        reason=(
            f"{metric_source}; folded-pose skeleton path estimate, not a direct ground measurement"
        ),
    )


__all__ = ["HEIGHT_SCHEMA", "estimate_height"]
