"""Gemini RGB-D human-height estimates for the remote live service.

The implementation intentionally has no manual-height input.  A standing
measurement is emitted only when a complete person is visible in the aligned
depth image.  Other postures use a depth-backed skeletal-length prediction and
are marked as such so they are not confused with a direct stature measurement.
"""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np


SCHEMA = "gemini_human_height_estimate.v4"
DEPTH_CORRECTION_SCHEMA = "gemini_height_depth_correction_request.v1"
_MIN_DEPTH_M = 0.2
_MAX_DEPTH_M = 10.0
_KEYPOINT_SCORE = 0.25
_ANKLE_INDICES = (15, 16)
_HEIGHT_MIN_M = 0.65
_HEIGHT_MAX_M = 2.50
_FOOT_DEPTH_TOLERANCE_M = 0.28
_MAX_CANDIDATE_DISAGREEMENT_M = 0.16
_LYING_FOOT_EXTENSION_M = 0.08
_LYING_MIN_STRAIGHTNESS = 0.88


def _base_result(
    posture: str | None,
    depth_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "unavailable",
        "height_cm": None,
        "height_lower_cm": None,
        "height_upper_cm": None,
        "confidence": 0.0,
        "method": None,
        "posture": str(posture or "未填写"),
        "source": "gemini_aligned_depth_and_intrinsics",
        "depth_calibration": dict(
            depth_calibration
            or {
                "provided": False,
                "applied": False,
                "reason": "not_provided",
            }
        ),
        "manual_reference_height_required": False,
        "prediction_not_ground_truth": True,
        "clinical_use": False,
        "requires_human_review": True,
    }


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _prepare_depth_calibration(
    source: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], float, float, float]:
    """Validate a local fixed-station depth correction before applying it."""
    if not isinstance(source, Mapping):
        return {"provided": False, "applied": False, "reason": "not_provided"}, 1.0, 0.0, 0.0
    state: dict[str, Any] = {
        "provided": True,
        "applied": False,
        "profile_id": str(source.get("profile_id") or "") or None,
        "station_id": str(source.get("station_id") or "") or None,
        "schema": str(source.get("schema") or ""),
    }
    if source.get("schema") != DEPTH_CORRECTION_SCHEMA:
        state["reason"] = "unsupported_schema"
        return state, 1.0, 0.0, 0.0
    if source.get("manual_person_height_used") is not False:
        state["reason"] = "manual_person_height_not_allowed"
        return state, 1.0, 0.0, 0.0
    if source.get("model") != "z_affine":
        state["reason"] = "unsupported_depth_model"
        return state, 1.0, 0.0, 0.0
    scale = _finite_float(source.get("scale"))
    offset_m = _finite_float(source.get("offset_m"))
    fit_rmse_m = _finite_float(source.get("fit_rmse_m"))
    if (
        scale is None
        or offset_m is None
        or fit_rmse_m is None
        or not 0.85 <= scale <= 1.15
        or abs(offset_m) > 0.15
        or not 0.0 <= fit_rmse_m <= 0.04
    ):
        state["reason"] = "invalid_correction_values"
        return state, 1.0, 0.0, 0.0
    state.update(
        {
            "applied": True,
            "reason": "validated_fixed_station_profile",
            "model": "z_affine",
            "scale": scale,
            "offset_m": offset_m,
            "fit_rmse_m": fit_rmse_m,
            "manual_person_height_used": False,
        }
    )
    return state, scale, offset_m, fit_rmse_m


def _correct_depth_value(value: float | None, scale: float, offset_m: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    corrected = float(value) * float(scale) + float(offset_m)
    return corrected if np.isfinite(corrected) else None


def _intrinsics(values: dict[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        fx = float(values["fx"])
        fy = float(values["fy"])
        cx = float(values["cx"])
        cy = float(values["cy"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        return None
    return fx, fy, cx, cy


def _is_standing(posture: str | None) -> bool:
    normalized = str(posture or "").strip().lower()
    return "站立" in normalized or "standing" in normalized or normalized == "stand"


def _is_lying(posture: str | None) -> bool:
    normalized = str(posture or "").strip().lower()
    return any(
        token in normalized
        for token in ("仰卧", "俯卧", "侧卧", "平躺", "lying", "supine", "prone")
    )


def _project_to_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Remove the component normal to a support plane."""
    value = np.asarray(vector, dtype=np.float32)
    unit_normal = np.asarray(normal, dtype=np.float32)
    return value - float(np.dot(value, unit_normal)) * unit_normal


def _validated_support_plane(
    source: Mapping[str, Any] | None,
) -> tuple[np.ndarray, float, dict[str, Any]] | None:
    """Read an optional calibrated bed/support plane in camera coordinates."""
    if not isinstance(source, Mapping):
        return None
    try:
        normal = np.asarray(source["normal_camera"], dtype=np.float32)
        offset = float(source["plane_offset_m"])
    except (KeyError, TypeError, ValueError):
        return None
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isfinite(offset):
        return None
    magnitude = float(np.linalg.norm(normal))
    if magnitude <= 1e-6:
        return None
    normal = normal / magnitude
    offset /= magnitude
    return normal, offset, {
        "source": str(source.get("source") or "provided_support_plane"),
        "normal_camera": normal.tolist(),
        "plane_offset_m": float(offset),
        "provided": True,
    }


def _bbox_from_result(result: dict[str, Any]) -> tuple[float, float, float, float] | None:
    candidate = result.get("selected_bbox_xyxy")
    if not isinstance(candidate, (list, tuple)) or len(candidate) != 4:
        tracking = result.get("tracking") or {}
        selected = tracking.get("selected_index")
        for item in tracking.get("candidates") or []:
            if isinstance(item, dict) and item.get("index") == selected:
                candidate = item.get("bbox_xyxy")
                break
    if not isinstance(candidate, (list, tuple)) or len(candidate) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in candidate)
    except (TypeError, ValueError):
        return None
    if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _target_depth(result: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    tracking = result.get("tracking") or {}
    selected = tracking.get("selected_index")
    for item in tracking.get("candidates") or []:
        if not isinstance(item, dict) or item.get("index") != selected:
            continue
        try:
            depth = float(item.get("selection_depth_m", item.get("depth_m")))
        except (TypeError, ValueError):
            depth = None
        try:
            iqr = float(item.get("depth_iqr_m"))
        except (TypeError, ValueError):
            iqr = None
        try:
            valid_fraction = float(item.get("depth_valid_fraction"))
        except (TypeError, ValueError):
            valid_fraction = None
        if depth is not None and np.isfinite(depth) and _MIN_DEPTH_M <= depth <= _MAX_DEPTH_M:
            return depth, iqr, valid_fraction
    return None, None, None


def _keypoints(result: dict[str, Any]) -> np.ndarray:
    points = np.asarray(result.get("selected_keypoints") or [], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    return points[:, :3]


def _in_frame_keypoints(
    keypoints: np.ndarray,
    image_shape: tuple[int, int],
    *,
    minimum_score: float = _KEYPOINT_SCORE,
) -> list[tuple[int, np.ndarray]]:
    """Return only confident keypoints whose pixel coordinates are usable."""
    if keypoints.ndim != 2 or keypoints.shape[1] < 3:
        return []
    image_height, image_width = image_shape
    output: list[tuple[int, np.ndarray]] = []
    for index, point in enumerate(keypoints):
        if len(point) < 3 or float(point[2]) < minimum_score:
            continue
        x, y = float(point[0]), float(point[1])
        if not (np.isfinite([x, y]).all() and 0.0 <= x < image_width and 0.0 <= y < image_height):
            continue
        output.append((index, point))
    return output


def _pose_depth_reference(
    depth_m: np.ndarray,
    keypoints: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    *,
    tracker_depth_m: float | None,
) -> tuple[float | None, dict[str, Any]]:
    """Estimate selected-person depth while rejecting a background tracker value."""
    preferred_indices = (0, 5, 6, 11, 12, 13, 14, 15, 16)
    points: list[np.ndarray] = []
    used_indices: list[int] = []
    for index in preferred_indices:
        if index >= len(keypoints):
            continue
        point = _point_from_keypoint(depth_m, keypoints[index], intrinsics)
        if point is None:
            continue
        points.append(point)
        used_indices.append(index)
    diagnostic: dict[str, Any] = {
        "tracker_depth_m": tracker_depth_m,
        "pose_depth_count": int(len(points)),
        "pose_depth_keypoint_indices": used_indices,
    }
    pose_depth: float | None = None
    if points:
        values = np.asarray([point[2] for point in points], dtype=np.float32)
        median = float(np.median(values))
        robust = np.abs(values - median) <= max(0.10, 0.18 * median)
        if int(robust.sum()) >= 2:
            median = float(np.median(values[robust]))
            values = values[robust]
        pose_depth = median if _MIN_DEPTH_M <= median <= _MAX_DEPTH_M else None
        diagnostic["pose_depth_median_m"] = pose_depth
        diagnostic["pose_depth_iqr_m"] = (
            float(np.diff(np.percentile(values, [25.0, 75.0]))[0])
            if len(values) >= 2
            else None
        )
    tracker_valid = (
        tracker_depth_m is not None
        and np.isfinite(tracker_depth_m)
        and _MIN_DEPTH_M <= float(tracker_depth_m) <= _MAX_DEPTH_M
    )
    if pose_depth is None and tracker_valid:
        diagnostic["source"] = "tracker_depth"
        return float(tracker_depth_m), diagnostic
    if pose_depth is None:
        diagnostic["source"] = "unavailable"
        return None, diagnostic
    if not tracker_valid:
        diagnostic["source"] = "pose_keypoint_depth"
        return pose_depth, diagnostic
    disagreement = abs(float(tracker_depth_m) - pose_depth)
    diagnostic["tracker_pose_disagreement_m"] = disagreement
    if disagreement > max(0.30, 0.24 * pose_depth):
        diagnostic["source"] = "pose_keypoint_depth_after_tracker_rejection"
        return pose_depth, diagnostic
    diagnostic["source"] = "tracker_and_pose_depth"
    return float(0.5 * (float(tracker_depth_m) + pose_depth)), diagnostic


def _point_from_keypoint(
    depth_m: np.ndarray,
    keypoint: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> np.ndarray | None:
    if len(keypoint) < 3 or float(keypoint[2]) < _KEYPOINT_SCORE:
        return None
    height, width = depth_m.shape
    u = int(round(float(keypoint[0])))
    v = int(round(float(keypoint[1])))
    if not (0 <= u < width and 0 <= v < height):
        return None
    radius = 7
    x0, x1 = max(0, u - radius), min(width, u + radius + 1)
    y0, y1 = max(0, v - radius), min(height, v + radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch >= _MIN_DEPTH_M) & (patch <= _MAX_DEPTH_M)
    if int(valid.sum()) < 6:
        return None
    values = patch[valid]
    median = float(np.median(values))
    robust = valid & (np.abs(patch - median) <= max(0.03, 0.08 * median))
    if int(robust.sum()) < 4:
        robust = valid
    ys, xs = np.where(robust)
    z = patch[robust]
    px = xs.astype(np.float32) + float(x0)
    py = ys.astype(np.float32) + float(y0)
    fx, fy, cx, cy = intrinsics
    return np.asarray(
        [
            float(np.median((px - cx) * z / fx)),
            float(np.median((py - cy) * z / fy)),
            float(np.median(z)),
        ],
        dtype=np.float32,
    )


def _fill_small_depth_holes(
    depth_m: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Fill only small enclosed invalid-depth holes.

    Large missing regions are left invalid because filling them could bridge a
    person and the background.  The limited closing operation targets the
    small holes commonly seen around hair, ankles, and clothing edges.
    """
    if not isinstance(depth_m, np.ndarray) or depth_m.ndim != 2:
        return depth_m, 0
    valid = np.isfinite(depth_m) & (depth_m >= _MIN_DEPTH_M) & (depth_m <= _MAX_DEPTH_M)
    if int(valid.sum()) < 20:
        return depth_m, 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    holes = closed & ~valid
    if not holes.any():
        return depth_m, 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        holes.astype(np.uint8), connectivity=8
    )
    selected = np.zeros_like(holes, dtype=np.uint8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area <= 24 and width <= 8 and height <= 8:
            selected[labels == label] = 255
    if not np.any(selected):
        return depth_m, 0

    source = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32,
        copy=False,
    )
    try:
        inpainted = cv2.inpaint(source, selected, 3.0, cv2.INPAINT_NS)
    except cv2.error:
        return depth_m, 0
    fill_mask = selected > 0
    usable = fill_mask & np.isfinite(inpainted)
    usable &= (inpainted >= _MIN_DEPTH_M) & (inpainted <= _MAX_DEPTH_M)
    if not usable.any():
        return depth_m, 0
    output = depth_m.copy()
    output[usable] = inpainted[usable]
    return output, int(usable.sum())


def _robust_point_from_pixels(
    depth_m: np.ndarray,
    image_y: np.ndarray,
    image_x: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    *,
    min_count: int = 4,
) -> tuple[np.ndarray, float, int] | None:
    """Project a pixel set after rejecting local depth outliers."""
    if len(image_x) == 0 or len(image_y) == 0:
        return None
    image_y = np.asarray(image_y, dtype=np.int32)
    image_x = np.asarray(image_x, dtype=np.int32)
    height, width = depth_m.shape
    inside = (
        (image_x >= 0)
        & (image_x < width)
        & (image_y >= 0)
        & (image_y < height)
    )
    if int(inside.sum()) < min_count:
        return None
    image_x = image_x[inside]
    image_y = image_y[inside]
    values = depth_m[image_y, image_x].astype(np.float32, copy=False)
    valid = np.isfinite(values) & (values >= _MIN_DEPTH_M) & (values <= _MAX_DEPTH_M)
    if int(valid.sum()) < min_count:
        return None
    image_x = image_x[valid]
    image_y = image_y[valid]
    values = values[valid]
    median = float(np.median(values))
    robust = np.abs(values - median) <= max(0.03, 0.08 * median)
    if int(robust.sum()) >= min_count:
        image_x = image_x[robust]
        image_y = image_y[robust]
        values = values[robust]
    fx, fy, cx, cy = intrinsics
    point = np.asarray(
        [
            float(np.median((image_x - cx) * values / fx)),
            float(np.median((image_y - cy) * values / fy)),
            float(np.median(values)),
        ],
        dtype=np.float32,
    )
    q1, q3 = np.percentile(values, [25.0, 75.0])
    return point, float(q3 - q1), int(len(values))


def _rgb_head_crown_candidate(
    color_bgr: np.ndarray | None,
    keypoints: np.ndarray,
) -> dict[str, Any] | None:
    """Find a visible hair/skull boundary around the selected face.

    COCO keypoints stop at the face, while Gemini often has invalid depth on
    dark hair. This candidate uses RGB only for the image endpoint; its metric
    depth is supplied later by valid facial depth samples. It is deliberately
    rejected when the boundary is too close to a frame edge or is not a
    plausible component around the face.
    """
    if (
        not isinstance(color_bgr, np.ndarray)
        or color_bgr.ndim != 3
        or color_bgr.shape[2] < 3
        or keypoints.ndim != 2
        or keypoints.shape[1] < 3
    ):
        return None
    image_height, image_width = color_bgr.shape[:2]
    facial = [
        keypoints[index]
        for index in range(min(5, len(keypoints)))
        if float(keypoints[index, 2]) >= _KEYPOINT_SCORE
        and 0.0 <= float(keypoints[index, 0]) < image_width
        and 0.0 <= float(keypoints[index, 1]) < image_height
    ]
    if len(facial) < 2:
        return None
    face_xy = np.asarray([[float(point[0]), float(point[1])] for point in facial])
    face_top = float(np.min(face_xy[:, 1]))
    face_center_x = float(np.median(face_xy[:, 0]))

    ears = [
        keypoints[index]
        for index in (3, 4)
        if index < len(keypoints)
        and float(keypoints[index, 2]) >= _KEYPOINT_SCORE
        and 2.0 <= float(keypoints[index, 0]) < image_width - 2.0
        and 0.0 <= float(keypoints[index, 1]) < image_height
    ]
    eyes = [
        keypoints[index]
        for index in (1, 2)
        if index < len(keypoints)
        and float(keypoints[index, 2]) >= _KEYPOINT_SCORE
        and 0.0 <= float(keypoints[index, 0]) < image_width
        and 0.0 <= float(keypoints[index, 1]) < image_height
    ]
    if len(ears) == 2:
        head_width_px = float(np.linalg.norm(ears[0][:2] - ears[1][:2]))
    elif len(eyes) == 2:
        head_width_px = float(np.linalg.norm(eyes[0][:2] - eyes[1][:2])) * 2.25
    else:
        head_width_px = float(np.ptp(face_xy[:, 0])) * 1.65
    head_width_px = float(np.clip(head_width_px, 18.0, image_width * 0.24))

    x0 = max(0, int(np.floor(face_center_x - 0.85 * head_width_px)))
    x1 = min(image_width, int(np.ceil(face_center_x + 0.85 * head_width_px)))
    y0 = max(0, int(np.floor(face_top - 1.35 * head_width_px)))
    y1 = min(image_height, int(np.ceil(face_top + 0.30 * head_width_px)))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    gray = cv2.cvtColor(color_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(
        np.float32,
        copy=False,
    )
    roi_height, roi_width = gray.shape
    border_width = max(3, int(round(0.12 * roi_width)))
    border_height = max(3, int(round(0.12 * roi_height)))
    border = np.concatenate(
        (
            gray[:border_height].ravel(),
            gray[:, :border_width].ravel(),
            gray[:, -border_width:].ravel(),
        )
    )
    background_level = float(np.median(border))
    background_mad = float(np.median(np.abs(border - background_level)))
    best: tuple[float, int, int, int, int, int, float] | None = None

    thresholds = (
        background_level - max(20.0, 4.0 * background_mad),
        background_level - max(28.0, 5.0 * background_mad),
        background_level - max(35.0, 6.0 * background_mad),
        background_level - max(45.0, 7.0 * background_mad),
    )
    for threshold in thresholds:
        dark = (gray <= threshold).astype(np.uint8)
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            dark,
            connectivity=8,
        )
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            component_width = int(stats[label, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if (
                area < max(8, int(head_width_px * head_width_px * 0.015))
                or component_width < max(4, int(head_width_px * 0.12))
                or component_height < 2
            ):
                continue
            component = labels == label
            row_counts = component.sum(axis=1)
            supported_rows = np.flatnonzero(
                row_counts >= max(2, int(round(head_width_px * 0.04)))
            )
            if len(supported_rows) == 0:
                continue
            top = y0 + int(supported_rows[0])
            bottom = y0 + int(supported_rows[-1])
            center_x = x0 + float(np.median(np.where(component)[1]))
            extension = face_top - float(top)
            if (
                top <= 4
                or top > face_top - 0.15 * head_width_px
                or bottom < face_top - 0.95 * head_width_px
                or abs(center_x - face_center_x) > 0.65 * head_width_px
                or not 0.15 * head_width_px <= extension <= 1.30 * head_width_px
            ):
                continue
            component_values = gray[component]
            if len(component_values) == 0:
                continue
            contrast = background_level - float(np.median(component_values))
            if contrast < max(12.0, 1.5 * background_mad):
                continue
            score = (
                2.0 * float(np.log1p(area))
                + 2.0 * min(1.0, component_width / max(0.8 * head_width_px, 1.0))
                + min(1.0, component_height / max(0.8 * head_width_px, 1.0))
                - 2.0 * abs(extension - 0.75 * head_width_px) / max(head_width_px, 1.0)
                - 2.0 * abs(center_x - face_center_x) / max(head_width_px, 1.0)
            )
            candidate = (
                score,
                top,
                bottom,
                area,
                component_width,
                component_height,
                center_x,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        return None
    score, top, bottom, area, component_width, component_height, center_x = best
    return {
        "top_image_y": int(top),
        "bottom_image_y": int(bottom),
        "center_x": float(center_x),
        "head_width_pixels": float(head_width_px),
        "face_top_image_y": float(face_top),
        "extension_pixels": float(face_top - top),
        "component_area": int(area),
        "component_width_pixels": int(component_width),
        "component_height_pixels": int(component_height),
        "background_level": background_level,
        "background_mad": background_mad,
        "score": float(score),
        "roi_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "source": "rgb_head_silhouette",
    }


def _head_crown_candidate(
    depth_m: np.ndarray,
    keypoints: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    *,
    body_mask: np.ndarray,
    observed_top_image_y: int,
) -> dict[str, Any] | None:
    """Estimate a missing crown endpoint from face geometry and depth support.

    COCO has no crown keypoint.  The candidate is therefore conservative: it
    first searches for depth connected to the visible face, and only uses a
    small landmark-based projection when the depth image has a crown hole.
    """
    if keypoints.ndim != 2 or keypoints.shape[1] < 3:
        return None
    height, width = depth_m.shape
    facial_indices = tuple(index for index in range(min(5, len(keypoints))))
    facial = [
        keypoints[index]
        for index in facial_indices
        if float(keypoints[index, 2]) >= _KEYPOINT_SCORE
        and 0.0 <= float(keypoints[index, 0]) < width
        and 0.0 <= float(keypoints[index, 1]) < height
    ]
    if len(facial) < 2:
        return None

    face_xy = np.asarray([[float(point[0]), float(point[1])] for point in facial])
    face_top = float(np.min(face_xy[:, 1]))
    face_center_x = float(np.median(face_xy[:, 0]))
    ear_points = [
        keypoints[index]
        for index in (3, 4)
        if index < len(keypoints)
        and float(keypoints[index, 2]) >= _KEYPOINT_SCORE
        and 0.0 <= float(keypoints[index, 0]) < width
        and 0.0 <= float(keypoints[index, 1]) < height
    ]
    eye_points = [
        keypoints[index]
        for index in (1, 2)
        if index < len(keypoints)
        and float(keypoints[index, 2]) >= _KEYPOINT_SCORE
        and 0.0 <= float(keypoints[index, 0]) < width
        and 0.0 <= float(keypoints[index, 1]) < height
    ]
    if len(ear_points) == 2:
        head_width_px = float(np.linalg.norm(ear_points[0][:2] - ear_points[1][:2]))
    elif len(eye_points) == 2:
        head_width_px = float(np.linalg.norm(eye_points[0][:2] - eye_points[1][:2])) * 2.25
    else:
        head_width_px = float(np.ptp(face_xy[:, 0])) * 1.65
    head_width_px = float(np.clip(head_width_px, 18.0, width * 0.24))
    # COCO has no crown landmark. At this station the eye/ear line is roughly
    # 0.75 head-width below the top of the skull. The old shorter prior cut off
    # the hair by 5-8 cm. Keep this as a bounded anatomical prior, and only
    # replace it with depth when depth support actually reaches that height.
    crown_fraction = 0.45 if head_width_px <= 20.0 else 0.75
    crown_extension_px = float(
        np.clip(crown_fraction * head_width_px, 12.0, width * 0.10)
    )
    crown_guess_y = int(round(face_top - crown_extension_px))
    if head_width_px <= 20.0:
        # Tiny synthetic/test subjects do not have enough image support for a
        # reliable skull prior. Keep the candidate on the observed depth edge
        # instead of inventing an endpoint above the available body pixels.
        crown_guess_y = max(crown_guess_y, int(observed_top_image_y))
    if crown_guess_y <= 2 or crown_guess_y >= height - 2:
        return None

    facial_depth_points = [
        _point_from_keypoint(depth_m, point, intrinsics) for point in facial
    ]
    facial_depth_points = [point for point in facial_depth_points if point is not None]
    if not facial_depth_points:
        return None
    target_depth = float(np.median([point[2] for point in facial_depth_points]))
    tolerance = float(np.clip(max(0.05, 0.09 * target_depth), 0.05, 0.16))
    half_width = max(10.0, 0.62 * head_width_px)
    y0 = max(0, int(np.floor(crown_guess_y - 0.30 * head_width_px)))
    y1 = min(height, int(np.ceil(face_top + 0.82 * head_width_px)))
    x0 = max(0, int(np.floor(face_center_x - half_width)))
    x1 = min(width, int(np.ceil(face_center_x + half_width)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    crop = depth_m[y0:y1, x0:x1]
    valid = np.isfinite(crop) & (crop >= _MIN_DEPTH_M) & (crop <= _MAX_DEPTH_M)
    target_support = valid & (np.abs(crop - target_depth) <= tolerance)
    support = target_support.copy()
    if isinstance(body_mask, np.ndarray) and body_mask.shape == depth_m.shape:
        # The pose-supported mask may include background pixels around a
        # landmark. It is allowed to bridge a hole only after the depth itself
        # agrees with the face depth; otherwise a wall can become the crown.
        support |= (
            body_mask[y0:y1, x0:x1]
            & target_support
        )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        support.astype(np.uint8), connectivity=8
    )
    selected_labels: set[int] = set()
    anchor_count = 0
    for point in facial:
        px = int(round(float(point[0]))) - x0
        py = int(round(float(point[1]))) - y0
        if not (0 <= px < x1 - x0 and 0 <= py < y1 - y0):
            continue
        radius = max(2, int(round(head_width_px * 0.06)))
        local = labels[
            max(0, py - radius) : min(y1 - y0, py + radius + 1),
            max(0, px - radius) : min(x1 - x0, px + radius + 1),
        ]
        candidate_labels, candidate_counts = np.unique(
            local[local > 0], return_counts=True
        )
        if len(candidate_labels) == 0:
            continue
        label = int(candidate_labels[int(np.argmax(candidate_counts))])
        if int(stats[label, cv2.CC_STAT_AREA]) >= 3:
            selected_labels.add(label)
            anchor_count += 1
    component = np.isin(labels, list(selected_labels)) if selected_labels else np.zeros_like(labels, dtype=bool)
    row_counts = component.sum(axis=1)
    row_minimum = max(2, int(round(head_width_px * 0.035)))
    occupied_rows = np.flatnonzero(row_counts >= row_minimum)
    depth_supported = len(occupied_rows) > 0
    if depth_supported:
        top_row_local = int(occupied_rows[0])
        top_row = y0 + top_row_local
        top_band = max(2, int(round(head_width_px * 0.05)))
        projected = None
        # Use the first supported row whenever it has enough pixels. Taking
        # the median of the whole top band moves the endpoint downward and
        # creates a systematic short-height bias at the image scale used here.
        for band_rows in (1, min(top_band, 3)):
            yy, xx = np.where(
                component[
                    top_row_local : min(y1 - y0, top_row_local + band_rows),
                    :,
                ]
            )
            yy = yy + top_row_local
            xx = xx + x0
            projected = _robust_point_from_pixels(
                depth_m,
                yy + y0,
                xx,
                intrinsics,
                min_count=3,
            )
            if projected is not None:
                break
    else:
        top_row = crown_guess_y
        projected = None

    # A face-connected component that starts well below the anatomical crown
    # is forehead/face support, not evidence that the crown was measured. In
    # that case retain the expected crown row and project it using the robust
    # facial depth instead of reading a background pixel.
    depth_reaches_crown = bool(
        depth_supported
        and top_row <= int(round(crown_guess_y + 0.18 * head_width_px))
    )
    if depth_supported and not depth_reaches_crown:
        top_row = crown_guess_y
        projected = None

    projection_method = "depth_top_band" if projected is not None else None
    if projected is None and depth_reaches_crown:
        # If a selected-person depth component reaches the top, use that
        # component's upper edge. Falling back to crown_guess_y here can use a
        # valid but lower forehead pixel and systematically shorten stature.
        fallback_row = top_row
        projected = _robust_point_from_pixels(
            depth_m,
            np.asarray([fallback_row]),
            np.asarray([int(round(face_center_x))]),
            intrinsics,
            min_count=1,
        )
        if projected is not None:
            projection_method = (
                "depth_top_edge_single_pixel"
                if depth_reaches_crown
                else "depth_crown_prior_single_pixel"
            )
    if projected is None:
        fx, fy, cx, cy = intrinsics
        projected = (
            np.asarray(
                [
                    (face_center_x - cx) * target_depth / fx,
                    (float(top_row) - cy) * target_depth / fy,
                    target_depth,
                ],
                dtype=np.float32,
            ),
            tolerance,
            0,
        )
        projection_method = (
            "metric_crown_projection"
            if not depth_reaches_crown
            else "metric_crown_projection_depth_missing"
        )

    point, depth_iqr, point_count = projected
    gap_px = int(observed_top_image_y - top_row)
    depth_quality = float(np.exp(-depth_iqr / 0.05))
    anchor_quality = float(anchor_count / max(len(facial), 1))
    support_quality = 1.0 if depth_supported else 0.35
    confidence = float(
        np.clip(
            0.22 + 0.25 * depth_quality + 0.28 * anchor_quality + 0.20 * support_quality,
            0.0,
            0.90,
        )
    )
    return {
        "point_camera_m": point,
        "depth_iqr_m": float(depth_iqr),
        "point_count": int(point_count),
        "top_image_y": int(top_row),
        "crown_guess_image_y": int(crown_guess_y),
        "observed_top_image_y": int(observed_top_image_y),
        "extension_pixels": int(gap_px),
        "head_width_pixels": float(head_width_px),
        "face_keypoint_count": int(len(facial)),
        "depth_anchor_count": int(anchor_count),
        "depth_supported": bool(depth_supported),
        "depth_reaches_crown": depth_reaches_crown,
        "projection_method": projection_method,
        "confidence": confidence,
        "source": "depth_connected_head_crown"
        if depth_supported
        else "landmark_head_crown_projection",
    }


def _mean_points(points: list[np.ndarray | None]) -> np.ndarray | None:
    valid = [point for point in points if point is not None]
    if not valid:
        return None
    return np.mean(np.stack(valid, axis=0), axis=0)


def _pose_anchored_component_labels(
    labels: np.ndarray,
    stats: np.ndarray,
    keypoints: np.ndarray,
    *,
    x_offset: int,
    y_offset: int,
) -> tuple[set[int], int]:
    """Recover depth components split by holes using the selected person's pose."""
    crop_height, crop_width = labels.shape
    radius = int(np.clip(round(min(crop_height, crop_width) * 0.055), 10, 28))
    selected: set[int] = set()
    anchored_keypoint_count = 0

    for keypoint in keypoints:
        if len(keypoint) < 3 or float(keypoint[2]) < _KEYPOINT_SCORE:
            continue
        x = int(round(float(keypoint[0]) - x_offset))
        y = int(round(float(keypoint[1]) - y_offset))
        if not (0 <= x < crop_width and 0 <= y < crop_height):
            continue
        x0, x1 = max(0, x - radius), min(crop_width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(crop_height, y + radius + 1)
        patch = labels[y0:y1, x0:x1]
        candidate_labels, counts = np.unique(patch[patch > 0], return_counts=True)
        if len(candidate_labels) == 0:
            continue
        valid_candidates = [
            (int(count), int(label))
            for label, count in zip(candidate_labels, counts)
            if int(stats[int(label), cv2.CC_STAT_AREA]) >= 30
        ]
        if not valid_candidates:
            continue
        # The component occupying most of the local pose neighborhood is the
        # one supported by this selected-person landmark.
        _count, label = max(valid_candidates)
        selected.add(label)
        anchored_keypoint_count += 1
    return selected, anchored_keypoint_count


def _pose_depth_supported_foreground(
    crop: np.ndarray,
    valid: np.ndarray,
    keypoints: np.ndarray,
    *,
    x_offset: int,
    y_offset: int,
) -> tuple[np.ndarray, int]:
    """Add only local depth support around the selected person's landmarks."""
    crop_height, crop_width = crop.shape
    radius_x = int(np.clip(round(crop_width * 0.20), 14, 58))
    radius_y = int(np.clip(round(crop_height * 0.14), 18, 72))
    foreground = np.zeros_like(valid, dtype=bool)
    supported_keypoint_count = 0

    for keypoint in keypoints:
        if len(keypoint) < 3 or float(keypoint[2]) < _KEYPOINT_SCORE:
            continue
        x = int(round(float(keypoint[0]) - x_offset))
        y = int(round(float(keypoint[1]) - y_offset))
        if not (0 <= x < crop_width and 0 <= y < crop_height):
            continue
        x0, x1 = max(0, x - radius_x), min(crop_width, x + radius_x + 1)
        y0, y1 = max(0, y - radius_y), min(crop_height, y + radius_y + 1)
        patch = crop[y0:y1, x0:x1]
        patch_valid = valid[y0:y1, x0:x1]
        if int(patch_valid.sum()) < 6:
            continue
        median_depth = float(np.median(patch[patch_valid]))
        tolerance = float(np.clip(max(0.045, 0.04 * median_depth), 0.045, 0.14))
        support = patch_valid & (np.abs(patch - median_depth) <= tolerance)
        if int(support.sum()) < 6:
            continue
        foreground[y0:y1, x0:x1] |= support
        supported_keypoint_count += 1
    return foreground, supported_keypoint_count


def _camera_points_from_pixels(
    depth_m: np.ndarray,
    image_y: np.ndarray,
    image_x: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> np.ndarray:
    """Back-project valid depth pixels to Gemini camera coordinates."""
    z = depth_m[image_y, image_x].astype(np.float32, copy=False)
    fx, fy, cx, cy = intrinsics
    return np.column_stack(
        (
            (image_x.astype(np.float32) - float(cx)) * z / float(fx),
            (image_y.astype(np.float32) - float(cy)) * z / float(fy),
            z,
        )
    )


def _unit_vector(values: np.ndarray) -> np.ndarray | None:
    magnitude = float(np.sqrt(float(np.sum(values.astype(np.float64) ** 2))))
    if not np.isfinite(magnitude) or magnitude <= 1e-6:
        return None
    return values.astype(np.float32, copy=False) / magnitude


def _plane_from_triplet(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Create a plane from three points without loading a linear-algebra runtime."""
    if points.ndim != 2 or points.shape != (3, 3) or not np.isfinite(points).all():
        return None
    normal = _unit_vector(np.cross(points[1] - points[0], points[2] - points[0]))
    if normal is None:
        return None
    return normal, float(np.dot(normal, points[0]))


def _solve_3x3(matrix: list[list[float]], values: list[float]) -> np.ndarray | None:
    """Small Gaussian-elimination solver used for the floor-plane refinement."""
    augmented = [row[:] + [value] for row, value in zip(matrix, values)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-10:
            return None
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for item in range(column, 4):
            augmented[column][item] /= divisor
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            for item in range(column, 4):
                augmented[row][item] -= factor * augmented[column][item]
    solution = np.asarray([augmented[row][3] for row in range(3)], dtype=np.float32)
    return solution if np.isfinite(solution).all() else None


def _refine_floor_plane(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Fit y = ax + bz + c, appropriate after the floor-like normal gate."""
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 3:
        return None
    x = points[:, 0].astype(np.float64, copy=False)
    y = points[:, 1].astype(np.float64, copy=False)
    z = points[:, 2].astype(np.float64, copy=False)
    count = float(len(points))
    coefficients = _solve_3x3(
        [
            [float(np.sum(x * x)), float(np.sum(x * z)), float(np.sum(x))],
            [float(np.sum(x * z)), float(np.sum(z * z)), float(np.sum(z))],
            [float(np.sum(x)), float(np.sum(z)), count],
        ],
        [float(np.sum(x * y)), float(np.sum(z * y)), float(np.sum(y))],
    )
    if coefficients is None:
        return None
    slope_x, slope_z, intercept = (float(value) for value in coefficients)
    normal = _unit_vector(np.asarray([-slope_x, 1.0, -slope_z], dtype=np.float32))
    if normal is None:
        return None
    return normal, float(intercept / max(float(np.sqrt(1.0 + slope_x**2 + slope_z**2)), 1e-6))


def _refine_support_plane(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Refine an arbitrary support plane with a small PCA fit."""
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 3:
        return None
    if not np.isfinite(points).all():
        return None
    center = np.mean(points.astype(np.float64, copy=False), axis=0)
    centered = points.astype(np.float64, copy=False) - center
    try:
        _u, _singular, vectors = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = _unit_vector(vectors[-1].astype(np.float32, copy=False))
    if normal is None:
        return None
    return normal, float(np.dot(normal, center.astype(np.float32, copy=False)))


def _estimate_support_plane(
    depth_m: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    bbox: tuple[float, float, float, float],
    body_points: list[np.ndarray],
) -> dict[str, Any]:
    """Estimate a broad bed/support plane around a lying person.

    Unlike the standing floor estimator, this routine does not assume that the
    plane normal is aligned with camera y. The person box is excluded and a
    surrounding ring is used so that the bed can be found even when the
    patient occupies most of the image.
    """
    image_height, image_width = depth_m.shape
    valid = np.isfinite(depth_m) & (depth_m >= _MIN_DEPTH_M) & (depth_m <= _MAX_DEPTH_M)
    x0f, y0f, x1f, y1f = bbox
    box_width = max(x1f - x0f, 20.0)
    box_height = max(y1f - y0f, 20.0)
    ring_x0 = max(0, int(np.floor(x0f - 0.42 * box_width)))
    ring_y0 = max(0, int(np.floor(y0f - 0.42 * box_height)))
    ring_x1 = min(image_width, int(np.ceil(x1f + 0.42 * box_width)))
    ring_y1 = min(image_height, int(np.ceil(y1f + 0.42 * box_height)))
    region = np.zeros_like(valid, dtype=bool)
    region[ring_y0:ring_y1, ring_x0:ring_x1] = True
    blocked = np.zeros_like(valid, dtype=bool)
    block_x0 = max(0, int(np.floor(x0f - 0.08 * box_width)))
    block_y0 = max(0, int(np.floor(y0f - 0.08 * box_height)))
    block_x1 = min(image_width, int(np.ceil(x1f + 0.08 * box_width)))
    block_y1 = min(image_height, int(np.ceil(y1f + 0.08 * box_height)))
    blocked[block_y0:block_y1, block_x0:block_x1] = True
    candidates = valid & region & ~blocked
    flat = np.flatnonzero(candidates)
    if len(flat) < 220:
        return {
            "status": "insufficient_support_plane_pixels",
            "candidate_pixel_count": int(len(flat)),
            "fit_method": "depth_ransac_arbitrary_plane",
        }

    sample_count = min(int(len(flat)), 3000)
    sample_indices = flat[
        np.linspace(0, len(flat) - 1, sample_count, dtype=np.int64)
    ]
    image_y, image_x = np.unravel_index(sample_indices, depth_m.shape)
    points = _camera_points_from_pixels(depth_m, image_y, image_x, intrinsics)
    body_array = np.asarray(body_points, dtype=np.float32)
    if body_array.ndim != 2 or body_array.shape[1:] != (3,) or len(body_array) < 2:
        return {
            "status": "insufficient_body_anchors_for_support_plane",
            "candidate_pixel_count": int(len(flat)),
            "fit_method": "depth_ransac_arbitrary_plane",
        }

    depth_median = float(np.median(points[:, 2]))
    distance_threshold_m = float(np.clip(0.010 + 0.004 * depth_median, 0.014, 0.035))
    minimum_inliers = max(100, int(sample_count * 0.05))
    rng = np.random.default_rng(20260902)
    iterations = min(220, max(100, sample_count // 20))
    best: dict[str, Any] | None = None
    for _iteration in range(iterations):
        sample = points[rng.choice(sample_count, size=3, replace=False)]
        provisional = _plane_from_triplet(sample)
        if provisional is None:
            continue
        normal, plane_offset = provisional
        residuals = np.abs(_plane_signed_distances(points, normal, plane_offset))
        inliers = residuals <= distance_threshold_m
        if int(inliers.sum()) < minimum_inliers:
            continue
        refined = _refine_support_plane(points[inliers])
        if refined is None:
            continue
        normal, plane_offset = refined
        residuals = np.abs(_plane_signed_distances(points, normal, plane_offset))
        inliers = residuals <= distance_threshold_m
        inlier_count = int(inliers.sum())
        if inlier_count < minimum_inliers:
            continue
        xs = image_x[inliers]
        ys = image_y[inliers]
        span_x = int(np.ptp(xs))
        span_y = int(np.ptp(ys))
        if (
            span_x < max(40, int(round(image_width * 0.10)))
            or span_y < max(30, int(round(image_height * 0.08)))
        ):
            continue
        anchor_distances = np.abs(_plane_signed_distances(body_array, normal, plane_offset))
        median_anchor_gap = float(np.median(anchor_distances))
        if not np.isfinite(median_anchor_gap) or median_anchor_gap > 0.45:
            continue
        inlier_fraction = float(inlier_count / max(sample_count, 1))
        support_score = min(1.0, inlier_fraction / 0.35)
        spread_score = min(
            1.0,
            0.5 * span_x / max(float(image_width) * 0.55, 1.0)
            + 0.5 * span_y / max(float(image_height) * 0.45, 1.0),
        )
        anchor_score = float(np.exp(-median_anchor_gap / 0.16))
        score = 0.60 * support_score + 0.22 * spread_score + 0.18 * anchor_score
        candidate = {
            "status": "ok",
            "score": float(score),
            "normal_camera": normal.tolist(),
            "plane_offset_m": float(plane_offset),
            "candidate_pixel_count": int(len(flat)),
            "sample_count": int(sample_count),
            "ransac_iterations": int(iterations),
            "plane_distance_threshold_m": float(distance_threshold_m),
            "inlier_count": int(inlier_count),
            "inlier_fraction": inlier_fraction,
            "residual_iqr_m": float(
                np.diff(np.percentile(residuals[inliers], [25.0, 75.0]))[0]
            ),
            "image_span_px": [span_x, span_y],
            "median_body_to_plane_m": median_anchor_gap,
            "max_body_to_plane_m": float(np.max(anchor_distances)),
            "fit_method": "depth_ransac_arbitrary_plane",
            "source": "depth_support_plane_ring",
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    if best is None:
        return {
            "status": "support_plane_not_confirmed",
            "candidate_pixel_count": int(len(flat)),
            "sample_count": int(sample_count),
            "ransac_iterations": int(iterations),
            "fit_method": "depth_ransac_arbitrary_plane",
        }
    return best


def _plane_signed_distances(
    points: np.ndarray,
    normal: np.ndarray,
    plane_offset: float,
) -> np.ndarray:
    """Evaluate a plane without a large BLAS matrix multiply in the UI runtime."""
    return (
        points[:, 0] * float(normal[0])
        + points[:, 1] * float(normal[1])
        + points[:, 2] * float(normal[2])
        - float(plane_offset)
    )


def _foot_sole_candidate(
    depth_m: np.ndarray,
    keypoint: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    *,
    reference_depth_m: float | None,
    body_mask: np.ndarray | None,
    person_height_px: float,
    person_width_px: float,
) -> dict[str, Any] | None:
    """Find a foot bottom connected to one ankle, not a nearby floor pixel."""
    image_height, image_width = depth_m.shape
    if len(keypoint) < 3 or float(keypoint[2]) < _KEYPOINT_SCORE:
        return None
    ankle_x = float(keypoint[0])
    ankle_y = float(keypoint[1])
    if not (0.0 <= ankle_x < image_width and 0.0 <= ankle_y < image_height):
        return None
    ankle_point = _point_from_keypoint(depth_m, keypoint, intrinsics)
    if ankle_point is None:
        return None
    ankle_depth = float(ankle_point[2])
    if reference_depth_m is not None:
        maximum_gap = max(_FOOT_DEPTH_TOLERANCE_M, 0.24 * float(reference_depth_m))
        if abs(ankle_depth - float(reference_depth_m)) > maximum_gap:
            return {
                "rejected": True,
                "reason": "ankle_depth_inconsistent_with_selected_person",
                "ankle_depth_m": ankle_depth,
                "reference_depth_m": float(reference_depth_m),
                "depth_gap_m": abs(ankle_depth - float(reference_depth_m)),
            }

    height_span = max(float(person_height_px), 160.0)
    width_span = max(float(person_width_px), 80.0)
    radius_x = int(np.clip(round(0.075 * width_span), 8, 42))
    y_above = int(np.clip(round(0.045 * height_span), 8, 34))
    # The lower part of the search window must cover a shoe, not the floor
    # behind it.  At this station the ankle-to-sole image span is normally
    # below 10% of the visible person height; a larger window lets the
    # downward-looking camera connect the shoe to a broad floor patch.
    y_below = int(np.clip(round(0.115 * height_span), 20, 82))
    x0 = max(0, int(round(ankle_x)) - radius_x)
    x1 = min(image_width, int(round(ankle_x)) + radius_x + 1)
    y0 = max(0, int(round(ankle_y)) - y_above)
    y1 = min(image_height, int(round(ankle_y)) + y_below + 1)
    if x1 - x0 < 5 or y1 - y0 < 8:
        return None

    crop = depth_m[y0:y1, x0:x1]
    valid = np.isfinite(crop) & (crop >= _MIN_DEPTH_M) & (crop <= _MAX_DEPTH_M)
    tolerance = float(np.clip(max(0.09, 0.16 * ankle_depth), 0.09, 0.30))
    support = valid & (np.abs(crop - ankle_depth) <= tolerance)
    # Do not OR in an unrestricted body mask here.  A mask built from a broad
    # depth band can include the floor below a shoe; the ankle-anchored depth
    # component and the bounded image window are the safer foot evidence.
    if int(support.sum()) < 5:
        return {
            "rejected": True,
            "reason": "no_connected_foot_depth",
            "ankle_depth_m": ankle_depth,
            "search_pixels": int(valid.sum()),
        }

    connected = cv2.morphologyEx(
        support.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5)),
        iterations=1,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
    anchor_x = int(round(ankle_x)) - x0
    anchor_y = int(round(ankle_y)) - y0
    selected_label = 0
    if 0 <= anchor_x < x1 - x0 and 0 <= anchor_y < y1 - y0:
        local_labels = labels[
            max(0, anchor_y - 4) : min(y1 - y0, anchor_y + 5),
            max(0, anchor_x - 4) : min(x1 - x0, anchor_x + 5),
        ]
        values, counts = np.unique(local_labels[local_labels > 0], return_counts=True)
        if len(values):
            selected_label = int(values[int(np.argmax(counts))])
    if selected_label == 0:
        viable = [
            label
            for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= 5
        ]
        if viable:
            selected_label = max(
                viable,
                key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
            )
    if selected_label == 0:
        return {
            "rejected": True,
            "reason": "foot_component_not_found",
            "ankle_depth_m": ankle_depth,
        }

    component = labels == selected_label
    row_counts = component.sum(axis=1)
    occupied_rows = np.flatnonzero(row_counts >= max(2, radius_x // 4))
    if len(occupied_rows) == 0:
        return {
            "rejected": True,
            "reason": "foot_component_has_no_bottom_edge",
            "ankle_depth_m": ankle_depth,
        }
    sole_row_local = int(occupied_rows[-1])
    band = max(2, int(round(0.025 * height_span)))
    selection = component[max(0, sole_row_local - band + 1) : sole_row_local + 1]
    image_y, image_x = np.where(selection)
    image_y = image_y + max(0, sole_row_local - band + 1) + y0
    image_x = image_x + x0
    projected = _robust_point_from_pixels(
        depth_m,
        image_y,
        image_x,
        intrinsics,
        min_count=3,
    )
    if projected is None:
        sole_point = ankle_point
        sole_iqr = tolerance
        point_count = 0
        used_ankle_as_sole = True
    else:
        sole_point, sole_iqr, point_count = projected
        used_ankle_as_sole = False
    return {
        "rejected": False,
        "ankle_image_xy": [ankle_x, ankle_y],
        "ankle_camera_point_m": ankle_point.tolist(),
        "sole_image_y": int(y0 + sole_row_local),
        "sole_camera_point_m": sole_point.tolist(),
        "sole_depth_iqr_m": float(sole_iqr),
        "sole_point_count": int(point_count),
        "ankle_depth_m": ankle_depth,
        "depth_tolerance_m": tolerance,
        "used_ankle_as_sole": bool(used_ankle_as_sole),
        "support_pixel_count": int(component.sum()),
    }


def _confirmed_feet(
    depth_m: np.ndarray,
    keypoints: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    *,
    bottom_image_y: int,
    edge_margin: int,
    reference_depth_m: float | None = None,
    body_mask: np.ndarray | None = None,
    person_height_px: float | None = None,
    person_width_px: float | None = None,
) -> dict[str, Any]:
    """Confirm in-frame soles and reject ankle pixels sampled from background."""
    image_height, image_width = depth_m.shape
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    height_span = float(person_height_px or max(bottom_image_y, 160))
    width_span = float(person_width_px or max(image_width * 0.20, 80))
    for index in _ANKLE_INDICES:
        if index >= len(keypoints):
            continue
        keypoint = keypoints[index]
        candidate = _foot_sole_candidate(
            depth_m,
            keypoint,
            intrinsics,
            reference_depth_m=reference_depth_m,
            body_mask=body_mask,
            person_height_px=height_span,
            person_width_px=width_span,
        )
        if candidate is None:
            continue
        candidate["keypoint_index"] = int(index)
        if candidate.get("rejected"):
            rejected.append(candidate)
        else:
            candidates.append(candidate)
    if not candidates:
        return {
            "confirmed": False,
            "reason": (
                "ankle_depth_inconsistent_with_selected_person"
                if any(item.get("reason") == "ankle_depth_inconsistent_with_selected_person" for item in rejected)
                else "no_visible_sole_with_valid_depth"
            ),
            "visible_ankle_count": 0,
            "rejected_ankles": rejected,
        }
    lowest_sole_y = max(int(item["sole_image_y"]) for item in candidates)
    if lowest_sole_y >= image_height - int(edge_margin):
        return {
            "confirmed": False,
            "reason": "sole_touches_image_edge",
            "visible_ankle_count": len(candidates),
            "lowest_sole_image_y": lowest_sole_y,
            "frame_height": int(image_height),
            "rejected_ankles": rejected,
        }
    return {
        "confirmed": True,
        "visible_ankle_count": len(candidates),
        "lowest_sole_image_y": lowest_sole_y,
        "body_bottom_image_y": int(bottom_image_y),
        "ankles": candidates,
        "rejected_ankles": rejected,
        "sole_edge_margin_px": int(image_height - lowest_sole_y),
    }


def _estimate_floor_plane(
    depth_m: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    body_mask: np.ndarray,
    *,
    target_depth_m: float | None,
    foot_points: list[np.ndarray],
    sole_count: int = 1,
) -> dict[str, Any]:
    """Find a likely floor plane from RGB-D geometry, without raw IMU axes.

    Gemini's IMU and depth-camera coordinate frames are not assumed to be the
    same.  Candidate planes therefore come only from lower-image depth points
    outside the selected person.  A plane must also be broad, floor-like in
    camera coordinates, and pass close to both the observed sole and ankle.
    """
    image_height, image_width = depth_m.shape
    valid = np.isfinite(depth_m) & (depth_m >= _MIN_DEPTH_M) & (depth_m <= _MAX_DEPTH_M)
    dilation = max(7, int(round(min(image_height, image_width) * 0.018)) | 1)
    blocked = cv2.dilate(
        body_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation)),
        iterations=1,
    ).astype(bool)
    candidates = valid & ~blocked
    # The lower half is where a standing measurement can observe the floor.
    candidates[: int(round(image_height * 0.42)), :] = False
    if target_depth_m is not None and np.isfinite(target_depth_m):
        # Do not assume that the floor has exactly the torso depth: a downward
        # looking camera can see nearer floor pixels near the image edge.
        lower = max(_MIN_DEPTH_M, float(target_depth_m) - 2.0)
        upper = min(_MAX_DEPTH_M, float(target_depth_m) + 5.0)
        candidates &= (depth_m >= lower) & (depth_m <= upper)

    flat = np.flatnonzero(candidates)
    if len(flat) < 250:
        return {
            "status": "insufficient_floor_pixels",
            "candidate_pixel_count": int(len(flat)),
        }
    # Bound per-frame work. The fixed seed makes an identical RGB-D frame
    # produce identical diagnostics, which matters for capture audit trails.
    sample_count = min(int(len(flat)), 6000)
    sample_indices = flat[np.linspace(0, len(flat) - 1, sample_count, dtype=np.int64)]
    image_y, image_x = np.unravel_index(sample_indices, depth_m.shape)
    points = _camera_points_from_pixels(depth_m, image_y, image_x, intrinsics)
    foot_array = np.asarray(foot_points, dtype=np.float32)
    if foot_array.ndim != 2 or foot_array.shape[0] < 2 or foot_array.shape[1] != 3:
        return {"status": "feet_not_available_for_floor_fit"}

    depth_median = float(np.median(points[:, 2]))
    distance_threshold_m = float(np.clip(0.014 + 0.006 * depth_median, 0.020, 0.045))
    minimum_inliers = max(120, int(sample_count * 0.035))
    # Most valid pixels belong to the person or distant background.  A loose
    # lower-image sample still admits furniture and wall-adjacent planes, which
    # can win when the true floor occupies a small fraction of the frame. Bias
    # hypotheses toward the bottom band where the feet meet the floor; every
    # hypothesis is still scored on all points so this does not hard-code a
    # station-specific floor location.
    floor_sample_mask = image_y >= int(round(image_height * 0.78))
    floor_sample_indices = np.flatnonzero(floor_sample_mask)
    if len(floor_sample_indices) < 30:
        floor_sample_indices = np.arange(sample_count, dtype=np.int64)
    floor_sample_count = int(len(floor_sample_indices))
    iterations = min(220, max(120, int(floor_sample_count // 30)))
    rng = np.random.default_rng(20260824)
    best: dict[str, Any] | None = None

    for _iteration in range(iterations):
        sample = points[
            floor_sample_indices[rng.choice(floor_sample_count, size=3, replace=False)]
        ]
        provisional = _plane_from_triplet(sample)
        if provisional is None:
            continue
        normal, plane_offset = provisional
        residuals = np.abs(_plane_signed_distances(points, normal, plane_offset))
        inliers = residuals <= distance_threshold_m
        if int(inliers.sum()) < minimum_inliers:
            continue
        refined = _refine_floor_plane(points[inliers])
        if refined is None:
            continue
        normal, plane_offset = refined
        residuals = np.abs(_plane_signed_distances(points, normal, plane_offset))
        inliers = residuals <= distance_threshold_m
        inlier_count = int(inliers.sum())
        if inlier_count < minimum_inliers:
            continue

        xs = image_x[inliers]
        ys = image_y[inliers]
        span_x = int(np.ptp(xs))
        span_y = int(np.ptp(ys))
        x_bins = np.unique(np.minimum((xs * 4) // max(image_width, 1), 3))
        if (
            span_x < max(48, int(round(image_width * 0.15)))
            or span_y < max(20, int(round(image_height * 0.07)))
            or len(x_bins) < 2
        ):
            continue

        # With the standard aligned-depth convention, a floor's normal has a
        # meaningful image-y component. This rejects a background wall whose
        # constant depth can otherwise dominate RANSAC support.
        normal_y_component = abs(float(normal[1]))
        horizontal_normal_component = float(
            np.sqrt(float(normal[0]) ** 2 + float(normal[2]) ** 2)
        )
        # A floor viewed by this fixed RGB-D camera has a dominant camera-y
        # normal.  Walls, table tops, and chair surfaces can also pass a loose
        # ``normal_y`` threshold, so reject planes with a large horizontal
        # normal before scoring them against the feet.
        if normal_y_component < 0.95 or horizontal_normal_component > 0.32:
            continue

        foot_distances = np.abs(
            _plane_signed_distances(foot_array, normal, plane_offset)
        )
        sole_count = int(np.clip(sole_count, 1, len(foot_distances) - 1))
        sole_distances = foot_distances[:sole_count]
        ankle_distances = foot_distances[sole_count:]
        sole_gap_m = float(np.median(sole_distances))
        nearest_sole_gap_m = float(np.min(sole_distances))
        nearest_ankle_gap_m = float(np.min(ankle_distances))
        if sole_gap_m > 0.18 or nearest_ankle_gap_m > 0.28:
            continue

        inlier_fraction = float(inlier_count / max(sample_count, 1))
        support_score = min(1.0, inlier_fraction / 0.35)
        proximity_score = float(
            0.58 * np.exp(-sole_gap_m / 0.07)
            + 0.42 * np.exp(-nearest_ankle_gap_m / 0.13)
        )
        geometry_score = min(1.0, normal_y_component / 0.65)
        score = 0.56 * support_score + 0.31 * proximity_score + 0.13 * geometry_score
        candidate = {
            "score": float(score),
            "normal_camera": normal.tolist(),
            "plane_offset_m": float(plane_offset),
            "candidate_pixel_count": int(len(flat)),
            "sample_count": int(sample_count),
            "floor_sample_count": floor_sample_count,
            "ransac_iterations": int(iterations),
            "plane_distance_threshold_m": distance_threshold_m,
            "inlier_count": inlier_count,
            "inlier_fraction": inlier_fraction,
            "residual_iqr_m": float(np.diff(np.percentile(residuals[inliers], [25.0, 75.0]))[0]),
            "image_span_px": [span_x, span_y],
            "horizontal_bin_count": int(len(x_bins)),
            "normal_y_component": normal_y_component,
            "horizontal_normal_component": horizontal_normal_component,
            "sole_to_plane_m": sole_gap_m,
            "nearest_sole_to_plane_m": nearest_sole_gap_m,
            "nearest_ankle_to_plane_m": nearest_ankle_gap_m,
            "fit_method": "depth_ransac_no_imu",
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best is None:
        return {
            "status": "floor_plane_not_confirmed",
            "candidate_pixel_count": int(len(flat)),
            "sample_count": int(sample_count),
            "ransac_iterations": int(iterations),
            "fit_method": "depth_ransac_no_imu",
        }
    best["status"] = "ok"
    return best


def _expanded_measurement_bbox(
    result: dict[str, Any],
    image_shape: tuple[int, int],
    keypoints: np.ndarray,
) -> tuple[tuple[int, int, int, int] | None, dict[str, Any]]:
    """Expand a detector box enough to search for feet and the crown."""
    bbox = _bbox_from_result(result)
    if bbox is None:
        return None, {"status": "no_selected_person"}
    image_height, image_width = image_shape
    x0f, y0f, x1f, y1f = bbox
    bbox_width = max(x1f - x0f, 30.0)
    bbox_height = max(y1f - y0f, 80.0)
    valid = _in_frame_keypoints(keypoints, image_shape, minimum_score=0.20)
    pose_x = [float(point[0]) for _index, point in valid]
    pose_y = [float(point[1]) for _index, point in valid]
    ankle_y = [
        float(keypoints[index, 1])
        for index in _ANKLE_INDICES
        if index < len(keypoints)
        and float(keypoints[index, 2]) >= 0.20
        and 0.0 <= float(keypoints[index, 1]) < image_height
    ]
    horizontal_pad = max(10.0, 0.06 * bbox_width)
    top_pad = max(6.0, 0.035 * bbox_height)
    bottom_pad = max(12.0, 0.14 * bbox_height)
    x0 = max(0, int(np.floor(min([x0f - horizontal_pad, *[x - horizontal_pad for x in pose_x]]))))
    x1 = min(image_width, int(np.ceil(max([x1f + horizontal_pad, *[x + horizontal_pad for x in pose_x]]))))
    y0 = max(0, int(np.floor(min([y0f - top_pad, *[y - top_pad for y in pose_y]]))))
    lower_candidates = [y1f + bottom_pad]
    lower_candidates.extend(y + bottom_pad for y in ankle_y)
    lower_candidates.extend(y + 0.04 * bbox_height for y in pose_y)
    y1 = min(image_height, int(np.ceil(max(lower_candidates))))
    if x1 <= x0 or y1 <= y0:
        return None, {"status": "invalid_expanded_person_box"}
    return (x0, y0, x1, y1), {
        "status": "ok",
        "detector_bbox_xyxy": [x0f, y0f, x1f, y1f],
        "expanded_bbox_xyxy": [x0, y0, x1, y1],
        "detector_touched_bottom_edge": bool(y1f >= image_height - 1),
        "in_frame_pose_keypoint_count": int(len(valid)),
        "in_frame_pose_y_range": [min(pose_y), max(pose_y)] if pose_y else None,
        "in_frame_ankle_y": ankle_y,
    }


def _skeleton_height_candidate(
    depth_m: np.ndarray,
    keypoints: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    *,
    reference_depth_m: float | None,
    crown_point: np.ndarray | None,
    feet: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an independent metric candidate from robust 3D pose segments."""
    points: list[np.ndarray | None] = []
    rejected_depth_indices: list[int] = []
    for index, keypoint in enumerate(keypoints):
        point = _point_from_keypoint(depth_m, keypoint, intrinsics)
        if point is not None and reference_depth_m is not None:
            gap = abs(float(point[2]) - float(reference_depth_m))
            if gap > max(0.42, 0.30 * float(reference_depth_m)):
                rejected_depth_indices.append(index)
                point = None
        points.append(point)

    face = _mean_points(points[index] for index in (0, 1, 2, 3, 4))
    shoulders = _mean_points(points[index] for index in (5, 6))
    hips = _mean_points(points[index] for index in (11, 12))
    if face is None or shoulders is None or hips is None:
        return {
            "status": "insufficient_pose_keypoints",
            "rejected_depth_keypoint_indices": rejected_depth_indices,
        }

    leg_chains: list[float] = []
    leg_indices: list[int] = []
    for hip_index, knee_index, ankle_index in ((11, 13, 15), (12, 14, 16)):
        hip, knee, ankle = points[hip_index], points[knee_index], points[ankle_index]
        if hip is None or knee is None or ankle is None:
            continue
        leg_chains.append(
            float(np.linalg.norm(hip - knee) + np.linalg.norm(knee - ankle))
        )
        leg_indices.append(ankle_index)
    if not leg_chains:
        return {
            "status": "insufficient_leg_chain",
            "rejected_depth_keypoint_indices": rejected_depth_indices,
        }

    head_to_shoulders = float(np.linalg.norm(face - shoulders))
    torso = float(np.linalg.norm(shoulders - hips))
    if head_to_shoulders < 0.05 or torso < 0.10:
        return {
            "status": "implausible_pose_segments",
            "rejected_depth_keypoint_indices": rejected_depth_indices,
        }
    crown_face = (
        float(np.linalg.norm(crown_point - face))
        if crown_point is not None
        else float(np.clip(0.55 * head_to_shoulders, 0.08, 0.17))
    )
    sole_extensions: list[float] = []
    ankle_candidates = feet.get("ankles") if isinstance(feet, Mapping) else None
    if isinstance(ankle_candidates, list):
        for item in ankle_candidates:
            if not isinstance(item, Mapping):
                continue
            try:
                ankle = np.asarray(item["ankle_camera_point_m"], dtype=np.float32)
                sole = np.asarray(item["sole_camera_point_m"], dtype=np.float32)
            except (KeyError, TypeError, ValueError):
                continue
            if ankle.shape == (3,) and sole.shape == (3,) and np.isfinite(ankle).all() and np.isfinite(sole).all():
                extension = float(np.linalg.norm(sole - ankle))
                if extension <= 0.18:
                    sole_extensions.append(extension)
    foot_extension = float(np.median(sole_extensions)) if sole_extensions else 0.035
    height_m = (
        crown_face
        + head_to_shoulders
        + torso
        + float(np.mean(leg_chains))
        + foot_extension
    )
    if not _HEIGHT_MIN_M <= height_m <= _HEIGHT_MAX_M:
        return {
            "status": "height_outside_expected_range",
            "raw_height_cm": height_m * 100.0,
            "rejected_depth_keypoint_indices": rejected_depth_indices,
        }
    symmetry_quality = 1.0
    if len(leg_chains) == 2:
        symmetry_quality = float(
            np.exp(-abs(leg_chains[0] - leg_chains[1]) / max(0.04, float(np.mean(leg_chains))))
        )
    valid_point_count = sum(point is not None for point in points)
    confidence = float(
        np.clip(
            0.18
            + 0.035 * min(valid_point_count, 14)
            + 0.10 * float(len(leg_chains) == 2)
            + 0.10 * symmetry_quality
            + 0.06 * float(crown_point is not None),
            0.0,
            0.78,
        )
    )
    uncertainty_cm = float(
        np.clip(
            8.0
            + (2 - len(leg_chains)) * 4.0
            + (1.0 - symmetry_quality) * 8.0
            + max(0, 8 - valid_point_count) * 1.2,
            7.0,
            24.0,
        )
    )
    return {
        "status": "ok",
        "height_cm": height_m * 100.0,
        "height_lower_cm": height_m * 100.0 - uncertainty_cm,
        "height_upper_cm": height_m * 100.0 + uncertainty_cm,
        "confidence": confidence,
        "method": "skeleton_chain_metric",
        "not_direct_stature_measurement": True,
        "candidate_height_cm": height_m * 100.0,
        "measurement_quality": {
            "valid_keypoint_count": int(valid_point_count),
            "valid_leg_count": int(len(leg_chains)),
            "leg_chain_m": leg_chains,
            "leg_ankle_indices": leg_indices,
            "leg_symmetry_quality": symmetry_quality,
            "head_to_shoulders_m": head_to_shoulders,
            "torso_m": torso,
            "crown_to_face_m": crown_face,
            "foot_extension_m": foot_extension,
            "rejected_depth_keypoint_indices": rejected_depth_indices,
        },
    }


def _direct_lying_estimate(
    depth_m: np.ndarray,
    result: dict[str, Any],
    intrinsics: tuple[float, float, float, float],
    *,
    support_plane: Mapping[str, Any] | None = None,
    depth_scale_correction: float = 1.0,
    calibration_rmse_m: float = 0.0,
) -> dict[str, Any]:
    """Measure supine/prone body length along a bed or support plane.

    This is intentionally separate from standing stature. A lying subject is
    measured from a head endpoint to a foot endpoint after projecting both
    endpoints onto the support plane. If the pose is bent, the direct endpoint
    distance is not a valid body-length measurement and is rejected.
    """
    keypoints = _keypoints(result)
    image_height, image_width = depth_m.shape
    bbox = _bbox_from_result(result)
    if bbox is None:
        return {"status": "no_selected_person"}
    x0, y0, x1, y1 = bbox
    edge_margin = max(5, int(round(min(image_height, image_width) * 0.012)))
    if (
        x0 <= edge_margin
        or y0 <= edge_margin
        or x1 >= image_width - edge_margin
        or y1 >= image_height - edge_margin
    ):
        return {
            "status": "person_touches_image_edge",
            "guidance": "平躺长度测量需要头顶和双脚完整入镜，并与画面边缘保留空白。",
            "measurement_quality": {
                "bbox_xyxy": [x0, y0, x1, y1],
                "frame_edge_margin_px": int(edge_margin),
            },
        }

    raw_reference_depth, source_iqr, source_valid_fraction = _target_depth(result)
    reference_depth, reference_quality = _pose_depth_reference(
        depth_m,
        keypoints,
        intrinsics,
        tracker_depth_m=raw_reference_depth,
    )
    if reference_depth is None:
        reference_depth = float(np.nanmedian(depth_m))
        if not np.isfinite(reference_depth) or not _MIN_DEPTH_M <= reference_depth <= _MAX_DEPTH_M:
            return {"status": "insufficient_valid_depth"}

    points: list[np.ndarray | None] = []
    rejected_depth_indices: list[int] = []
    for index, keypoint in enumerate(keypoints):
        point = _point_from_keypoint(depth_m, keypoint, intrinsics)
        if point is not None:
            gap = abs(float(point[2]) - float(reference_depth))
            if gap > max(0.42, 0.30 * float(reference_depth)):
                rejected_depth_indices.append(index)
                point = None
        points.append(point)

    face = _mean_points([points[index] for index in range(min(5, len(points)))])
    shoulders = _mean_points([points[index] for index in (5, 6) if index < len(points)])
    hips = _mean_points([points[index] for index in (11, 12) if index < len(points)])
    if face is None or shoulders is None or hips is None:
        return {
            "status": "insufficient_pose_keypoints",
            "measurement_quality": {
                "rejected_depth_keypoint_indices": rejected_depth_indices,
            },
        }

    leg_entries: list[dict[str, Any]] = []
    for side, hip_index, knee_index, ankle_index in (
        ("left", 11, 13, 15),
        ("right", 12, 14, 16),
    ):
        if max(hip_index, knee_index, ankle_index) >= len(points):
            continue
        hip, knee, ankle = points[hip_index], points[knee_index], points[ankle_index]
        if hip is None or knee is None or ankle is None:
            continue
        leg_entries.append(
            {
                "side": side,
                "hip": hip,
                "knee": knee,
                "ankle": ankle,
                "leg_chain_m": float(np.linalg.norm(hip - knee) + np.linalg.norm(knee - ankle)),
            }
        )
    if not leg_entries:
        return {
            "status": "insufficient_leg_chain",
            "measurement_quality": {
                "rejected_depth_keypoint_indices": rejected_depth_indices,
            },
        }

    body_points = [face, shoulders, hips]
    body_points.extend(
        point
        for entry in leg_entries
        for point in (entry["hip"], entry["knee"], entry["ankle"])
    )
    provided_plane = _validated_support_plane(support_plane)
    if provided_plane is None:
        plane = _estimate_support_plane(depth_m, intrinsics, bbox, body_points)
    else:
        normal, plane_offset, plane = provided_plane
        plane = dict(plane)
        plane["status"] = "ok"
        plane["fit_method"] = "provided_calibrated_support_plane"
    if plane.get("status") != "ok":
        return {
            "status": "support_plane_not_confirmed",
            "guidance": "未确认床面/支撑面，请让床面在人体两侧露出，或先建立床面标定。",
            "measurement_quality": {
                "support_plane": plane,
                "reference_depth": reference_quality,
                "rejected_depth_keypoint_indices": rejected_depth_indices,
            },
        }

    normal = np.asarray(plane["normal_camera"], dtype=np.float32)
    plane_offset = float(plane["plane_offset_m"])
    plane_gaps = np.abs(_plane_signed_distances(np.asarray(body_points), normal, plane_offset))
    median_body_gap = float(np.median(plane_gaps))
    max_body_gap = float(np.max(plane_gaps))
    if median_body_gap > 0.30 or max_body_gap > 0.65:
        return {
            "status": "body_support_plane_disagree",
            "guidance": "人体关键点与床面三维关系不一致，请检查床面、遮挡和深度有效性。",
            "measurement_quality": {
                "support_plane": plane,
                "body_to_plane_gaps_m": plane_gaps.tolist(),
                "median_body_to_plane_m": median_body_gap,
                "max_body_to_plane_m": max_body_gap,
            },
        }

    ankle_center = np.mean(
        np.stack([entry["ankle"] for entry in leg_entries], axis=0), axis=0
    )
    axis = _unit_vector(_project_to_plane(ankle_center - face, normal))
    if axis is None:
        return {"status": "body_axis_not_confirmed"}
    head_to_shoulder_m = float(np.linalg.norm(face - shoulders))
    torso_m = float(np.linalg.norm(shoulders - hips))
    if head_to_shoulder_m < 0.05 or torso_m < 0.10:
        return {"status": "implausible_pose_segments"}
    head_extension_m = float(np.clip(0.55 * head_to_shoulder_m, 0.09, 0.18))
    head_endpoint = face - axis * head_extension_m

    foot_endpoints: list[np.ndarray] = []
    for entry in leg_entries:
        leg_direction = _unit_vector(
            _project_to_plane(entry["ankle"] - entry["knee"], normal)
        )
        if leg_direction is None:
            leg_direction = axis
        foot_endpoint = entry["ankle"] + leg_direction * _LYING_FOOT_EXTENSION_M
        entry["foot_endpoint_camera_m"] = foot_endpoint.tolist()
        foot_endpoints.append(foot_endpoint)
    foot_endpoint_center = np.mean(np.stack(foot_endpoints, axis=0), axis=0)
    direct_lengths = [
        float(np.linalg.norm(_project_to_plane(foot - head_endpoint, normal)))
        for foot in foot_endpoints
    ]
    direct_length_m = float(np.median(direct_lengths))
    chain_lengths = [
        head_extension_m
        + head_to_shoulder_m
        + torso_m
        + float(entry["leg_chain_m"])
        + _LYING_FOOT_EXTENSION_M
        for entry in leg_entries
    ]
    chain_length_m = float(np.mean(chain_lengths))
    straightness = float(np.clip(direct_length_m / max(chain_length_m, 1e-6), 0.0, 1.0))
    torso_axis = _unit_vector(_project_to_plane(hips - shoulders, normal))
    body_axis_alignment = (
        abs(float(np.dot(torso_axis, axis))) if torso_axis is not None else 0.0
    )
    endpoint_spread_m = float(
        max(
            np.linalg.norm(_project_to_plane(foot - foot_endpoint_center, normal))
            for foot in foot_endpoints
        )
    )
    quality: dict[str, Any] = {
        "algorithm": SCHEMA,
        "measurement_quantity": "supine_body_length",
        "support_plane": plane,
        "reference_depth": reference_quality,
        "target_depth_m": reference_depth,
        "target_depth_raw_m": raw_reference_depth,
        "target_depth_valid_fraction": source_valid_fraction,
        "body_to_plane_gaps_m": plane_gaps.tolist(),
        "median_body_to_plane_m": median_body_gap,
        "max_body_to_plane_m": max_body_gap,
        "head_endpoint_camera_m": head_endpoint.tolist(),
        "foot_endpoints_camera_m": [foot.tolist() for foot in foot_endpoints],
        "head_extension_m": head_extension_m,
        "head_to_shoulder_m": head_to_shoulder_m,
        "torso_m": torso_m,
        "leg_chain_m": [float(entry["leg_chain_m"]) for entry in leg_entries],
        "chain_length_m": chain_length_m,
        "direct_length_per_leg_m": direct_lengths,
        "direct_length_m": direct_length_m,
        "straightness": straightness,
        "body_axis_alignment": body_axis_alignment,
        "foot_endpoint_spread_m": endpoint_spread_m,
        "valid_leg_count": len(leg_entries),
        "rejected_depth_keypoint_indices": rejected_depth_indices,
        "depth_calibration_rmse_cm": float(calibration_rmse_m) * 100.0,
    }
    if straightness < _LYING_MIN_STRAIGHTNESS:
        quality["body_length_not_measurable"] = True
        return {
            "status": "lying_body_not_straight",
            "guidance": "平躺身体未充分伸直，尤其是膝盖弯曲时不能把头脚直线当作身高。",
            "diagnostic_body_length_cm": direct_length_m * 100.0,
            "measurement_quality": quality,
        }

    plane_iqr_m = float(plane.get("residual_iqr_m", 0.03) or 0.03)
    depth_iqr_m = float(source_iqr or 0.03)
    plane_quality = float(np.exp(-plane_iqr_m / 0.035))
    endpoint_quality = float(np.exp(-depth_iqr_m / 0.05))
    straightness_quality = float(np.clip(straightness, 0.0, 1.0))
    confidence = float(
        np.clip(
            0.28
            + 0.24 * plane_quality
            + 0.18 * endpoint_quality
            + 0.18 * straightness_quality
            + 0.08 * body_axis_alignment
            + 0.04 * float(len(leg_entries) == 2),
            0.0,
            0.94,
        )
    )
    uncertainty_cm = float(
        np.clip(
            1.5
            + 100.0 * plane_iqr_m
            + 70.0 * depth_iqr_m
            + (1.0 - straightness) * 35.0
            + 3.0 * float(len(leg_entries) == 1)
            + 100.0 * calibration_rmse_m,
            3.0,
            18.0,
        )
    )
    height_cm = direct_length_m * 100.0
    quality["confidence_components"] = {
        "plane_quality": plane_quality,
        "endpoint_quality": endpoint_quality,
        "straightness_quality": straightness_quality,
    }
    return {
        "status": "ok",
        "height_cm": height_cm,
        "height_lower_cm": height_cm - uncertainty_cm,
        "height_upper_cm": height_cm + uncertainty_cm,
        "confidence": confidence,
        "method": "direct_metric_support_plane_body_length",
        "measurement_quantity": "supine_body_length",
        "not_direct_stature_measurement": True,
        "measurement_quality": quality,
    }


def _direct_standing_estimate(
    depth_m: np.ndarray,
    result: dict[str, Any],
    intrinsics: tuple[float, float, float, float],
    *,
    color_bgr: np.ndarray | None = None,
    depth_scale_correction: float = 1.0,
    depth_offset_correction_m: float = 0.0,
    calibration_rmse_m: float = 0.0,
) -> dict[str, Any]:
    keypoints = _keypoints(result)
    image_height, image_width = depth_m.shape
    bbox_result, bbox_quality = _expanded_measurement_bbox(
        result,
        (image_height, image_width),
        keypoints,
    )
    if bbox_result is None:
        return bbox_quality
    x0, y0, x1, y1 = bbox_result
    if x1 - x0 < 30 or y1 - y0 < 80:
        return {"status": "person_box_too_small", "bbox": bbox_quality}

    # Only fill small enclosed holes. Large holes are kept invalid so a person
    # cannot be joined to a wall or floor by an aggressive interpolation.
    working_depth, filled_hole_pixels = _fill_small_depth_holes(depth_m)
    crop = working_depth[y0:y1, x0:x1]
    valid = np.isfinite(crop) & (crop >= _MIN_DEPTH_M) & (crop <= _MAX_DEPTH_M)
    if int(valid.sum()) < 100:
        return {"status": "insufficient_valid_depth", "bbox": bbox_quality}

    raw_reference_depth, source_iqr, source_valid_fraction = _target_depth(result)
    if source_iqr is not None:
        source_iqr *= abs(float(depth_scale_correction))
    tracker_reference_depth = _correct_depth_value(
        raw_reference_depth,
        depth_scale_correction,
        depth_offset_correction_m,
    )
    reference_depth, reference_quality = _pose_depth_reference(
        working_depth,
        keypoints,
        intrinsics,
        tracker_depth_m=tracker_reference_depth,
    )
    if reference_depth is None:
        reference_depth = float(np.percentile(crop[valid], 25))
        reference_quality["source"] = "person_crop_percentile_fallback"
    tolerance = float(
        np.clip(
            max(0.18, 0.14 * float(reference_depth), 2.5 * float(source_iqr or 0.0)),
            0.18,
            0.45,
        )
    )
    foreground = valid & (np.abs(crop - reference_depth) <= tolerance)
    pose_depth_foreground, pose_depth_anchor_count = _pose_depth_supported_foreground(
        crop,
        valid,
        keypoints,
        x_offset=x0,
        y_offset=y0,
    )
    foreground |= pose_depth_foreground
    if int(foreground.sum()) < 80:
        return {"status": "target_depth_not_separable", "bbox": bbox_quality}

    binary = foreground.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
    if count <= 1:
        return {"status": "body_component_not_found", "bbox": bbox_quality}
    crop_height, crop_width = crop.shape
    seed = np.zeros_like(binary, dtype=bool)
    seed[
        int(crop_height * 0.22): int(crop_height * 0.78),
        int(crop_width * 0.20): int(crop_width * 0.80),
    ] = True
    best_label = 0
    best_score = -1.0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 50:
            continue
        component = labels == label
        score = float((component & seed).sum()) * 2.0 + float(area) * 0.05
        if score > best_score:
            best_score = score
            best_label = label
    if best_label == 0:
        return {"status": "body_component_not_found", "bbox": bbox_quality}
    component_labels = {best_label}
    anchored_labels, anchored_keypoint_count = _pose_anchored_component_labels(
        labels,
        stats,
        keypoints,
        x_offset=x0,
        y_offset=y0,
    )
    component_labels.update(anchored_labels)
    body_mask = foreground & np.isin(labels, list(component_labels))
    row_counts = body_mask.sum(axis=1)
    row_minimum = max(4, int(round(crop_width * 0.01)))
    occupied_rows = np.flatnonzero(row_counts >= row_minimum)
    if len(occupied_rows) < 20:
        return {"status": "body_vertical_extent_too_short", "bbox": bbox_quality}

    face_points = [
        np.asarray(point[:2], dtype=np.float32)
        for index, point in _in_frame_keypoints(keypoints, (image_height, image_width))
        if index < 5
    ]
    head = _mean_points(face_points)
    if head is None:
        return {
            "status": "head_not_confirmed",
            "guidance": "请保持正面站立并让头顶清晰可见，再进行身高测量。",
            "bbox": bbox_quality,
        }
    head_x = float(head[0]) - float(x0)
    head_y = float(head[1]) - float(y0)
    head_half_width = max(crop_width * 0.20, 18.0)
    top_columns = np.zeros_like(body_mask, dtype=bool)
    x_start = max(0, int(np.floor(head_x - head_half_width)))
    x_end = min(crop_width, int(np.ceil(head_x + head_half_width)))
    y_end = min(crop_height, int(np.ceil(head_y + crop_height * 0.20)))
    top_columns[:y_end, x_start:x_end] = True
    head_rows = np.flatnonzero((body_mask & top_columns).sum(axis=1) >= row_minimum)
    top_row = int(head_rows[0]) if len(head_rows) else int(occupied_rows[0])
    body_bottom_row = int(occupied_rows[-1])
    observed_top_image_y = y0 + top_row
    body_bottom_image_y = y0 + body_bottom_row

    band = max(3, int(round(crop_height * 0.025)))

    def endpoint(row: int, upper: bool) -> tuple[np.ndarray, float, int] | None:
        if upper:
            available = body_mask[row: min(crop_height, row + band), :]
            row_counts = available.sum(axis=1)
            supported_rows = np.flatnonzero(row_counts >= 3)
            edge_row = row + int(supported_rows[0]) if len(supported_rows) else row
            selection = body_mask[
                edge_row : min(crop_height, edge_row + min(3, band)),
                :,
            ]
            yy, xx = np.where(selection)
            yy = yy + edge_row
        else:
            lower = max(0, row - band + 1)
            selection = body_mask[lower: row + 1, :]
            yy, xx = np.where(selection)
            yy = yy + lower
        if len(xx) < 10:
            return None
        return _robust_point_from_pixels(
            working_depth,
            yy + y0,
            xx + x0,
            intrinsics,
            min_count=10,
        )

    top_endpoint = endpoint(top_row, upper=True)
    if top_endpoint is None:
        return {"status": "insufficient_top_depth", "bbox": bbox_quality}
    top_endpoint_point, top_endpoint_iqr, top_endpoint_count = top_endpoint
    body_mask_full = np.zeros_like(working_depth, dtype=bool)
    body_mask_full[y0:y1, x0:x1] = body_mask
    crown = _head_crown_candidate(
        working_depth,
        keypoints,
        intrinsics,
        body_mask=body_mask_full,
        observed_top_image_y=observed_top_image_y,
    )
    if crown is not None:
        top_point = np.asarray(crown["point_camera_m"], dtype=np.float32)
        top_iqr = float(crown.get("depth_iqr_m", top_endpoint_iqr) or top_endpoint_iqr)
        top_count = int(crown.get("point_count", 0) or 0)
        top_image_y = int(crown.get("top_image_y", observed_top_image_y))
        top_source = str(crown.get("source") or "head_crown_candidate")
        top_confidence = float(np.clip(crown.get("confidence", 0.0), 0.0, 1.0))
    else:
        top_point = top_endpoint_point
        top_iqr = top_endpoint_iqr
        top_count = top_endpoint_count
        top_image_y = observed_top_image_y
        top_source = "body_mask_top_endpoint"
        top_confidence = 0.42

    rgb_crown = _rgb_head_crown_candidate(color_bgr, keypoints)
    if rgb_crown is not None:
        facial_depth_points = [
            _point_from_keypoint(working_depth, keypoints[index], intrinsics)
            for index in range(min(5, len(keypoints)))
        ]
        facial_depth_points = [
            point for point in facial_depth_points if point is not None
        ]
        if facial_depth_points:
            facial_depth_values = np.asarray(
                [float(point[2]) for point in facial_depth_points],
                dtype=np.float32,
            )
            rgb_depth_m = float(np.median(facial_depth_values))
            rgb_depth_iqr_m = float(
                np.diff(np.percentile(facial_depth_values, [25.0, 75.0]))[0]
            )
            depth_is_same_person = reference_depth is None or abs(
                rgb_depth_m - float(reference_depth)
            ) <= max(0.28, 0.20 * rgb_depth_m)
            rgb_top_image_y = int(rgb_crown["top_image_y"])
            # A color boundary may replace a missing crown depth point, but it
            # must not move the endpoint down below a better depth observation.
            rgb_is_not_lower = rgb_top_image_y <= top_image_y + max(
                2,
                int(round(0.08 * float(rgb_crown["head_width_pixels"]))),
            )
            if depth_is_same_person and rgb_is_not_lower:
                fx, fy, cx, cy = intrinsics
                rgb_center_x = float(rgb_crown["center_x"])
                top_point = np.asarray(
                    [
                        (rgb_center_x - cx) * rgb_depth_m / fx,
                        (rgb_top_image_y - cy) * rgb_depth_m / fy,
                        rgb_depth_m,
                    ],
                    dtype=np.float32,
                )
                # RGB supplies the endpoint row; facial depth supplies metric
                # scale. Include a small row-localization uncertainty because
                # the hair silhouette is not a depth measurement itself.
                row_uncertainty_m = max(0.008, 0.40 * rgb_depth_m / fy)
                top_iqr = max(rgb_depth_iqr_m, row_uncertainty_m)
                top_count = int(rgb_crown.get("component_area", 0) or 0)
                top_image_y = rgb_top_image_y
                top_source = "rgb_head_silhouette_with_facial_depth"
                top_confidence = float(
                    np.clip(
                        0.52
                        + 0.12 * min(1.0, float(rgb_crown.get("score", 0.0)) / 20.0)
                        + 0.12 * np.exp(-rgb_depth_iqr_m / 0.04)
                        + 0.08 * float(depth_is_same_person),
                        0.0,
                        0.90,
                    )
                )

    edge_margin = max(8, int(round((y1 - y0) * 0.025)))
    if top_image_y <= edge_margin:
        return {
            "status": "person_touches_image_edge",
            "guidance": "站立测量需要头顶和双脚完整入镜，并与画面上下边缘保留间距。",
            "measurement_quality": {
                "bbox": bbox_quality,
                "top_image_y": int(top_image_y),
                "frame_edge_margin_px": int(edge_margin),
            },
        }

    feet = _confirmed_feet(
        working_depth,
        keypoints,
        intrinsics,
        bottom_image_y=body_bottom_image_y,
        edge_margin=edge_margin,
        reference_depth_m=reference_depth,
        body_mask=body_mask_full,
        person_height_px=float(y1 - y0),
        person_width_px=float(x1 - x0),
    )
    quality: dict[str, Any] = {
        "algorithm": "gemini_human_height_estimate.v4",
        "vertical_coverage": float((body_bottom_row - top_row + 1) / max(crop_height, 1)),
        "endpoint_depth_iqr_cm": (top_iqr + top_endpoint_iqr) * 50.0,
        "target_depth_m": reference_depth,
        "target_depth_raw_m": raw_reference_depth,
        "target_depth_window_m": tolerance,
        "target_depth_valid_fraction": source_valid_fraction,
        "reference_depth": reference_quality,
        "bbox": bbox_quality,
        "bbox_image_y": [int(y0), int(y1)],
        "frame_edge_margin_px": int(edge_margin),
        "top_image_y": int(top_image_y),
        "observed_body_top_image_y": int(observed_top_image_y),
        "body_bottom_image_y": int(body_bottom_image_y),
        # Compatibility alias for earlier experiment reports.
        "bottom_image_y": int(body_bottom_image_y),
        "top_source": top_source,
        "top_confidence": top_confidence,
        "top_crown": crown,
        "rgb_crown": rgb_crown,
        "pose_anchored_component_count": int(len(component_labels)),
        "pose_anchor_count": int(anchored_keypoint_count),
        "pose_depth_anchor_count": int(pose_depth_anchor_count),
        "top_point_camera_m": top_point.tolist(),
        "feet": feet,
        "depth_hole_fill_pixels": int(filled_hole_pixels),
        "depth_calibration_rmse_cm": float(calibration_rmse_m) * 100.0,
    }
    skeleton = _skeleton_height_candidate(
        working_depth,
        keypoints,
        intrinsics,
        reference_depth_m=reference_depth,
        crown_point=top_point,
        feet=feet,
    )
    quality["skeleton_candidate"] = skeleton
    if not feet.get("confirmed"):
        return {
            "status": "feet_not_confirmed",
            "guidance": "请后退或调整相机，让双脚、脚踝和脚下地面完整进入画面。",
            "measurement_quality": quality,
        }

    sole_points = [
        np.asarray(item["sole_camera_point_m"], dtype=np.float32)
        for item in feet.get("ankles", [])
        if isinstance(item, Mapping) and item.get("sole_camera_point_m") is not None
    ]
    ankle_points = [
        np.asarray(item["ankle_camera_point_m"], dtype=np.float32)
        for item in feet.get("ankles", [])
        if isinstance(item, Mapping) and item.get("ankle_camera_point_m") is not None
    ]
    if not sole_points or not ankle_points:
        return {
            "status": "insufficient_foot_geometry",
            "measurement_quality": quality,
        }
    sole_center = np.mean(np.stack(sole_points, axis=0), axis=0)
    body_axis_reference = top_point - sole_center
    no_floor_extent_m = float(np.linalg.norm(body_axis_reference))

    floor = _estimate_floor_plane(
        working_depth,
        intrinsics,
        body_mask_full,
        target_depth_m=reference_depth,
        foot_points=sole_points + ankle_points,
        sole_count=len(sole_points),
    )
    quality["ground_plane"] = floor
    quality["candidate_heights_cm"] = {
        "top_to_sole_euclidean": no_floor_extent_m * 100.0,
    }
    if floor.get("status") != "ok":
        if skeleton.get("status") == "ok":
            quality["candidate_heights_cm"]["skeleton_chain_metric"] = float(
                skeleton["height_cm"]
            )
        return {
            "status": "floor_plane_not_confirmed",
            "guidance": (
                "未确认脚下地面平面，因此不显示可能截断的身高。请让头顶、双脚和双脚两侧的地面完整入镜，"
                "并保持站立不动。"
            ),
            "diagnostic_direct_metric_extent_cm": no_floor_extent_m * 100.0,
            "measurement_quality": quality,
        }

    normal = np.asarray(floor["normal_camera"], dtype=np.float32)
    plane_offset = float(floor["plane_offset_m"])
    floor_height_m = abs(float(np.dot(normal, top_point)) - plane_offset)
    sole_plane_gaps_m = [
        abs(float(np.dot(normal, point)) - plane_offset) for point in sole_points
    ]
    ankle_plane_gaps_m = [
        abs(float(np.dot(normal, point)) - plane_offset) for point in ankle_points
    ]
    median_sole_gap_m = float(np.median(sole_plane_gaps_m))
    nearest_sole_gap_m = float(np.min(sole_plane_gaps_m))
    nearest_ankle_gap_m = float(np.min(ankle_plane_gaps_m))
    body_axis_length_m = float(np.linalg.norm(body_axis_reference))
    body_axis_alignment = (
        abs(float(np.dot(body_axis_reference / body_axis_length_m, normal)))
        if body_axis_length_m > 1e-6
        else 0.0
    )
    projected_extent_m = abs(float(np.dot(normal, body_axis_reference)))
    extent_difference_m = abs(floor_height_m - projected_extent_m)
    floor["head_to_plane_m"] = floor_height_m
    floor["median_sole_to_plane_m"] = median_sole_gap_m
    floor["nearest_sole_to_plane_m"] = nearest_sole_gap_m
    floor["nearest_ankle_to_plane_m"] = nearest_ankle_gap_m
    floor["body_axis_alignment"] = body_axis_alignment
    floor["plane_vs_projected_extent_difference_m"] = extent_difference_m
    quality["candidate_heights_cm"].update(
        {
            "head_to_ground_plane": floor_height_m * 100.0,
            "top_to_sole_vertical_projection": projected_extent_m * 100.0,
        }
    )
    target_depth_quality = float(
        np.clip(
            source_valid_fraction if source_valid_fraction is not None else 0.7,
            0.0,
            1.0,
        )
    )
    endpoint_fraction = min(
        1.0,
        min(max(top_count, 3), max(10, top_endpoint_count))
        / max(10.0, crop_width * band * 0.10),
    )
    top_depth_stability = float(np.exp(-top_iqr / 0.05))
    sole_depth_stability = float(
        np.exp(
            -float(
                np.median(
                    [
                        float(item.get("sole_depth_iqr_m", 0.05) or 0.05)
                        for item in feet.get("ankles", [])
                    ]
                )
            )
            / 0.05
        )
    )
    floor_valid = (
        _HEIGHT_MIN_M <= floor_height_m <= _HEIGHT_MAX_M
        and median_sole_gap_m <= 0.12
        and nearest_ankle_gap_m <= 0.34
        and body_axis_alignment >= 0.70
        and extent_difference_m <= 0.18
    )
    if not floor_valid:
        return {
            "status": "floor_body_geometry_disagree",
            "guidance": "地面、脚部和人体轴线的三维证据不一致，请保持正面站立并重新取景。",
            "diagnostic_direct_metric_extent_cm": projected_extent_m * 100.0,
            "measurement_quality": quality,
        }

    plane_iqr_m = float(floor.get("residual_iqr_m", 0.03) or 0.03)
    foot_fallback_fraction = float(
        np.mean(
            [
                bool(item.get("used_ankle_as_sole"))
                for item in feet.get("ankles", [])
            ]
        )
    )
    floor_confidence = float(
        np.clip(
            0.25
            + 0.15 * float(floor.get("score", 0.0) or 0.0)
            + 0.12 * top_confidence
            + 0.12 * top_depth_stability
            + 0.10 * sole_depth_stability
            + 0.10 * float(np.exp(-plane_iqr_m / 0.035))
            + 0.08 * float(np.exp(-median_sole_gap_m / 0.06))
            + 0.05 * body_axis_alignment
            + 0.03 * target_depth_quality
            - 0.08 * foot_fallback_fraction,
            0.0,
            0.96,
        )
    )
    floor_uncertainty_cm = float(
        np.clip(
            2.0
            + 100.0 * (top_iqr + plane_iqr_m)
            + 40.0 * median_sole_gap_m
            + (1.0 - floor_confidence) * 10.0
            + 6.0 * foot_fallback_fraction,
            3.0,
            25.0,
        )
    )
    floor_uncertainty_cm = float(
        np.sqrt(floor_uncertainty_cm**2 + (100.0 * calibration_rmse_m) ** 2)
    )
    quality["direct_metric_floor_confidence"] = floor_confidence
    quality["direct_metric_floor_uncertainty_cm"] = floor_uncertainty_cm
    quality["direct_metric_projected_extent_confidence"] = float(
        np.clip(floor_confidence * 0.94, 0.0, 1.0)
    )

    quality["candidate_heights_cm"]["skeleton_chain_metric"] = (
        float(skeleton["height_cm"])
        if skeleton.get("status") == "ok"
        else None
    )
    if skeleton.get("status") == "ok":
        skeleton_height_m = float(skeleton["height_cm"]) / 100.0
        candidate_difference_m = abs(floor_height_m - skeleton_height_m)
        quality["candidate_height_difference_cm"] = candidate_difference_m * 100.0
        if candidate_difference_m > _MAX_CANDIDATE_DISAGREEMENT_M:
            return {
                "status": "height_candidates_disagree",
                "guidance": "头顶/地面测量与关键点骨骼链不一致，请保持不动重新采集多帧。",
                "measurement_quality": quality,
            }
        agreement = float(np.exp(-candidate_difference_m / 0.06))
        skeleton_confidence = float(np.clip(skeleton.get("confidence", 0.0), 0.0, 1.0))
        direct_weight = max(0.05, floor_confidence * (0.70 + 0.30 * top_confidence))
        # A pose-chain estimate is useful as a sanity check, but it is less
        # reliable than a confirmed floor plane for upright stature: COCO
        # joints are not crown/sole landmarks and each depth sample can sit on
        # clothing or an occluding surface. Keep it as a small stabilizing
        # contribution instead of allowing it to pull a good plane measurement
        # toward a systematic skeletal under-estimate.
        skeleton_weight = 0.03 * skeleton_confidence * agreement
        if floor_confidence < 0.55 or top_confidence < 0.50:
            skeleton_weight = 0.08 * skeleton_confidence * agreement
        final_height_m = (
            direct_weight * floor_height_m + skeleton_weight * skeleton_height_m
        ) / (direct_weight + skeleton_weight)
        final_confidence = float(
            np.clip(
                0.52 * floor_confidence
                + 0.22 * skeleton_confidence
                + 0.26 * agreement,
                0.0,
                0.96,
            )
        )
        uncertainty_cm = float(
            np.sqrt(
                floor_uncertainty_cm**2
                + (100.0 * candidate_difference_m * 0.50) ** 2
            )
        )
        quality["candidate_fusion"] = {
            "method": "quality_weighted_floor_and_skeleton_consensus",
            "floor_height_cm": floor_height_m * 100.0,
            "skeleton_height_cm": skeleton_height_m * 100.0,
            "floor_weight": direct_weight,
            "skeleton_weight": skeleton_weight,
            "skeleton_used_as_primary": bool(skeleton_weight >= direct_weight),
            "agreement_score": agreement,
        }
    else:
        final_height_m = floor_height_m
        final_confidence = floor_confidence
        uncertainty_cm = floor_uncertainty_cm
        quality["candidate_fusion"] = {
            "method": "floor_only_skeleton_unavailable",
            "floor_height_cm": floor_height_m * 100.0,
        }
    quality["candidate_heights_cm"]["final_consensus"] = final_height_m * 100.0
    return {
        "status": "ok",
        "height_cm": final_height_m * 100.0,
        "height_lower_cm": final_height_m * 100.0 - uncertainty_cm,
        "height_upper_cm": final_height_m * 100.0 + uncertainty_cm,
        "confidence": final_confidence,
        "method": "direct_metric_floor_plane",
        "measurement_quality": quality,
    }


def _pose_conditioned_estimate(
    depth_m: np.ndarray,
    result: dict[str, Any],
    intrinsics: tuple[float, float, float, float],
    *,
    calibration_rmse_m: float = 0.0,
) -> dict[str, Any]:
    keypoints = _keypoints(result)
    if len(keypoints) < 17:
        return {"status": "insufficient_pose_keypoints"}
    points = [_point_from_keypoint(depth_m, item, intrinsics) for item in keypoints]
    face = _mean_points(points[index] for index in (0, 1, 2, 3, 4))
    shoulders = _mean_points(points[index] for index in (5, 6))
    hips = _mean_points(points[index] for index in (11, 12))
    if face is None or shoulders is None or hips is None:
        return {"status": "insufficient_pose_keypoints"}
    head_to_shoulders = float(np.linalg.norm(face - shoulders))
    torso = float(np.linalg.norm(shoulders - hips))
    leg_lengths: list[float] = []
    for hip_index, knee_index, ankle_index in ((11, 13, 15), (12, 14, 16)):
        hip, knee, ankle = points[hip_index], points[knee_index], points[ankle_index]
        if hip is None or knee is None or ankle is None:
            continue
        leg_lengths.append(float(np.linalg.norm(hip - knee) + np.linalg.norm(knee - ankle)))
    if not leg_lengths or head_to_shoulders < 0.05 or torso < 0.10:
        return {"status": "insufficient_pose_keypoints"}
    # The crown is not a COCO landmark.  This small, explicit anatomy prior is
    # why non-standing output is labelled a pose-conditioned prediction.
    crown_extension = float(np.clip(0.45 * head_to_shoulders, 0.07, 0.16))
    height_m = head_to_shoulders + crown_extension + torso + float(np.mean(leg_lengths)) + 0.03
    if not 0.65 <= height_m <= 2.50:
        return {"status": "height_outside_expected_range", "raw_height_cm": height_m * 100.0}
    segment_count = 3 + len(leg_lengths) * 2
    confidence = float(np.clip(0.22 + 0.045 * segment_count, 0.0, 0.58))
    uncertainty_cm = float(
        np.clip(
            14.0 + (2 - len(leg_lengths)) * 4.0 + (0.58 - confidence) * 12.0,
            14.0,
            30.0,
        )
    )
    uncertainty_cm = float(
        np.sqrt(uncertainty_cm**2 + (100.0 * calibration_rmse_m) ** 2)
    )
    return {
        "status": "ok",
        "height_cm": height_m * 100.0,
        "height_lower_cm": height_m * 100.0 - uncertainty_cm,
        "height_upper_cm": height_m * 100.0 + uncertainty_cm,
        "confidence": confidence,
        "method": "pose_conditioned",
        "not_direct_stature_measurement": True,
        "measurement_quality": {
            "valid_leg_count": len(leg_lengths),
            "head_to_shoulders_m": head_to_shoulders,
            "torso_m": torso,
            "mean_leg_chain_m": float(np.mean(leg_lengths)),
            "crown_extension_prior_m": crown_extension,
            "depth_calibration_rmse_cm": float(calibration_rmse_m) * 100.0,
        },
    }


def estimate_human_height(
    *,
    depth_raw: np.ndarray,
    depth_scale_m: float,
    intrinsics: dict[str, Any],
    result: dict[str, Any],
    posture: str | None,
    color_bgr: np.ndarray | None = None,
    imu: dict[str, Any] | None = None,
    depth_correction: Mapping[str, Any] | None = None,
    support_plane: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a depth/intrinsics-only height estimate or a reason to reject it."""
    calibration, correction_scale, correction_offset_m, calibration_rmse_m = (
        _prepare_depth_calibration(depth_correction)
    )
    output = _base_result(posture, calibration)
    camera = _intrinsics(intrinsics)
    if camera is None:
        output["status"] = "invalid_intrinsics"
        return output
    if not isinstance(depth_raw, np.ndarray) or depth_raw.ndim != 2:
        output["status"] = "invalid_depth_image"
        return output
    try:
        scale = float(depth_scale_m)
    except (TypeError, ValueError):
        scale = 0.0
    if not np.isfinite(scale) or scale <= 0:
        output["status"] = "invalid_depth_scale"
        return output
    raw_depth_m = depth_raw.astype(np.float32, copy=False) * scale
    if calibration["applied"]:
        depth_m = raw_depth_m.copy()
        valid_raw = np.isfinite(raw_depth_m) & (raw_depth_m > 0.0)
        depth_m[valid_raw] = (
            depth_m[valid_raw] * correction_scale + correction_offset_m
        )
        output["source"] = (
            "gemini_aligned_depth_and_intrinsics_with_fixed_station_calibration"
        )
    else:
        depth_m = raw_depth_m
    if _is_standing(posture):
        # Raw Gemini IMU axes are deliberately not used here: their transform
        # into the aligned-depth camera frame has not been calibrated.
        estimate = _direct_standing_estimate(
            depth_m,
            result,
            camera,
            color_bgr=color_bgr,
            depth_scale_correction=correction_scale,
            depth_offset_correction_m=correction_offset_m,
            calibration_rmse_m=calibration_rmse_m,
        )
    elif _is_lying(posture):
        estimate = _direct_lying_estimate(
            depth_m,
            result,
            camera,
            support_plane=support_plane,
            depth_scale_correction=correction_scale,
            calibration_rmse_m=calibration_rmse_m,
        )
    else:
        estimate = _pose_conditioned_estimate(
            depth_m,
            result,
            camera,
            calibration_rmse_m=calibration_rmse_m,
        )
    output.update(estimate)
    return output
