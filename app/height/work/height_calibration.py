"""Persistent fixed-station depth calibration for Gemini height measurement.

This module deliberately calibrates the camera with a rigid planar target at
known distances.  It never accepts a person's height as an input.  The fitted
correction is scoped to the separate stature-estimation path; body-region
inference continues to use Gemini's original depth scale.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA = "gemini_height_depth_calibration.v1"
REQUEST_SCHEMA = "gemini_height_depth_correction_request.v1"
MIN_SAMPLES = 3
MIN_DISTANCE_SPAN_M = 0.40
MIN_BOARD_DISTANCE_M = 0.40
MAX_BOARD_DISTANCE_M = 6.00
MAX_FRAME_IQR_M = 0.025
MIN_FRAME_VALID_FRACTION = 0.80
INTRINSICS_RELATIVE_TOLERANCE = 0.005
PRINCIPAL_POINT_TOLERANCE_PX = 2.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_name(value: Any, fallback: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "").strip())
    normalized = normalized.strip("._-")
    return normalized or fallback


def _intrinsics_dict(intrinsics: Mapping[str, Any]) -> dict[str, float] | None:
    output: dict[str, float] = {}
    for key in ("fx", "fy", "cx", "cy"):
        value = _finite_float(intrinsics.get(key))
        if value is None:
            return None
        output[key] = value
    if output["fx"] <= 0.0 or output["fy"] <= 0.0:
        return None
    return output


def camera_signature(
    *,
    device_serial: str | None,
    depth_shape: tuple[int, int],
    intrinsics: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the device/mode identity that a calibration profile is bound to."""
    if len(depth_shape) != 2:
        raise ValueError("depth_shape must contain height and width")
    height, width = (int(depth_shape[0]), int(depth_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("depth_shape must be positive")
    values = _intrinsics_dict(intrinsics)
    if values is None:
        raise ValueError("invalid intrinsics")
    serial = str(device_serial or "").strip()
    if not serial:
        raise ValueError("device serial is required for a persistent calibration")
    return {
        "device_serial": serial,
        "depth_width": width,
        "depth_height": height,
        "intrinsics": values,
    }


def sample_center_board_depth(
    depth_frames: Iterable[np.ndarray],
    *,
    depth_scale_m: float,
    roi_fraction: float = 0.18,
) -> dict[str, Any]:
    """Measure a flat board in a central ROI over several live depth frames."""
    scale = _finite_float(depth_scale_m)
    if scale is None or scale <= 0.0:
        return {"status": "invalid_depth_scale"}
    fraction = _finite_float(roi_fraction)
    if fraction is None or not 0.05 <= fraction <= 0.50:
        return {"status": "invalid_roi_fraction"}

    per_frame: list[dict[str, float]] = []
    frame_shape: tuple[int, int] | None = None
    roi_xyxy: list[int] | None = None
    supplied_count = 0
    for frame in depth_frames:
        supplied_count += 1
        if not isinstance(frame, np.ndarray) or frame.ndim != 2:
            continue
        height, width = frame.shape
        if height < 20 or width < 20:
            continue
        if frame_shape is None:
            frame_shape = (height, width)
            half_width = max(12, int(round(width * fraction / 2.0)))
            half_height = max(12, int(round(height * fraction / 2.0)))
            center_x, center_y = width // 2, height // 2
            roi_xyxy = [
                max(0, center_x - half_width),
                max(0, center_y - half_height),
                min(width, center_x + half_width),
                min(height, center_y + half_height),
            ]
        if frame_shape != (height, width) or roi_xyxy is None:
            continue
        x0, y0, x1, y1 = roi_xyxy
        values = frame[y0:y1, x0:x1].astype(np.float32, copy=False) * scale
        valid = np.isfinite(values) & (values >= 0.20) & (values <= 10.0)
        valid_fraction = float(valid.mean()) if values.size else 0.0
        if int(valid.sum()) < 80 or valid_fraction < MIN_FRAME_VALID_FRACTION:
            continue
        selected = values[valid]
        q1, median, q3 = np.percentile(selected, [25.0, 50.0, 75.0])
        iqr = float(q3 - q1)
        if not math.isfinite(iqr) or iqr > MAX_FRAME_IQR_M:
            continue
        per_frame.append(
            {
                "median_depth_m": float(median),
                "iqr_m": iqr,
                "valid_fraction": valid_fraction,
            }
        )

    if len(per_frame) < 3:
        return {
            "status": "insufficient_stable_board_frames",
            "supplied_frame_count": supplied_count,
            "accepted_frame_count": len(per_frame),
            "guidance": "Keep a large, matte, flat board in the central depth ROI and wait for stable depth.",
        }

    medians = np.asarray([item["median_depth_m"] for item in per_frame], dtype=np.float64)
    center = float(np.median(medians))
    median_abs_deviation = float(np.median(np.abs(medians - center)))
    stable = np.abs(medians - center) <= max(0.004, 3.0 * 1.4826 * median_abs_deviation)
    if int(stable.sum()) < 3:
        return {
            "status": "board_depth_not_stable",
            "supplied_frame_count": supplied_count,
            "accepted_frame_count": len(per_frame),
        }
    selected = medians[stable]
    q1, median, q3 = np.percentile(selected, [25.0, 50.0, 75.0])
    return {
        "status": "ok",
        "method": "central_roi_multiframe_median",
        "measured_depth_m": float(median),
        "frame_count": supplied_count,
        "accepted_frame_count": int(len(per_frame)),
        "inlier_frame_count": int(stable.sum()),
        "median_absolute_deviation_m": median_abs_deviation,
        "inter_frame_iqr_m": float(q3 - q1),
        "median_intra_frame_iqr_m": float(
            np.median([item["iqr_m"] for item in per_frame])
        ),
        "median_valid_fraction": float(
            np.median([item["valid_fraction"] for item in per_frame])
        ),
        "roi_xyxy": roi_xyxy,
        "depth_shape": [int(frame_shape[1]), int(frame_shape[0])],
        "raw_depth_scale_m": scale,
    }


def build_calibration_sample(
    *,
    known_distance_m: float,
    board_measurement: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach a physically measured board distance to a stable depth sample."""
    distance = _finite_float(known_distance_m)
    measured = _finite_float(board_measurement.get("measured_depth_m"))
    if distance is None or not MIN_BOARD_DISTANCE_M <= distance <= MAX_BOARD_DISTANCE_M:
        raise ValueError(
            f"known board distance must be between {MIN_BOARD_DISTANCE_M:.1f} and {MAX_BOARD_DISTANCE_M:.1f} m"
        )
    if board_measurement.get("status") != "ok" or measured is None:
        raise ValueError("board measurement is not stable")
    return {
        "known_distance_m": distance,
        "measured_depth_m": measured,
        "measurement_error_before_fit_m": distance - measured,
        "method": str(board_measurement.get("method") or "central_roi_multiframe_median"),
        "frame_count": int(board_measurement.get("frame_count", 0) or 0),
        "accepted_frame_count": int(
            board_measurement.get("accepted_frame_count", 0) or 0
        ),
        "inlier_frame_count": int(board_measurement.get("inlier_frame_count", 0) or 0),
        "median_absolute_deviation_m": _finite_float(
            board_measurement.get("median_absolute_deviation_m")
        ),
        "inter_frame_iqr_m": _finite_float(
            board_measurement.get("inter_frame_iqr_m")
        ),
        "median_intra_frame_iqr_m": _finite_float(
            board_measurement.get("median_intra_frame_iqr_m")
        ),
        "median_valid_fraction": _finite_float(
            board_measurement.get("median_valid_fraction")
        ),
        "roi_xyxy": list(board_measurement.get("roi_xyxy") or []),
        "captured_at": _utc_now(),
    }


def _weighted_affine_fit(
    measured: np.ndarray,
    known: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float] | None:
    # This is a two-parameter fit. The closed-form normal equations avoid
    # dispatching a tiny least-squares problem through the platform BLAS/MKL
    # stack, which has proved unstable in the desktop capture process.
    x = np.asarray(measured, dtype=np.float64)
    y = np.asarray(known, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or w.shape != x.shape:
        return None
    if len(x) < 2 or not np.isfinite(np.concatenate((x, y, w))).all() or np.any(w <= 0.0):
        return None
    sum_w = float(np.sum(w))
    sum_x = float(np.sum(w * x))
    sum_y = float(np.sum(w * y))
    sum_xx = float(np.sum(w * x * x))
    sum_xy = float(np.sum(w * x * y))
    determinant = sum_xx * sum_w - sum_x * sum_x
    if not np.isfinite(determinant) or abs(determinant) <= 1e-12:
        return None
    scale = (sum_xy * sum_w - sum_x * sum_y) / determinant
    offset = (sum_xx * sum_y - sum_x * sum_xy) / determinant
    if not np.isfinite([scale, offset]).all():
        return None
    return float(scale), float(offset)


def fit_depth_correction(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Robustly fit true_distance = scale * measured_depth + offset."""
    accepted: list[dict[str, Any]] = []
    for index, source in enumerate(samples):
        known = _finite_float(source.get("known_distance_m"))
        measured = _finite_float(source.get("measured_depth_m"))
        iqr = _finite_float(source.get("inter_frame_iqr_m"))
        if (
            known is None
            or measured is None
            or not MIN_BOARD_DISTANCE_M <= known <= MAX_BOARD_DISTANCE_M
            or not 0.20 <= measured <= 10.0
        ):
            continue
        accepted.append(
            {
                "sample_index": index,
                "known_distance_m": known,
                "measured_depth_m": measured,
                "inter_frame_iqr_m": max(float(iqr or 0.003), 0.001),
            }
        )
    if len(accepted) < MIN_SAMPLES:
        return {
            "status": "too_few_samples",
            "sample_count": len(accepted),
            "minimum_sample_count": MIN_SAMPLES,
        }

    measured = np.asarray(
        [item["measured_depth_m"] for item in accepted], dtype=np.float64
    )
    known = np.asarray(
        [item["known_distance_m"] for item in accepted], dtype=np.float64
    )
    if (
        float(np.ptp(measured)) < MIN_DISTANCE_SPAN_M
        or float(np.ptp(known)) < MIN_DISTANCE_SPAN_M
    ):
        return {
            "status": "insufficient_distance_span",
            "sample_count": len(accepted),
            "measured_distance_span_m": float(np.ptp(measured)),
            "known_distance_span_m": float(np.ptp(known)),
            "minimum_distance_span_m": MIN_DISTANCE_SPAN_M,
        }

    weights = 1.0 / np.square(
        np.asarray([item["inter_frame_iqr_m"] for item in accepted], dtype=np.float64)
        + 0.003
    )
    inliers = np.ones(len(accepted), dtype=bool)
    fitted: tuple[float, float] | None = None
    residuals = np.zeros(len(accepted), dtype=np.float64)
    for _ in range(4):
        fitted = _weighted_affine_fit(measured[inliers], known[inliers], weights[inliers])
        if fitted is None:
            return {"status": "fit_failed", "sample_count": len(accepted)}
        scale, offset = fitted
        residuals = known - (scale * measured + offset)
        residual_center = float(np.median(residuals[inliers]))
        residual_mad = float(
            np.median(np.abs(residuals[inliers] - residual_center))
        )
        threshold = max(0.008, 3.0 * 1.4826 * residual_mad)
        next_inliers = np.abs(residuals - residual_center) <= threshold
        if int(next_inliers.sum()) < MIN_SAMPLES:
            break
        if np.array_equal(next_inliers, inliers):
            inliers = next_inliers
            break
        inliers = next_inliers

    if fitted is None or int(inliers.sum()) < MIN_SAMPLES:
        return {
            "status": "too_few_fit_inliers",
            "sample_count": len(accepted),
            "inlier_count": int(inliers.sum()),
        }
    fitted = _weighted_affine_fit(measured[inliers], known[inliers], weights[inliers])
    if fitted is None:
        return {"status": "fit_failed", "sample_count": len(accepted)}
    scale, offset = fitted
    residuals = known - (scale * measured + offset)
    inlier_residuals = residuals[inliers]
    rmse_m = float(np.sqrt(np.mean(np.square(inlier_residuals))))
    max_abs_residual_m = float(np.max(np.abs(inlier_residuals)))
    if not 0.85 <= scale <= 1.15 or abs(offset) > 0.15:
        return {
            "status": "implausible_depth_correction",
            "sample_count": len(accepted),
            "inlier_count": int(inliers.sum()),
            "scale": scale,
            "offset_m": offset,
        }
    if rmse_m > 0.020 or max_abs_residual_m > 0.040:
        return {
            "status": "fit_residual_too_large",
            "sample_count": len(accepted),
            "inlier_count": int(inliers.sum()),
            "fit_rmse_m": rmse_m,
            "max_abs_residual_m": max_abs_residual_m,
        }

    sample_diagnostics = []
    for item, residual, inlier in zip(accepted, residuals, inliers):
        sample_diagnostics.append(
            {
                "sample_index": item["sample_index"],
                "known_distance_m": item["known_distance_m"],
                "measured_depth_m": item["measured_depth_m"],
                "residual_m": float(residual),
                "fit_inlier": bool(inlier),
            }
        )
    return {
        "status": "ok",
        "model": "z_affine",
        "scale": scale,
        "offset_m": offset,
        "fit_rmse_m": rmse_m,
        "max_abs_residual_m": max_abs_residual_m,
        "sample_count": len(accepted),
        "inlier_count": int(inliers.sum()),
        "measured_distance_span_m": float(np.ptp(measured[inliers])),
        "known_distance_span_m": float(np.ptp(known[inliers])),
        "sample_diagnostics": sample_diagnostics,
    }


def build_profile(
    *,
    station_id: str,
    device_serial: str,
    depth_shape: tuple[int, int],
    intrinsics: Mapping[str, Any],
    raw_depth_scale_m: float,
    samples: Iterable[Mapping[str, Any]],
    operator: str | None = None,
) -> dict[str, Any]:
    """Create an auditable calibration profile from board-distance samples."""
    signature = camera_signature(
        device_serial=device_serial,
        depth_shape=depth_shape,
        intrinsics=intrinsics,
    )
    scale = _finite_float(raw_depth_scale_m)
    if scale is None or scale <= 0.0:
        raise ValueError("raw depth scale must be positive")
    normalized_samples = [dict(item) for item in samples]
    fit = fit_depth_correction(normalized_samples)
    return {
        "schema": SCHEMA,
        "profile_id": f"height-cal-{uuid.uuid4().hex[:12]}",
        "status": "valid" if fit.get("status") == "ok" else "rejected",
        "created_at": _utc_now(),
        "station_id": _safe_name(station_id, "fixed-station"),
        "operator": str(operator or "").strip() or None,
        "camera": signature,
        "raw_depth_scale_m": scale,
        "depth_correction": fit if fit.get("status") == "ok" else None,
        "fit_failure": None if fit.get("status") == "ok" else fit,
        "samples": normalized_samples,
        "manual_person_height_used": False,
        "calibration_target": "rigid_planar_board_at_known_camera_distance",
        "clinical_use": False,
        "requires_human_review": True,
    }


def profile_matches(
    profile: Mapping[str, Any],
    *,
    device_serial: str | None,
    depth_shape: tuple[int, int],
    intrinsics: Mapping[str, Any],
) -> tuple[bool, str]:
    """Return whether a profile can be applied to the current Gemini stream."""
    if profile.get("schema") != SCHEMA or profile.get("status") != "valid":
        return False, "profile_not_valid"
    correction = profile.get("depth_correction")
    if not isinstance(correction, Mapping) or correction.get("status") != "ok":
        return False, "profile_has_no_valid_correction"
    saved_camera = profile.get("camera")
    if not isinstance(saved_camera, Mapping):
        return False, "profile_camera_missing"
    try:
        current = camera_signature(
            device_serial=device_serial,
            depth_shape=depth_shape,
            intrinsics=intrinsics,
        )
    except ValueError as error:
        return False, str(error)
    if str(saved_camera.get("device_serial") or "") != current["device_serial"]:
        return False, "camera_serial_mismatch"
    if (
        int(saved_camera.get("depth_width", -1)) != current["depth_width"]
        or int(saved_camera.get("depth_height", -1)) != current["depth_height"]
    ):
        return False, "depth_resolution_mismatch"
    saved_intrinsics = _intrinsics_dict(saved_camera.get("intrinsics") or {})
    if saved_intrinsics is None:
        return False, "profile_intrinsics_invalid"
    for key in ("fx", "fy"):
        relative_error = abs(saved_intrinsics[key] - current["intrinsics"][key]) / max(
            saved_intrinsics[key], 1e-6
        )
        if relative_error > INTRINSICS_RELATIVE_TOLERANCE:
            return False, f"{key}_mismatch"
    for key in ("cx", "cy"):
        if abs(saved_intrinsics[key] - current["intrinsics"][key]) > PRINCIPAL_POINT_TOLERANCE_PX:
            return False, f"{key}_mismatch"
    return True, "ok"


def correction_request(
    profile: Mapping[str, Any],
    *,
    device_serial: str | None,
    depth_shape: tuple[int, int],
    intrinsics: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build the small metadata payload accepted by the remote height service."""
    matched, _reason = profile_matches(
        profile,
        device_serial=device_serial,
        depth_shape=depth_shape,
        intrinsics=intrinsics,
    )
    if not matched:
        return None
    correction = profile.get("depth_correction")
    assert isinstance(correction, Mapping)
    return {
        "schema": REQUEST_SCHEMA,
        "profile_id": str(profile.get("profile_id") or ""),
        "station_id": str(profile.get("station_id") or ""),
        "created_at": str(profile.get("created_at") or ""),
        "camera_serial": str(device_serial or ""),
        "model": str(correction.get("model") or "z_affine"),
        "scale": float(correction["scale"]),
        "offset_m": float(correction["offset_m"]),
        "fit_rmse_m": float(correction.get("fit_rmse_m", 0.0) or 0.0),
        "manual_person_height_used": False,
        "clinical_use": False,
        "requires_human_review": True,
    }


def save_profile(root: Path, profile: Mapping[str, Any]) -> Path:
    """Save a new immutable profile below the capture root without deleting prior ones."""
    if profile.get("schema") != SCHEMA:
        raise ValueError("unexpected calibration schema")
    camera = profile.get("camera")
    if not isinstance(camera, Mapping):
        raise ValueError("profile has no camera signature")
    serial = _safe_name(camera.get("device_serial"), "unknown-camera")
    profile_id = _safe_name(profile.get("profile_id"), "height-calibration")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target_dir = Path(root) / "height_calibration" / serial
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{timestamp}_{profile_id}.json"
    payload = json.dumps(profile, ensure_ascii=False, indent=2) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".height-calibration-", suffix=".json", dir=target_dir
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return target


def load_profile(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("calibration profile must be a JSON object")
    return value


def load_latest_matching_profile(
    root: Path,
    *,
    device_serial: str | None,
    depth_shape: tuple[int, int],
    intrinsics: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, Path | None]:
    """Find the newest valid profile for exactly this Gemini stream mode."""
    serial = _safe_name(device_serial, "unknown-camera")
    directory = Path(root) / "height_calibration" / serial
    if not directory.is_dir():
        return None, None
    for path in sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            profile = load_profile(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        matched, _reason = profile_matches(
            profile,
            device_serial=device_serial,
            depth_shape=depth_shape,
            intrinsics=intrinsics,
        )
        if matched:
            return profile, path
    return None, None


def profile_summary(profile: Mapping[str, Any] | None) -> str:
    if not isinstance(profile, Mapping):
        return "未加载固定机位深度标定"
    correction = profile.get("depth_correction")
    if not isinstance(correction, Mapping) or correction.get("status") != "ok":
        return "深度标定档案无效"
    return (
        f"已加载标定 {profile.get('station_id', 'fixed-station')} / "
        f"比例 {float(correction.get('scale', 1.0)):.5f} / "
        f"偏置 {float(correction.get('offset_m', 0.0)) * 1000.0:+.1f} mm / "
        f"拟合 RMSE {float(correction.get('fit_rmse_m', 0.0)) * 1000.0:.1f} mm"
    )
