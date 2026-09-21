from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


CAPTURE_SCHEMA = "gemini2l_capture.v1"


def aligned_point_cloud(
    depth: np.ndarray,
    intrinsics: dict[str, float],
    depth_scale_m: float,
    rgb: np.ndarray | None = None,
    stride: int = 4,
    min_depth_m: float = 0.2,
    max_depth_m: float = 10.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Back-project an aligned depth image and optional RGB into meters."""
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("depth must have shape [H, W]")
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.shape != (*depth.shape, 3):
            raise ValueError("rgb must have shape [H, W, 3] matching depth")
    scale = float(depth_scale_m)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("depth_scale_m must be positive")
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("fx and fy must be positive")
    step = max(int(stride), 1)
    sampled_depth = depth[::step, ::step].astype(np.float32) * scale
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= float(min_depth_m))
        & (sampled_depth <= float(max_depth_m))
    )
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = None if rgb is None else np.empty((0, 3), dtype=np.uint8)
        return empty_points, empty_colors
    z = sampled_depth[rows, cols]
    x_pixels = cols.astype(np.float32) * step
    y_pixels = rows.astype(np.float32) * step
    points = np.column_stack(
        (
            (x_pixels - cx) * z / fx,
            (y_pixels - cy) * z / fy,
            z,
        )
    ).astype(np.float32, copy=False)
    colors = None
    if rgb is not None:
        colors = rgb[rows * step, cols * step].astype(np.uint8, copy=False)
    return points, colors


def build_capture_record(
    *,
    record_id: str,
    subject_id: str,
    session_id: str,
    device_id: str,
    rgb: str,
    depth: str,
    ir: str | None,
    pointcloud: str | None,
    intrinsics: dict[str, float],
    depth_scale_m: float,
    timestamps_us: dict[str, int | None],
    imu: dict[str, Any] | None,
    frame_timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "schema": CAPTURE_SCHEMA,
        "id": str(record_id),
        "subject_id": str(subject_id),
        "session_id": str(session_id),
        "device_id": str(device_id),
        "rgb": str(rgb),
        "depth": str(depth),
        "ir": None if ir is None else str(ir),
        "pointcloud": None if pointcloud is None else str(pointcloud),
        "intrinsics": {
            "fx": float(intrinsics["fx"]),
            "fy": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]),
            "cy": float(intrinsics["cy"]),
            "depth_scale": float(depth_scale_m),
        },
        "depth_scale": float(depth_scale_m),
        "timestamps_us": timestamps_us,
        "imu": imu,
        "annotation_status": "unlabeled",
    }
    if frame_timing is not None:
        record["frame_timing"] = frame_timing
    return record


def capture_record_errors(record: dict[str, Any], root: str | Path) -> list[str]:
    errors: list[str] = []
    if record.get("schema") != CAPTURE_SCHEMA:
        errors.append(f"unsupported schema: {record.get('schema')!r}")
    for key in ("id", "subject_id", "session_id", "device_id", "rgb", "depth"):
        if not record.get(key):
            errors.append(f"missing {key}")
    root_path = Path(root)
    for key in ("rgb", "depth", "ir", "pointcloud"):
        value = record.get(key)
        if not value:
            continue
        path = Path(value)
        path = path if path.is_absolute() else root_path / path
        if not path.is_file():
            errors.append(f"missing file for {key}: {path}")
    intrinsics = record.get("intrinsics") or {}
    for key in ("fx", "fy", "cx", "cy"):
        try:
            value = float(intrinsics[key])
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid intrinsics.{key}")
            continue
        if not np.isfinite(value) or (key in ("fx", "fy") and value <= 0.0):
            errors.append(f"invalid intrinsics.{key}")
    try:
        depth_scale = float(record.get("depth_scale", 0.0))
    except (TypeError, ValueError):
        depth_scale = 0.0
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        errors.append("invalid depth_scale")
    timestamps = record.get("timestamps_us") or {}
    try:
        rgb_timestamp = int(timestamps["rgb"])
        depth_timestamp = int(timestamps["depth"])
    except (KeyError, TypeError, ValueError):
        errors.append("invalid RGB/depth timestamps")
    else:
        if rgb_timestamp <= 0 or depth_timestamp <= 0:
            errors.append("RGB/depth timestamps must be positive")
        if abs(rgb_timestamp - depth_timestamp) > 100_000:
            errors.append("RGB/depth timestamp delta exceeds 100 ms")
    frame_timing = record.get("frame_timing")
    if frame_timing is not None:
        try:
            capture_monotonic_ns = int(frame_timing["capture_monotonic_ns"])
        except (KeyError, TypeError, ValueError):
            errors.append("invalid frame_timing.capture_monotonic_ns")
        else:
            if capture_monotonic_ns <= 0:
                errors.append("invalid frame_timing.capture_monotonic_ns")
        streams = frame_timing.get("streams") or {}
        for stream_name in ("rgb", "depth"):
            stream = streams.get(stream_name) or {}
            try:
                int(stream["frame_index"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"invalid frame_timing.streams.{stream_name}.frame_index")
    return errors
