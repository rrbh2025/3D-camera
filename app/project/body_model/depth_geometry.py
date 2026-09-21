from __future__ import annotations

from typing import Iterable

import numpy as np

from .depth import CameraIntrinsics, depth_to_meters


def _scaled_bbox(
    bbox: Iterable[float],
    depth_shape: tuple[int, int],
    reference_image_size: tuple[int, int] | None,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    depth_height, depth_width = depth_shape
    if reference_image_size is not None:
        ref_width, ref_height = reference_image_size
        if ref_width <= 0 or ref_height <= 0:
            raise ValueError("reference image size must be positive")
        scale_x = depth_width / float(ref_width)
        scale_y = depth_height / float(ref_height)
        x0, x1 = x0 * scale_x, x1 * scale_x
        y0, y1 = y0 * scale_y, y1 * scale_y
    return (
        max(0, min(depth_width, int(np.floor(x0)))),
        max(0, min(depth_height, int(np.floor(y0)))),
        max(0, min(depth_width, int(np.ceil(x1)) + 1)),
        max(0, min(depth_height, int(np.ceil(y1)) + 1)),
    )


def points_in_bbox(
    depth: np.ndarray,
    bbox: Iterable[float],
    intrinsics: CameraIntrinsics,
    reference_image_size: tuple[int, int] | None = None,
    stride: int = 3,
    min_depth_m: float = 0.2,
    max_depth_m: float = 10.0,
    reference_depth_m: float | None = None,
    max_depth_delta_m: float | None = None,
    depth_meters: np.ndarray | None = None,
) -> np.ndarray:
    """Return robustly sampled camera points inside an aligned image box."""
    if depth_meters is None:
        meters = depth_to_meters(depth, intrinsics.depth_scale)
    else:
        meters = np.asarray(depth_meters, dtype=np.float32)
        if meters.ndim != 2 or meters.shape != np.asarray(depth).shape:
            raise ValueError("depth_meters must match the depth image shape")
    x0, y0, x1, y1 = _scaled_bbox(bbox, meters.shape, reference_image_size)
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 3), dtype=np.float32)
    crop = meters[y0:y1: max(int(stride), 1), x0:x1: max(int(stride), 1)]
    valid = (crop >= float(min_depth_m)) & (crop <= float(max_depth_m))
    if reference_depth_m is not None and max_depth_delta_m is not None:
        reference = float(reference_depth_m)
        delta = float(max_depth_delta_m)
        if np.isfinite(reference) and np.isfinite(delta) and delta > 0.0:
            valid &= np.abs(crop - reference) <= delta
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    step = max(int(stride), 1)
    xs = (x0 + cols * step).astype(np.float32)
    ys = (y0 + rows * step).astype(np.float32)
    zs = crop[rows, cols].astype(np.float32)
    return np.column_stack(
        ((xs - intrinsics.cx) * zs / intrinsics.fx,
         (ys - intrinsics.cy) * zs / intrinsics.fy,
         zs)
    ).astype(np.float32, copy=False)


def _trimmed_values(values: np.ndarray, trim_fraction: float) -> np.ndarray:
    if values.size < 4 or trim_fraction <= 0:
        return values
    values = np.sort(values, axis=0)
    count = int(values.shape[0] * min(max(trim_fraction, 0.0), 0.45))
    if count == 0 or values.shape[0] <= 2 * count:
        return values
    return values[count:-count]


def summarize_points(points: np.ndarray, trim_fraction: float = 0.05) -> dict | None:
    """Summarize a region point cloud using trimmed robust statistics."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.size == 0:
        return None
    trimmed = _trimmed_values(points, trim_fraction)
    center = np.median(trimmed, axis=0)
    lower = np.percentile(trimmed, 5, axis=0)
    upper = np.percentile(trimmed, 95, axis=0)
    mad = np.median(np.abs(trimmed - center), axis=0)
    return {
        "point_count": int(points.shape[0]),
        "center_3d": [float(value) for value in center],
        "lower_3d": [float(value) for value in lower],
        "upper_3d": [float(value) for value in upper],
        "extent_3d": [float(value) for value in upper - lower],
        "mad_3d": [float(value) for value in mad],
        "depth_median": float(center[2]),
        "depth_iqr": float(np.percentile(trimmed[:, 2], 75) - np.percentile(trimmed[:, 2], 25)),
    }


def region_geometry(
    depth: np.ndarray,
    bbox: Iterable[float],
    intrinsics: CameraIntrinsics,
    reference_image_size: tuple[int, int] | None = None,
    stride: int = 3,
    min_depth_m: float = 0.2,
    max_depth_m: float = 10.0,
    reference_depth_m: float | None = None,
    max_depth_delta_m: float | None = None,
    depth_meters: np.ndarray | None = None,
) -> dict | None:
    points = points_in_bbox(
        depth,
        bbox,
        intrinsics,
        reference_image_size=reference_image_size,
        stride=stride,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        reference_depth_m=reference_depth_m,
        max_depth_delta_m=max_depth_delta_m,
        depth_meters=depth_meters,
    )
    return summarize_points(points)
