from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("fx and fy must be positive")
        if self.depth_scale <= 0:
            raise ValueError("depth_scale must be positive")

    @classmethod
    def from_mapping(cls, value: dict) -> "CameraIntrinsics":
        return cls(
            fx=float(value["fx"]),
            fy=float(value["fy"]),
            cx=float(value["cx"]),
            cy=float(value["cy"]),
            depth_scale=float(value.get("depth_scale", 1.0)),
        )


def _finite_depth(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return values[np.isfinite(values) & (values > 0)]


def depth_to_meters(depth: np.ndarray, depth_scale: float = 1.0) -> np.ndarray:
    """Convert a raw depth image to float32 meters without inventing values."""
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("depth must be a two-dimensional array")
    values = values * float(depth_scale)
    values[~np.isfinite(values)] = 0.0
    values[values < 0.0] = 0.0
    return values


def sample_depth(
    depth: np.ndarray,
    x: float,
    y: float,
    radius: int = 3,
    depth_scale: float = 1.0,
    min_depth_m: float | None = None,
    max_depth_m: float | None = None,
    reference_depth_m: float | None = None,
    max_depth_delta_m: float | None = None,
) -> float | None:
    """Return a robust metric depth sample around a pixel."""
    if depth.ndim != 2:
        raise ValueError("depth must be a two-dimensional array")
    ix, iy = int(round(x)), int(round(y))
    y0, y1 = max(0, iy - radius), min(depth.shape[0], iy + radius + 1)
    x0, x1 = max(0, ix - radius), min(depth.shape[1], ix + radius + 1)
    values = _finite_depth(depth[y0:y1, x0:x1]) * float(depth_scale)
    if min_depth_m is not None:
        values = values[values >= float(min_depth_m)]
    if max_depth_m is not None:
        values = values[values <= float(max_depth_m)]
    if reference_depth_m is not None and max_depth_delta_m is not None:
        reference = float(reference_depth_m)
        delta = max(float(max_depth_delta_m), 0.0)
        values = values[np.abs(values - reference) <= delta]
    if values.size == 0:
        return None
    # A median is stable for small holes and isolated invalid pixels.
    return float(np.median(values))


def project_pixel_to_camera(
    x: float,
    y: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> list[float]:
    """Project one RGB-D pixel into the camera coordinate system."""
    z = float(depth_m)
    return [
        (float(x) - intrinsics.cx) * z / intrinsics.fx,
        (float(y) - intrinsics.cy) * z / intrinsics.fy,
        z,
    ]


def center_3d(
    center_2d: Iterable[float],
    depth: np.ndarray | None,
    intrinsics: CameraIntrinsics | None,
    image_size: tuple[int, int] | None = None,
    min_depth_m: float | None = None,
    max_depth_m: float | None = None,
    reference_depth_m: float | None = None,
    max_depth_delta_m: float = 0.75,
) -> list[float] | None:
    if depth is None or intrinsics is None:
        return None
    x, y = [float(value) for value in center_2d]
    if image_size is not None:
        image_width, image_height = [float(value) for value in image_size]
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_size must contain positive width and height")
        x *= depth.shape[1] / image_width
        y *= depth.shape[0] / image_height
    value = sample_depth(
        depth,
        x,
        y,
        depth_scale=intrinsics.depth_scale,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        reference_depth_m=reference_depth_m,
        max_depth_delta_m=max_depth_delta_m,
    )
    if value is None and reference_depth_m is not None:
        # Retry a wider neighborhood before using the tracked person's robust
        # median depth. This prevents a small D2C hole from projecting a body
        # part onto distant background while retaining local surface depth
        # whenever nearby measurements exist.
        value = sample_depth(
            depth,
            x,
            y,
            radius=15,
            depth_scale=intrinsics.depth_scale,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            reference_depth_m=reference_depth_m,
            max_depth_delta_m=max_depth_delta_m,
        )
        if value is None:
            reference = float(reference_depth_m)
            lower_ok = min_depth_m is None or reference >= float(min_depth_m)
            upper_ok = max_depth_m is None or reference <= float(max_depth_m)
            if np.isfinite(reference) and lower_ok and upper_ok:
                value = reference
    if value is None:
        return None
    return project_pixel_to_camera(x, y, value, intrinsics)


def depth_to_points(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: np.ndarray | None = None,
    stride: int = 1,
    min_depth_m: float = 0.2,
    max_depth_m: float = 10.0,
) -> np.ndarray:
    """Back-project valid depth pixels to an ``[N, 3]`` camera point cloud."""
    meters = depth_to_meters(depth, intrinsics.depth_scale)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != meters.shape:
            raise ValueError("mask must have the same shape as depth")
    stride = max(int(stride), 1)
    sampled = meters[::stride, ::stride]
    valid = (sampled >= float(min_depth_m)) & (sampled <= float(max_depth_m))
    if mask is not None:
        valid &= mask[::stride, ::stride]
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    xs = cols.astype(np.float32) * stride
    ys = rows.astype(np.float32) * stride
    zs = sampled[rows, cols].astype(np.float32)
    points = np.column_stack(
        ((xs - intrinsics.cx) * zs / intrinsics.fx,
         (ys - intrinsics.cy) * zs / intrinsics.fy,
         zs)
    )
    return points.astype(np.float32, copy=False)


def color_to_depth_pixel(
    xy: Iterable[float],
    depth_shape: tuple[int, int],
    color_size: tuple[int, int],
) -> tuple[float, float]:
    """Map an aligned RGB coordinate to the depth image coordinate system."""
    x, y = [float(value) for value in xy]
    color_width, color_height = [float(value) for value in color_size]
    depth_height, depth_width = depth_shape
    if color_width <= 0 or color_height <= 0:
        raise ValueError("color_size must contain positive width and height")
    return (x * depth_width / color_width, y * depth_height / color_height)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to camera-space points."""
    points = np.asarray(points, dtype=np.float32)
    transform = np.asarray(transform, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape [4, 4]")
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1
    )
    return (homogeneous @ transform.T)[:, :3].astype(np.float32, copy=False)
