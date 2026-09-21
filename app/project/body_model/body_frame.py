from __future__ import annotations

from typing import Iterable

import numpy as np

from .depth import CameraIntrinsics, center_3d
from .regions import _as_keypoints


def _normalize(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < eps:
        return None
    return vector / norm


def keypoints_to_camera(
    keypoints: Iterable,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    image_size: tuple[int, int] | None = None,
    min_depth_m: float = 0.2,
    max_depth_m: float = 10.0,
    reference_depth_m: float | None = None,
    max_depth_delta_m: float = 0.75,
) -> np.ndarray:
    """Project COCO-17 keypoints to camera coordinates, preserving holes."""
    points = _as_keypoints(keypoints)
    output = np.full((points.shape[0], 3), np.nan, dtype=np.float32)
    for index, point in enumerate(points):
        if point[2] <= 0 or not np.all(np.isfinite(point[:2])):
            continue
        projected = center_3d(
            point[:2],
            depth,
            intrinsics,
            image_size=image_size,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            reference_depth_m=reference_depth_m,
            max_depth_delta_m=max_depth_delta_m,
        )
        if projected is not None:
            output[index] = np.asarray(projected, dtype=np.float32)
    return output


def estimate_body_frame(
    keypoints: Iterable,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    image_size: tuple[int, int] | None = None,
    min_depth_m: float = 0.2,
    max_depth_m: float = 10.0,
    reference_depth_m: float | None = None,
    max_depth_delta_m: float = 0.75,
) -> dict | None:
    """Estimate a patient-centered orthonormal frame from RGB-D pose.

    The returned ``camera_from_body`` matrix maps coordinates expressed in the
    body frame to the camera frame. The origin is the 3D hip midpoint, the
    y-axis points from hips toward shoulders, and the x-axis follows the
    shoulder line. It is intended for scale and relation features, not for
    clinical diagnosis.
    """
    camera_points = keypoints_to_camera(
        keypoints,
        depth,
        intrinsics,
        image_size=image_size,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        reference_depth_m=reference_depth_m,
        max_depth_delta_m=max_depth_delta_m,
    )
    required = [5, 6, 11, 12]
    if not np.all(np.isfinite(camera_points[required])):
        return None
    left_shoulder, right_shoulder = camera_points[5], camera_points[6]
    left_hip, right_hip = camera_points[11], camera_points[12]
    shoulder_mid = (left_shoulder + right_shoulder) * 0.5
    hip_mid = (left_hip + right_hip) * 0.5
    x_axis = _normalize(right_shoulder - left_shoulder)
    vertical = _normalize(shoulder_mid - hip_mid)
    if x_axis is None or vertical is None:
        return None
    # Gram-Schmidt makes the axes stable when the patient is slightly rotated.
    vertical = _normalize(vertical - x_axis * np.dot(vertical, x_axis))
    if vertical is None:
        return None
    z_axis = _normalize(np.cross(x_axis, vertical))
    if z_axis is None:
        return None
    vertical = _normalize(np.cross(z_axis, x_axis))
    if vertical is None:
        return None
    rotation = np.column_stack([x_axis, vertical, z_axis]).astype(np.float32)
    camera_from_body = np.eye(4, dtype=np.float32)
    camera_from_body[:3, :3] = rotation
    camera_from_body[:3, 3] = hip_mid
    # Avoid a BLAS-backed matrix multiply here. On Windows, Anaconda MKL and
    # PyTorch can load different OpenMP runtimes; this fixed-size transform is
    # cheap to express element-wise and avoids a process-level abort in mixed
    # RGB-D inference environments.
    body_points = np.sum(
        (camera_points - hip_mid)[:, :, None] * rotation[None, :, :],
        axis=1,
    )
    body_points[~np.all(np.isfinite(camera_points), axis=1)] = np.nan
    return {
        "origin_camera": hip_mid.tolist(),
        "axes_camera": rotation.tolist(),
        "camera_from_body": camera_from_body.tolist(),
        "keypoints_3d_camera": camera_points.tolist(),
        "keypoints_3d_body": body_points.tolist(),
        "shoulder_width_m": float(np.linalg.norm(right_shoulder - left_shoulder)),
        "torso_length_m": float(np.linalg.norm(shoulder_mid - hip_mid)),
    }
