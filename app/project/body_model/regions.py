from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .depth import CameraIntrinsics, center_3d, depth_to_meters
from .depth_geometry import region_geometry


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

REGION_NAMES = [
    "head",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "chest",
    "abdomen",
    "pelvis",
    "spine",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
]

REGION_KEYPOINT_GROUPS = {
    "head": [0, 1, 2, 3, 4],
    "neck": [0, 5, 6],
    "left_shoulder": [5, 7, 11],
    "right_shoulder": [6, 8, 12],
    "chest": [5, 6, 11, 12],
    "abdomen": [5, 6, 11, 12],
    "pelvis": [11, 12],
    "spine": [5, 6, 11, 12],
    "left_arm": [5, 7, 9],
    "right_arm": [6, 8, 10],
    "left_leg": [11, 13, 15],
    "right_leg": [12, 14, 16],
}


def body_region_centers_3d(
    keypoints_3d: Iterable,
    scores: Iterable | None = None,
    threshold: float = 0.25,
) -> dict[str, np.ndarray]:
    """Derive metric 3D region centers from COCO-17 camera-space joints."""
    points = np.asarray(keypoints_3d, dtype=np.float32)
    if points.shape != (len(COCO_KEYPOINT_NAMES), 3):
        raise ValueError("keypoints_3d must have shape [17, 3]")
    if scores is None:
        confidence = np.ones((len(COCO_KEYPOINT_NAMES),), dtype=np.float32)
    else:
        confidence = np.asarray(scores, dtype=np.float32).reshape(
            len(COCO_KEYPOINT_NAMES)
        )

    def mean(indices: list[int]) -> np.ndarray | None:
        selected = points[indices]
        selected_scores = confidence[indices]
        valid = np.all(np.isfinite(selected), axis=1) & (
            selected_scores >= float(threshold)
        )
        if not np.any(valid):
            return None
        weights = np.clip(selected_scores[valid], 1e-4, 1.0)
        return np.average(selected[valid], axis=0, weights=weights).astype(
            np.float32
        )

    def interpolate(
        first: np.ndarray | None,
        second: np.ndarray | None,
        weight: float,
    ) -> np.ndarray | None:
        if first is None or second is None:
            return None
        return (first * (1.0 - weight) + second * weight).astype(np.float32)

    head = mean([0, 1, 2, 3, 4])
    shoulder_mid = mean([5, 6])
    hip_mid = mean([11, 12])
    centers = {
        "head": head,
        "neck": interpolate(head, shoulder_mid, 0.5),
        "left_shoulder": mean([5]),
        "right_shoulder": mean([6]),
        "chest": interpolate(shoulder_mid, hip_mid, 0.25),
        "abdomen": interpolate(shoulder_mid, hip_mid, 0.58),
        "pelvis": hip_mid,
        "spine": interpolate(shoulder_mid, hip_mid, 0.5),
        "left_arm": mean([5, 7, 9]),
        "right_arm": mean([6, 8, 10]),
        "left_leg": mean([11, 13, 15]),
        "right_leg": mean([12, 14, 16]),
    }
    return {
        name: (
            np.full((3,), np.nan, dtype=np.float32)
            if centers[name] is None
            else centers[name]
        )
        for name in REGION_NAMES
    }


@dataclass
class RegionPrediction:
    name: str
    center_2d: list[float] | None
    bbox_2d: list[float] | None
    confidence: float
    visible: bool
    keypoints: list[str]
    center_3d: list[float] | None = None
    geometry_3d: dict | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "center_2d": self.center_2d,
            "bbox_2d": self.bbox_2d,
            "center_3d": self.center_3d,
            "geometry_3d": self.geometry_3d,
            "confidence": round(float(self.confidence), 5),
            "visible": bool(self.visible),
            "keypoints": self.keypoints,
        }


def _as_keypoints(value: Iterable) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        if array.size % 3 == 0:
            array = array.reshape(-1, 3)
        elif array.size % 2 == 0:
            array = array.reshape(-1, 2)
        else:
            raise ValueError("flat keypoints must contain 2 or 3 values per point")
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError("keypoints must have shape [N, 2] or [N, 3]")
    if array.shape[1] == 2:
        array = np.concatenate([array, np.ones((array.shape[0], 1), dtype=np.float32)], axis=1)
    if array.shape[0] < len(COCO_KEYPOINT_NAMES):
        raise ValueError("the body baseline requires 17 COCO keypoints")
    return array[: len(COCO_KEYPOINT_NAMES)]


def _valid(points: np.ndarray, threshold: float) -> np.ndarray:
    return points[np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]) & (points[:, 2] >= threshold)]


def _apply_depth_mask(
    depth: np.ndarray | None,
    depth_mask: np.ndarray | None,
) -> np.ndarray | None:
    if depth is None or depth_mask is None:
        return depth
    values = np.asarray(depth)
    mask = np.asarray(depth_mask, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("depth_mask must have the same shape as depth")
    masked = values.copy()
    masked[~mask] = 0
    return masked


def _mean(points: np.ndarray, indices: list[int], threshold: float) -> np.ndarray | None:
    selected = points[indices]
    selected = _valid(selected, threshold)
    if selected.size == 0:
        return None
    return selected[:, :2].mean(axis=0)


def _confidence(points: np.ndarray, indices: list[int], threshold: float) -> float:
    selected = points[indices]
    selected = selected[np.isfinite(selected[:, 2])]
    if selected.size == 0:
        return 0.0
    visible = selected[:, 2] >= threshold
    if not np.any(visible):
        return 0.0
    return float(np.clip(selected[visible, 2].mean(), 0.0, 1.0))


def _bbox(points: np.ndarray, indices: list[int], margin: float, threshold: float) -> list[float] | None:
    selected = _valid(points[indices], threshold)
    if selected.size == 0:
        return None
    xy = selected[:, :2]
    x0, y0 = xy.min(axis=0) - margin
    x1, y1 = xy.max(axis=0) + margin
    return [float(x0), float(y0), float(x1), float(y1)]


def _box_from_center(center: np.ndarray | None, width: float, height: float) -> list[float] | None:
    if center is None:
        return None
    half_w, half_h = max(width / 2.0, 1.0), max(height / 2.0, 1.0)
    return [
        float(center[0] - half_w),
        float(center[1] - half_h),
        float(center[0] + half_w),
        float(center[1] + half_h),
    ]


def keypoint_region_priors(
    keypoints: Iterable,
    keypoint_threshold: float = 0.10,
) -> dict[str, np.ndarray]:
    """Build normalized region priors from one COCO-17 pose.

    Returns ``center`` and ``size`` in the same coordinate system as the input
    keypoints, plus a confidence-weighted validity flag. The learned RGB-D
    head uses this prior to sample local image features instead of relying on
    global pooling alone.
    """
    points = _as_keypoints(keypoints)
    threshold = float(keypoint_threshold)
    centers = np.zeros((len(REGION_NAMES), 2), dtype=np.float32)
    sizes = np.zeros((len(REGION_NAMES), 2), dtype=np.float32)
    valid = np.zeros((len(REGION_NAMES),), dtype=np.float32)
    scores = np.zeros((len(REGION_NAMES),), dtype=np.float32)
    for region_index, name in enumerate(REGION_NAMES):
        indices = REGION_KEYPOINT_GROUPS[name]
        selected = points[indices]
        selected = selected[
            np.isfinite(selected[:, 0])
            & np.isfinite(selected[:, 1])
            & (selected[:, 2] >= threshold)
        ]
        if selected.size == 0:
            continue
        xy = selected[:, :2]
        lower = xy.min(axis=0)
        upper = xy.max(axis=0)
        centers[region_index] = (lower + upper) * 0.5
        sizes[region_index] = np.maximum(upper - lower, 1e-3)
        valid[region_index] = 1.0
        scores[region_index] = float(np.clip(selected[:, 2].mean(), 0.0, 1.0))
    return {
        "center": centers,
        "size": sizes,
        "valid": valid,
        "score": scores,
    }


class BodyRegionEstimator:
    """Convert COCO-17 pose output into coarse anatomical regions.

    This is a deterministic geometry baseline. It is intentionally kept
    separate from the pose network so a learned RGB-D region head can replace
    it without changing the camera and MMPose adapters.
    """

    def __init__(self, keypoint_threshold: float = 0.25):
        self.keypoint_threshold = float(keypoint_threshold)

    def estimate(
        self,
        keypoints: Iterable,
        depth: np.ndarray | None = None,
        intrinsics: CameraIntrinsics | None = None,
        image_size: tuple[int, int] | None = None,
        include_geometry: bool = False,
        geometry_stride: int = 3,
        min_depth_m: float = 0.2,
        max_depth_m: float = 10.0,
        reference_depth_m: float | None = None,
        max_depth_delta_m: float = 0.75,
        depth_mask: np.ndarray | None = None,
    ) -> dict:
        points = _as_keypoints(keypoints)
        effective_depth = _apply_depth_mask(depth, depth_mask)
        threshold = self.keypoint_threshold
        valid_points = _valid(points, threshold)
        if valid_points.size == 0:
            raise ValueError("no visible keypoints meet the confidence threshold")

        y_min, y_max = valid_points[:, 1].min(), valid_points[:, 1].max()
        body_height = max(float(y_max - y_min), 10.0)
        shoulder = _mean(points, [5, 6], threshold)
        hips = _mean(points, [11, 12], threshold)
        head = _mean(points, [0, 1, 2, 3, 4], threshold)
        shoulder_width = self._distance(points, 5, 6, fallback=body_height * 0.22)
        hip_width = self._distance(points, 11, 12, fallback=shoulder_width * 0.75)

        regions: list[RegionPrediction] = []
        priors = keypoint_region_priors(points, threshold)
        depth_meters = (
            None
            if not include_geometry
            or effective_depth is None
            or intrinsics is None
            else depth_to_meters(effective_depth, intrinsics.depth_scale)
        )

        def add(
            name: str,
            center: np.ndarray | None,
            box: list[float] | None,
            indices: list[int],
        ) -> None:
            confidence = _confidence(points, indices, threshold)
            region_keypoints = [COCO_KEYPOINT_NAMES[index] for index in indices]
            regions.append(
                RegionPrediction(
                    name=name,
                    center_2d=None if center is None else [float(center[0]), float(center[1])],
                    bbox_2d=box,
                    center_3d=None if center is None else center_3d(
                        center,
                        effective_depth,
                        intrinsics,
                        image_size=image_size,
                        min_depth_m=min_depth_m,
                        max_depth_m=max_depth_m,
                        reference_depth_m=reference_depth_m,
                        max_depth_delta_m=max_depth_delta_m,
                    ),
                    confidence=confidence,
                    visible=box is not None and confidence > 0.0,
                    keypoints=region_keypoints,
                    geometry_3d=(
                        None
                        if not include_geometry or box is None or effective_depth is None or intrinsics is None
                        else region_geometry(
                            effective_depth,
                            box,
                            intrinsics,
                            reference_image_size=image_size,
                            stride=geometry_stride,
                            min_depth_m=min_depth_m,
                            max_depth_m=max_depth_m,
                            reference_depth_m=reference_depth_m,
                            max_depth_delta_m=max_depth_delta_m,
                            depth_meters=depth_meters,
                        )
                    ),
                )
            )

        add("head", head, _bbox(points, [0, 1, 2, 3, 4], body_height * 0.04, threshold), [0, 1, 2, 3, 4])

        neck = None if head is None or shoulder is None else (head + shoulder) / 2.0
        add("neck", neck, _box_from_center(neck, shoulder_width * 0.34, body_height * 0.14), [0, 5, 6])

        left_shoulder = self._point(points, 5, threshold)
        right_shoulder = self._point(points, 6, threshold)
        add("left_shoulder", left_shoulder, _box_from_center(left_shoulder, shoulder_width * 0.42, body_height * 0.22), [5, 7, 11])
        add("right_shoulder", right_shoulder, _box_from_center(right_shoulder, shoulder_width * 0.42, body_height * 0.22), [6, 8, 12])

        chest = self._interpolate(shoulder, hips, 0.25)
        abdomen = self._interpolate(shoulder, hips, 0.58)
        pelvis = hips
        add("chest", chest, _box_from_center(chest, shoulder_width * 1.15, body_height * 0.23), [5, 6, 11, 12])
        add("abdomen", abdomen, _box_from_center(abdomen, max(hip_width, shoulder_width * 0.9), body_height * 0.24), [5, 6, 11, 12])
        add("pelvis", pelvis, _box_from_center(pelvis, hip_width * 1.35, body_height * 0.2), [11, 12])

        spine_indices = [5, 6, 11, 12]
        add("spine", self._interpolate(shoulder, hips, 0.5), _bbox(points, spine_indices, shoulder_width * 0.12, threshold), spine_indices)

        add("left_arm", self._polyline_center(points, [5, 7, 9], threshold), _bbox(points, [5, 7, 9], body_height * 0.045, threshold), [5, 7, 9])
        add("right_arm", self._polyline_center(points, [6, 8, 10], threshold), _bbox(points, [6, 8, 10], body_height * 0.045, threshold), [6, 8, 10])
        add("left_leg", self._polyline_center(points, [11, 13, 15], threshold), _bbox(points, [11, 13, 15], body_height * 0.055, threshold), [11, 13, 15])
        add("right_leg", self._polyline_center(points, [12, 14, 16], threshold), _bbox(points, [12, 14, 16], body_height * 0.055, threshold), [12, 14, 16])

        return {
            "schema": "body_regions.coco17.v1",
            "keypoint_names": COCO_KEYPOINT_NAMES,
            "keypoint_threshold": threshold,
            "valid_keypoint_count": int(valid_points.shape[0]),
            "region_priors": {
                "center": priors["center"].tolist(),
                "size": priors["size"].tolist(),
                "valid": priors["valid"].tolist(),
                "score": priors["score"].tolist(),
            },
            "geometry": {
                "enabled": bool(include_geometry),
                "depth_mask_applied": bool(depth_mask is not None),
                "depth_range_m": [float(min_depth_m), float(max_depth_m)],
                "stride": int(max(geometry_stride, 1)),
            },
            "person": {
                "bbox_2d": [float(valid_points[:, 0].min()), float(valid_points[:, 1].min()), float(valid_points[:, 0].max()), float(valid_points[:, 1].max())],
                "confidence": float(np.clip(valid_points[:, 2].mean(), 0.0, 1.0)),
                "body_height_px": body_height,
            },
            "regions": [region.to_dict() for region in regions],
        }

    @staticmethod
    def _point(points: np.ndarray, index: int, threshold: float) -> np.ndarray | None:
        if points[index, 2] < threshold or not np.all(np.isfinite(points[index, :2])):
            return None
        return points[index, :2].copy()

    @staticmethod
    def _distance(points: np.ndarray, left: int, right: int, fallback: float) -> float:
        a = points[left]
        b = points[right]
        if a[2] <= 0 or b[2] <= 0 or not np.all(np.isfinite(a[:2])) or not np.all(np.isfinite(b[:2])):
            return float(fallback)
        return max(float(np.linalg.norm(a[:2] - b[:2])), 1.0)

    @staticmethod
    def _interpolate(a: np.ndarray | None, b: np.ndarray | None, weight: float) -> np.ndarray | None:
        if a is None or b is None:
            return None
        return a * (1.0 - weight) + b * weight

    @staticmethod
    def _polyline_center(points: np.ndarray, indices: list[int], threshold: float) -> np.ndarray | None:
        selected = _valid(points[indices], threshold)
        if selected.size == 0:
            return None
        return selected[:, :2].mean(axis=0)
