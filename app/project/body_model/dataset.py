from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .depth import CameraIntrinsics, center_3d as project_center_3d
from .regions import (
    COCO_KEYPOINT_NAMES,
    REGION_NAMES,
    body_region_centers_3d,
)


COCO_FLIP_PAIRS = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)
REGION_FLIP_NAMES = {
    "left_shoulder": "right_shoulder",
    "right_shoulder": "left_shoulder",
    "left_arm": "right_arm",
    "right_arm": "left_arm",
    "left_leg": "right_leg",
    "right_leg": "left_leg",
}


def _load_array(path: Path, depth: bool = False) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("cv2 is required to load image-based RGB-D records") from error
    flag = cv2.IMREAD_UNCHANGED if depth else cv2.IMREAD_COLOR
    value = cv2.imread(str(path), flag)
    if value is None:
        raise FileNotFoundError(path)
    if not depth:
        value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    return value


def _resolve(
    root: Path,
    value: str | None,
    *,
    fallback_root: Path | None = None,
) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    primary = root / path
    if fallback_root is not None and not primary.is_file():
        fallback = fallback_root / path
        if fallback.is_file():
            return fallback
    return primary


def _record_source_root(manifest: Path, record: dict[str, Any]) -> Path | None:
    """Return the source session directory recorded by the curation stage."""
    curation = record.get("curation")
    if not isinstance(curation, dict):
        return None
    source_manifest = curation.get("source_manifest")
    if not source_manifest:
        return None
    source_path = Path(str(source_manifest))
    if not source_path.is_absolute():
        source_path = manifest.parent / source_path
    return source_path.parent


def _normalize_keypoints(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 3 if array.size % 3 == 0 else 2)
    if array.shape[-1] == 2:
        array = np.concatenate([array, np.ones((*array.shape[:-1], 1), dtype=np.float32)], axis=-1)
    if array.shape != (len(COCO_KEYPOINT_NAMES), 3):
        raise ValueError("dataset keypoints must have shape [17, 3]")
    return array


def _resize_depth_to_rgb(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = shape
    if depth.shape == (target_height, target_width):
        return depth
    if depth.ndim != 2:
        raise ValueError("depth must be a two-dimensional array")
    try:
        import cv2
    except ImportError:
        # Keep the loader usable for .npy smoke tests and minimal Windows
        # environments; production camera ingestion still uses OpenCV/SDK
        # alignment before this stage.
        rows = np.minimum(
            (np.arange(target_height) * depth.shape[0] // target_height),
            depth.shape[0] - 1,
        )
        cols = np.minimum(
            (np.arange(target_width) * depth.shape[1] // target_width),
            depth.shape[1] - 1,
        )
        return depth[rows[:, None], cols[None, :]]
    # Nearest-neighbour keeps invalid zero depth values and metric samples
    # intact; RGB-D hardware alignment is handled before this training stage.
    return cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)


def _resize_mask_to_rgb(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = shape
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("foreground mask must be two-dimensional")
    if mask.shape == (target_height, target_width):
        return mask.astype(bool, copy=False)
    try:
        import cv2
    except ImportError:
        rows = np.minimum(
            (np.arange(target_height) * mask.shape[0] // target_height),
            mask.shape[0] - 1,
        )
        cols = np.minimum(
            (np.arange(target_width) * mask.shape[1] // target_width),
            mask.shape[1] - 1,
        )
        return mask[rows[:, None], cols[None, :]].astype(bool, copy=False)
    return cv2.resize(
        mask.astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool, copy=False)


def _resize_rgbd(array: np.ndarray, width: int, height: int) -> np.ndarray:
    if array.shape[:2] == (height, width):
        return array
    try:
        import cv2
    except ImportError:
        rows = np.minimum(
            (np.arange(height) * array.shape[0] // height),
            array.shape[0] - 1,
        )
        cols = np.minimum(
            (np.arange(width) * array.shape[1] // width),
            array.shape[1] - 1,
        )
        return array[rows[:, None], cols[None, :]]
    return cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR)


def _augment_depth(
    depth: np.ndarray,
    depth_scale: float,
    *,
    noise_m: float = 0.0,
    dropout_probability: float = 0.0,
    scale_jitter: float = 0.0,
) -> np.ndarray:
    """Simulate common RGB-D sensor errors while preserving invalid holes."""
    values = np.asarray(depth, dtype=np.float32).copy()
    scale = float(depth_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("depth_scale must be positive")
    metric = values * scale
    valid = np.isfinite(metric) & (metric >= 0.2) & (metric <= 10.0)
    if not np.any(valid):
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    jitter = max(float(scale_jitter), 0.0)
    if jitter > 0.0:
        factor = np.random.uniform(max(0.5, 1.0 - jitter), 1.0 + jitter)
        metric[valid] *= factor
    sigma = max(float(noise_m), 0.0)
    if sigma > 0.0:
        metric[valid] += np.random.normal(0.0, sigma, size=int(valid.sum()))
    probability = float(np.clip(dropout_probability, 0.0, 1.0))
    if probability > 0.0:
        holes = valid & (np.random.random(metric.shape) < probability)
        metric[holes] = 0.0
    metric = np.clip(
        np.nan_to_num(metric, nan=0.0, posinf=10.0, neginf=0.0),
        0.0,
        10.0,
    )
    return metric / scale


def _normalize_xy(value: Any, width: int, height: int) -> np.ndarray:
    point = np.asarray(value, dtype=np.float32).reshape(2)
    return np.clip(point / np.asarray([width, height], dtype=np.float32), 0.0, 1.0)


def _clip_crop(value: Any, width: int, height: int) -> tuple[int, int, int, int]:
    crop = np.asarray(value, dtype=np.float32).reshape(4)
    x0 = int(np.clip(np.floor(crop[0]), 0, max(width - 1, 0)))
    y0 = int(np.clip(np.floor(crop[1]), 0, max(height - 1, 0)))
    x1 = int(np.clip(np.ceil(crop[2]), x0 + 1, width))
    y1 = int(np.clip(np.ceil(crop[3]), y0 + 1, height))
    return x0, y0, x1, y1


def _parse_metric_keypoints(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    output = np.full((len(COCO_KEYPOINT_NAMES), 3), np.nan, dtype=np.float32)
    try:
        values = list(value)
    except TypeError:
        return None
    for index, point in enumerate(values[: len(COCO_KEYPOINT_NAMES)]):
        if point is None:
            continue
        try:
            array = np.asarray(point, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            continue
        if array.size == 3 and np.all(np.isfinite(array)):
            output[index] = array
    return output


def _metric_features(points: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, float]:
    """Encode camera-space joints as root-relative scale-normalized features."""
    features = np.zeros((len(COCO_KEYPOINT_NAMES), 4), dtype=np.float32)
    context = np.zeros((4,), dtype=np.float32)
    if points is None:
        return features, context, 0.0
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (len(COCO_KEYPOINT_NAMES), 3):
        return features, context, 0.0
    valid = np.all(np.isfinite(points), axis=1)
    if int(valid.sum()) < 4:
        return features, context, 0.0
    root_candidates = points[[11, 12]]
    root_valid = np.all(np.isfinite(root_candidates), axis=1)
    root = (
        root_candidates[root_valid].mean(axis=0)
        if np.any(root_valid)
        else points[valid].mean(axis=0)
    )
    shoulder = points[[5, 6]]
    if np.all(np.isfinite(shoulder)):
        scale = float(np.linalg.norm(shoulder[0] - shoulder[1]))
    else:
        scale = float(np.linalg.norm(points[0] - root)) if valid[0] else 0.0
    scale = max(scale, 0.10)
    relative = np.zeros_like(points)
    relative[valid] = (points[valid] - root) / scale
    features[:, :3] = np.nan_to_num(relative, nan=0.0, posinf=0.0, neginf=0.0)
    features[:, 3] = valid.astype(np.float32)
    context[:] = np.asarray([root[0], root[1], root[2], scale], dtype=np.float32)
    context = np.clip(context, -10.0, 10.0)
    return features, context, 1.0


def _metric_region_geometry(
    points: np.ndarray | None,
    scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive metric region centers from camera-space keypoints when available."""
    centers = np.zeros((len(REGION_NAMES), 3), dtype=np.float32)
    available = np.zeros((len(REGION_NAMES),), dtype=np.float32)
    if points is None:
        return centers, available
    try:
        derived = body_region_centers_3d(points, scores=scores)
    except (TypeError, ValueError):
        return centers, available
    for index, name in enumerate(REGION_NAMES):
        point = np.asarray(derived[name], dtype=np.float32).reshape(-1)
        if point.shape == (3,) and np.all(np.isfinite(point)):
            centers[index] = point
            available[index] = 1.0
    return centers, available


def _metric_keypoints_from_depth(
    keypoints_pixels: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    image_size: tuple[int, int],
) -> np.ndarray:
    points = np.full((len(COCO_KEYPOINT_NAMES), 3), np.nan, dtype=np.float32)
    for index, keypoint in enumerate(keypoints_pixels):
        if keypoint[2] <= 0.0 or not np.all(np.isfinite(keypoint[:2])):
            continue
        projected = project_center_3d(
            keypoint[:2],
            depth,
            intrinsics,
            image_size=image_size,
            min_depth_m=0.2,
            max_depth_m=10.0,
        )
        if projected is not None:
            points[index] = np.asarray(projected, dtype=np.float32)
    return points


def _apply_metric_keypoint_augmentation(
    points: np.ndarray | None,
    normalized_xy_noise: np.ndarray,
    intrinsics: CameraIntrinsics | None,
    image_size: tuple[int, int],
    dropped: np.ndarray,
) -> np.ndarray | None:
    """Keep metric keypoints consistent with augmented image keypoints.

    ``normalized_xy_noise`` is expressed in the original image coordinate
    system. A pinhole back-projection converts it to camera-space X/Y error at
    each joint depth; dropped 2D joints are made unavailable in 3D as well.
    """
    if points is None:
        return None
    output = np.asarray(points, dtype=np.float32).copy()
    if output.shape != (len(COCO_KEYPOINT_NAMES), 3):
        return output
    noise = np.asarray(normalized_xy_noise, dtype=np.float32)
    if noise.shape != (len(COCO_KEYPOINT_NAMES), 2):
        raise ValueError("normalized_xy_noise must have shape [17, 2]")
    dropped_array = np.asarray(dropped, dtype=bool).reshape(-1)
    if dropped_array.shape != (len(COCO_KEYPOINT_NAMES),):
        raise ValueError("dropped must have shape [17]")
    valid = np.all(np.isfinite(output), axis=1)
    if intrinsics is not None:
        width, height = image_size
        fx_normalized = float(intrinsics.fx) / max(float(width), 1.0)
        fy_normalized = float(intrinsics.fy) / max(float(height), 1.0)
        valid &= output[:, 2] > 0.0
        if fx_normalized > 0.0 and fy_normalized > 0.0:
            output[valid, 0] += (
                noise[valid, 0] * output[valid, 2] / fx_normalized
            )
            output[valid, 1] += (
                noise[valid, 1] * output[valid, 2] / fy_normalized
            )
    output[dropped_array] = np.nan
    return output


def _region_heatmaps(
    centers: np.ndarray,
    boxes: np.ndarray,
    visible: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Render soft region targets in normalized image coordinates."""
    height, width = shape
    ys = (np.arange(height, dtype=np.float32) + 0.5) / max(height, 1)
    xs = (np.arange(width, dtype=np.float32) + 0.5) / max(width, 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    heatmaps = np.zeros((len(REGION_NAMES), height, width), dtype=np.float32)
    for index in range(len(REGION_NAMES)):
        if visible[index] <= 0:
            continue
        center_x, center_y = centers[index]
        box_width = max(float(boxes[index, 2] - boxes[index, 0]), 2.0 / max(width, 1))
        box_height = max(float(boxes[index, 3] - boxes[index, 1]), 2.0 / max(height, 1))
        sigma_x = max(box_width * 0.30, 1.0 / max(width, 1))
        sigma_y = max(box_height * 0.30, 1.0 / max(height, 1))
        distance = ((xx - center_x) / sigma_x) ** 2 + ((yy - center_y) / sigma_y) ** 2
        heatmaps[index] = np.exp(-0.5 * distance)
    return heatmaps


def build_rgbd_inference_sample(
    rgb: np.ndarray,
    depth: np.ndarray,
    keypoints_pixels: Any,
    intrinsics: CameraIntrinsics | dict[str, float] | None,
    depth_scale: float,
    *,
    crop_xyxy: Any | None = None,
    foreground_mask: np.ndarray | None = None,
    image_size: tuple[int, int] = (256, 256),
    record_id: str = "live",
) -> dict[str, Any]:
    """Build a model input sample directly from an in-memory RGB-D frame.

    The on-disk dataset and the live camera path must apply the same crop,
    intrinsics and metric-keypoint conventions. This helper keeps those
    conventions identical without writing a temporary manifest or image.
    ``rgb`` is RGB HWC and ``depth`` contains raw SDK depth units.
    """
    rgb_array = np.asarray(rgb)
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("rgb must have shape [H, W, 3]")
    if np.issubdtype(rgb_array.dtype, np.floating):
        maximum = float(np.nanmax(rgb_array)) if rgb_array.size else 0.0
        if maximum <= 1.5:
            rgb_array = rgb_array * 255.0
        rgb_array = np.clip(rgb_array, 0.0, 255.0).astype(np.uint8)
    else:
        rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8, copy=False)

    depth_array = np.asarray(depth)
    if depth_array.ndim != 2:
        raise ValueError("depth must have shape [H, W]")
    full_height, full_width = rgb_array.shape[:2]
    depth_array = _resize_depth_to_rgb(
        depth_array,
        (full_height, full_width),
    ).astype(np.float32, copy=False)
    mask_array = None
    if foreground_mask is not None:
        mask_array = _resize_mask_to_rgb(
            np.asarray(foreground_mask),
            (full_height, full_width),
        )

    crop_applied = crop_xyxy is not None
    if crop_xyxy is None:
        crop_x0, crop_y0, crop_x1, crop_y1 = 0, 0, full_width, full_height
    else:
        crop_x0, crop_y0, crop_x1, crop_y1 = _clip_crop(
            crop_xyxy,
            full_width,
            full_height,
        )
    rgb_crop = np.ascontiguousarray(rgb_array[crop_y0:crop_y1, crop_x0:crop_x1])
    depth_crop = np.ascontiguousarray(
        depth_array[crop_y0:crop_y1, crop_x0:crop_x1]
    )
    if mask_array is not None:
        mask_crop = np.ascontiguousarray(
            mask_array[crop_y0:crop_y1, crop_x0:crop_x1]
        )
        depth_crop = np.where(mask_crop, depth_crop, 0.0)
    else:
        mask_crop = None

    original_height, original_width = rgb_crop.shape[:2]
    keypoints = _normalize_keypoints(keypoints_pixels).copy()
    keypoints[:, 0] -= float(crop_x0)
    keypoints[:, 1] -= float(crop_y0)
    keypoints_pixels_crop = keypoints.copy()
    keypoints[:, :2] /= np.asarray(
        [original_width, original_height],
        dtype=np.float32,
    )
    keypoints[:, :2] = np.clip(keypoints[:, :2], 0.0, 1.0)

    try:
        scale = float(depth_scale)
    except (TypeError, ValueError) as error:
        raise ValueError("depth_scale must be positive") from error
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("depth_scale must be positive")

    camera_model = None
    camera_intrinsics = np.zeros((5,), dtype=np.float32)
    if intrinsics is not None:
        if isinstance(intrinsics, CameraIntrinsics):
            base_intrinsics = intrinsics
        else:
            base_intrinsics = CameraIntrinsics.from_mapping(
                {**intrinsics, "depth_scale": scale}
            )
        camera_model = CameraIntrinsics(
            fx=base_intrinsics.fx,
            fy=base_intrinsics.fy,
            cx=base_intrinsics.cx - float(crop_x0),
            cy=base_intrinsics.cy - float(crop_y0),
            depth_scale=scale,
        )
        camera_intrinsics = np.asarray(
            [
                camera_model.fx / original_width,
                camera_model.fy / original_height,
                camera_model.cx / original_width,
                camera_model.cy / original_height,
                1.0,
            ],
            dtype=np.float32,
        )

    metric_points = None
    if camera_model is not None:
        metric_points = _metric_keypoints_from_depth(
            keypoints_pixels_crop,
            depth_crop,
            camera_model,
            image_size=(original_width, original_height),
        )
    metric_region_centers, metric_region_available = _metric_region_geometry(
        metric_points,
        scores=keypoints_pixels_crop[:, 2],
    )
    metric_keypoints, metric_context, metric_available = _metric_features(
        metric_points
    )

    depth_m = np.clip(depth_crop * scale, 0.0, 10.0)
    rgbd = np.concatenate(
        [rgb_crop.astype(np.float32) / 255.0, depth_m[..., None]],
        axis=2,
    )
    target_width, target_height = int(image_size[0]), int(image_size[1])
    if target_width <= 0 or target_height <= 0:
        raise ValueError("image_size must contain positive dimensions")
    rgbd = _resize_rgbd(rgbd, target_width, target_height)
    heatmap_height = (target_height + 7) // 8
    heatmap_width = (target_width + 7) // 8
    zero_centers = np.zeros((len(REGION_NAMES), 2), dtype=np.float32)
    zero_boxes = np.zeros((len(REGION_NAMES), 4), dtype=np.float32)
    zero_regions = np.zeros((len(REGION_NAMES),), dtype=np.float32)

    return {
        "rgbd": np.transpose(rgbd, (2, 0, 1)).astype(np.float32),
        "keypoints": keypoints.astype(np.float32),
        "metric_keypoints": metric_keypoints.astype(np.float32),
        "metric_context": metric_context.astype(np.float32),
        "metric_available": np.asarray(metric_available, dtype=np.float32),
        "metric_region_centers": metric_region_centers.astype(np.float32),
        "metric_region_available": metric_region_available.astype(np.float32),
        "modality_mask": np.asarray([1.0, 1.0], dtype=np.float32),
        "camera_intrinsics": camera_intrinsics,
        "center_2d": zero_centers,
        "bbox_2d": zero_boxes,
        "visible": zero_regions.copy(),
        "confidence": zero_regions.copy(),
        "center_3d": np.zeros((len(REGION_NAMES), 3), dtype=np.float32),
        "has_3d": zero_regions.copy(),
        "region_heatmaps": np.zeros(
            (len(REGION_NAMES), heatmap_height, heatmap_width),
            dtype=np.float32,
        ),
        "meta": {
            "id": str(record_id),
            "width": original_width,
            "height": original_height,
            "source_width": full_width,
            "source_height": full_height,
            "crop_applied": bool(crop_applied),
            "crop_xyxy": [crop_x0, crop_y0, crop_x1, crop_y1],
            "depth_mask_applied": bool(mask_crop is not None),
            "foreground_mask_valid_fraction": (
                0.0 if mask_crop is None else float(np.mean(mask_crop))
            ),
        },
    }


class RGBDBodyRegionDataset:
    """JSONL RGB-D dataset for training a learned region refinement head.

    Each line contains ``rgb``, ``depth``, optional ``intrinsics``, ``keypoints``
    and a ``regions`` mapping/list with labels from the estimator schema.
    """

    def __init__(
        self,
        manifest: str | Path,
        root: str | Path | None = None,
        image_size: tuple[int, int] | None = None,
        heatmap_stride: int = 8,
        training: bool = False,
        horizontal_flip_probability: float = 0.5,
        color_jitter: float = 0.20,
        keypoint_jitter: float = 0.01,
        keypoint_dropout: float = 0.05,
        metric_keypoint_jitter: float | None = None,
        metric_input_dropout: float = 0.0,
        depth_noise_m: float = 0.0,
        depth_dropout: float = 0.0,
        depth_scale_jitter: float = 0.0,
    ):
        self.manifest = Path(manifest)
        self.root = Path(root) if root is not None else self.manifest.parent
        self.image_size = image_size
        self.heatmap_stride = max(int(heatmap_stride), 1)
        self.training = bool(training)
        self.horizontal_flip_probability = float(
            np.clip(horizontal_flip_probability, 0.0, 1.0)
        )
        self.color_jitter = max(float(color_jitter), 0.0)
        self.keypoint_jitter = max(float(keypoint_jitter), 0.0)
        self.metric_keypoint_jitter = max(
            self.keypoint_jitter
            if metric_keypoint_jitter is None
            else float(metric_keypoint_jitter),
            0.0,
        )
        self.keypoint_dropout = float(np.clip(keypoint_dropout, 0.0, 1.0))
        self.metric_input_dropout = float(np.clip(metric_input_dropout, 0.0, 1.0))
        self.depth_noise_m = max(float(depth_noise_m), 0.0)
        self.depth_dropout = float(np.clip(depth_dropout, 0.0, 1.0))
        self.depth_scale_jitter = max(float(depth_scale_jitter), 0.0)
        self.records = [
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source_root = _record_source_root(self.manifest, record)
        rgb_path = _resolve(
            self.root,
            record.get("rgb"),
            fallback_root=source_root,
        )
        depth_path = _resolve(
            self.root,
            record.get("depth"),
            fallback_root=source_root,
        )
        if rgb_path is None:
            raise ValueError("each record requires an rgb path")
        rgb = _load_array(rgb_path, depth=False)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RGB must be HWC-3")
        full_height, full_width = rgb.shape[:2]
        if depth_path is None:
            depth = np.zeros((full_height, full_width), dtype=np.float32)
            has_depth = 0.0
        else:
            depth = _load_array(depth_path, depth=True).astype(np.float32)
            has_depth = 1.0
        if depth.ndim != 2:
            raise ValueError("RGB must be HWC-3 and depth must be HW")
        depth = _resize_depth_to_rgb(depth, (full_height, full_width))
        segmentation = record.get("segmentation") or {}
        mask_path = _resolve(
            self.root,
            segmentation.get("mask") if isinstance(segmentation, dict) else None,
            fallback_root=source_root,
        )
        foreground_mask = None
        if mask_path is not None:
            foreground_mask = _resize_mask_to_rgb(
                _load_array(mask_path, depth=True),
                (full_height, full_width),
            )
        crop = record.get("crop_xyxy")
        crop_x0 = crop_y0 = 0
        if crop is not None:
            crop_x0, crop_y0, crop_x1, crop_y1 = _clip_crop(
                crop, full_width, full_height
            )
            rgb = rgb[crop_y0:crop_y1, crop_x0:crop_x1]
            depth = depth[crop_y0:crop_y1, crop_x0:crop_x1]
            if foreground_mask is not None:
                foreground_mask = foreground_mask[crop_y0:crop_y1, crop_x0:crop_x1]
        original_height, original_width = rgb.shape[:2]
        keypoints = _normalize_keypoints(record["keypoints"]).copy()
        keypoints[:, 0] -= float(crop_x0)
        keypoints[:, 1] -= float(crop_y0)
        regions = record.get("regions", {})
        if isinstance(regions, list):
            regions = {item["name"]: item for item in regions}
        has_metric_3d_labels = any(
            label.get("center_3d") is not None for label in regions.values()
        )
        flipped = (
            self.training
            and has_depth == 0.0
            and not has_metric_3d_labels
            and np.random.random() < self.horizontal_flip_probability
        )
        if flipped:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            depth = np.ascontiguousarray(depth[:, ::-1])
            if foreground_mask is not None:
                foreground_mask = np.ascontiguousarray(foreground_mask[:, ::-1])
            keypoints[:, 0] = float(original_width) - keypoints[:, 0]
            for left, right in COCO_FLIP_PAIRS:
                keypoints[[left, right]] = keypoints[[right, left]]
        keypoint_noise = np.zeros(
            (len(COCO_KEYPOINT_NAMES), 2), dtype=np.float32
        )
        dropped_keypoints = np.zeros(
            (len(COCO_KEYPOINT_NAMES),), dtype=bool
        )
        if self.training:
            labelled = keypoints[:, 2] > 0
            if self.keypoint_jitter > 0:
                keypoint_noise = np.random.normal(
                    0.0,
                    self.keypoint_jitter,
                    size=(len(COCO_KEYPOINT_NAMES), 2),
                ).astype(np.float32)
                keypoints[labelled, :2] += (
                    keypoint_noise[labelled]
                    * np.asarray([original_width, original_height], dtype=np.float32)
                )
                keypoints[:, 0] = np.clip(
                    keypoints[:, 0], 0.0, max(float(original_width - 1), 0.0)
                )
                keypoints[:, 1] = np.clip(
                    keypoints[:, 1], 0.0, max(float(original_height - 1), 0.0)
                )
                keypoint_noise[~labelled] = 0.0
            if self.keypoint_dropout > 0:
                dropped_keypoints = labelled & (
                    np.random.random(len(COCO_KEYPOINT_NAMES))
                    < self.keypoint_dropout
                )
                keypoints[dropped_keypoints, 2] = 0.0
        keypoints_pixels = keypoints.copy()
        keypoints[:, :2] /= np.asarray(
            [original_width, original_height], dtype=np.float32
        )
        keypoints[:, :2] = np.clip(keypoints[:, :2], 0.0, 1.0)
        centers = np.zeros((len(REGION_NAMES), 2), dtype=np.float32)
        boxes = np.zeros((len(REGION_NAMES), 4), dtype=np.float32)
        visible = np.zeros((len(REGION_NAMES),), dtype=np.float32)
        confidence = np.zeros((len(REGION_NAMES),), dtype=np.float32)
        center_3d = np.zeros((len(REGION_NAMES), 3), dtype=np.float32)
        has_3d = np.zeros((len(REGION_NAMES),), dtype=np.float32)
        for region_index, name in enumerate(REGION_NAMES):
            source_name = REGION_FLIP_NAMES.get(name, name) if flipped else name
            label = regions.get(source_name, {})
            center = label.get("center_2d")
            bbox = label.get("bbox_2d")
            if center is not None:
                adjusted_center = np.asarray(center, dtype=np.float32) - np.asarray(
                    [crop_x0, crop_y0], dtype=np.float32
                )
                if flipped:
                    adjusted_center[0] = float(original_width) - adjusted_center[0]
                centers[region_index] = _normalize_xy(
                    adjusted_center, original_width, original_height
                )
            if bbox is not None:
                adjusted_bbox = np.asarray(bbox, dtype=np.float32) - np.asarray(
                    [crop_x0, crop_y0, crop_x0, crop_y0], dtype=np.float32
                )
                if flipped:
                    old_x0, old_x1 = adjusted_bbox[0], adjusted_bbox[2]
                    adjusted_bbox[0] = float(original_width) - old_x1
                    adjusted_bbox[2] = float(original_width) - old_x0
                boxes[region_index] = np.clip(
                    adjusted_bbox
                    / np.asarray([original_width, original_height, original_width, original_height], dtype=np.float32),
                    0.0,
                    1.0,
                )
            visible[region_index] = float(bool(label.get("visible", False)))
            confidence[region_index] = float(np.clip(label.get("confidence", visible[region_index]), 0.0, 1.0))
            point = label.get("center_3d")
            if point is not None:
                center_3d[region_index] = np.asarray(point, dtype=np.float32)
                has_3d[region_index] = 1.0
        rgb = rgb.astype(np.float32) / 255.0
        if self.training and self.color_jitter > 0:
            contrast = np.random.uniform(
                1.0 - self.color_jitter, 1.0 + self.color_jitter
            )
            brightness = np.random.uniform(
                -self.color_jitter * 0.5, self.color_jitter * 0.5
            )
            rgb = np.clip((rgb - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
        intrinsics_record = record.get("intrinsics") or {}
        depth_scale = float(
            record.get("depth_scale", intrinsics_record.get("depth_scale", 1.0))
        )
        if self.training and has_depth:
            depth = _augment_depth(
                depth,
                depth_scale,
                noise_m=self.depth_noise_m,
                dropout_probability=self.depth_dropout,
                scale_jitter=self.depth_scale_jitter,
            )
        if foreground_mask is not None and has_depth > 0.0:
            depth = np.where(foreground_mask, depth, 0.0)
        depth_m = np.clip(depth * depth_scale, 0.0, 10.0)
        rgbd = np.concatenate([rgb, depth_m[..., None]], axis=2)
        camera_intrinsics = np.zeros((5,), dtype=np.float32)
        intrinsics = record.get("intrinsics")
        camera_model = None
        if intrinsics is not None:
            fx = float(intrinsics.get("fx", 0.0))
            fy = float(intrinsics.get("fy", 0.0))
            cx = float(intrinsics.get("cx", 0.0)) - float(crop_x0)
            cy = float(intrinsics.get("cy", 0.0)) - float(crop_y0)
            values = np.asarray([fx, fy, cx, cy], dtype=np.float32)
            if np.all(np.isfinite(values)) and fx > 0.0 and fy > 0.0:
                # Normalized intrinsics are unchanged by the later network
                # resize and make mixed-resolution records batchable.
                camera_intrinsics = np.asarray(
                    [
                        fx / original_width,
                        fy / original_height,
                        cx / original_width,
                        cy / original_height,
                        1.0,
                    ],
                    dtype=np.float32,
                )
                camera_model = CameraIntrinsics(
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    depth_scale=depth_scale,
                )
        metric_points = _parse_metric_keypoints(record.get("keypoints_3d"))
        if self.training and metric_points is not None:
            metric_noise = keypoint_noise
            if self.metric_keypoint_jitter != self.keypoint_jitter:
                if self.keypoint_jitter > 0.0:
                    metric_noise = keypoint_noise * (
                        self.metric_keypoint_jitter / self.keypoint_jitter
                    )
                elif self.metric_keypoint_jitter > 0.0:
                    metric_noise = np.random.normal(
                        0.0,
                        self.metric_keypoint_jitter,
                        size=(len(COCO_KEYPOINT_NAMES), 2),
                    ).astype(np.float32)
                    metric_noise[~(keypoints_pixels[:, 2] > 0.0)] = 0.0
            metric_points = _apply_metric_keypoint_augmentation(
                metric_points,
                metric_noise,
                camera_model,
                (original_width, original_height),
                dropped_keypoints,
            )
        if camera_model is not None and has_depth:
            depth_points = _metric_keypoints_from_depth(
                keypoints_pixels,
                depth,
                camera_model,
                image_size=(original_width, original_height),
            )
            if metric_points is None:
                metric_points = depth_points
            else:
                missing = ~np.all(np.isfinite(metric_points), axis=1)
                missing &= ~dropped_keypoints
                metric_points[missing] = depth_points[missing]
        if (
            self.training
            and metric_points is not None
            and self.metric_input_dropout > 0.0
            and np.random.random() < self.metric_input_dropout
        ):
            # Train both the metric-input path and the RGB/2D/depth fallback.
            metric_points = None
        metric_region_centers, metric_region_available = _metric_region_geometry(
            metric_points,
            scores=keypoints_pixels[:, 2],
        )
        metric_keypoints, metric_context, metric_available = _metric_features(
            metric_points
        )
        if self.image_size is not None:
            width, height = self.image_size
            rgbd = _resize_rgbd(rgbd, width, height)
        else:
            height, width = original_height, original_width
        heatmap_height = (int(height) + self.heatmap_stride - 1) // self.heatmap_stride
        heatmap_width = (int(width) + self.heatmap_stride - 1) // self.heatmap_stride
        region_heatmaps = _region_heatmaps(
            centers,
            boxes,
            visible,
            (heatmap_height, heatmap_width),
        )
        return {
            "rgbd": np.transpose(rgbd, (2, 0, 1)).astype(np.float32),
            "keypoints": keypoints.astype(np.float32),
            "metric_keypoints": metric_keypoints,
            "metric_context": metric_context,
            "metric_available": np.asarray(metric_available, dtype=np.float32),
            "metric_region_centers": metric_region_centers,
            "metric_region_available": metric_region_available,
            "modality_mask": np.asarray([1.0, has_depth], dtype=np.float32),
            "camera_intrinsics": camera_intrinsics,
            "center_2d": centers,
            "bbox_2d": boxes,
            "visible": visible,
            "confidence": confidence,
            "center_3d": center_3d,
            "has_3d": has_3d,
            "region_heatmaps": region_heatmaps,
            "meta": {
                "id": record.get("id", str(index)),
                "width": original_width,
                "height": original_height,
                "source_width": full_width,
                "source_height": full_height,
                "crop_applied": bool(crop is not None),
                "crop_xyxy": (
                    [0, 0, full_width, full_height]
                    if crop is None
                    else [crop_x0, crop_y0, crop_x1, crop_y1]
                ),
                "depth_mask_applied": bool(
                    foreground_mask is not None and has_depth > 0.0
                ),
                "foreground_mask_valid_fraction": (
                    0.0
                    if foreground_mask is None
                    else float(np.mean(foreground_mask))
                ),
            },
        }
