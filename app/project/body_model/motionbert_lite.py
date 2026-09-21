from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
import sys

import numpy as np

from .body_frame import keypoints_to_camera
from .depth import CameraIntrinsics


H36M_JOINT_NAMES = (
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "nose",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)


def _valid_xy_score(point: np.ndarray) -> bool:
    return bool(
        point.shape == (3,)
        and np.all(np.isfinite(point[:2]))
        and np.isfinite(point[2])
        and point[2] > 0.0
    )


def _mean_keypoints(points: np.ndarray, indices: list[int]) -> np.ndarray:
    selected = [points[index] for index in indices if _valid_xy_score(points[index])]
    if not selected:
        return np.zeros((3,), dtype=np.float32)
    values = np.stack(selected).astype(np.float32)
    return np.asarray(
        [values[:, 0].mean(), values[:, 1].mean(), values[:, 2].mean()],
        dtype=np.float32,
    )


def coco_to_h36m(keypoints: Iterable) -> np.ndarray:
    """Convert one COCO-17 pose to MotionBERT's H36M-17 order."""
    points = np.asarray(keypoints, dtype=np.float32)
    if points.shape != (17, 3):
        raise ValueError("COCO keypoints must have shape [17, 3]")
    output = np.zeros((17, 3), dtype=np.float32)
    derived = {
        0: [11, 12],
        7: [5, 6, 11, 12],
        8: [5, 6],
        10: [0, 1, 2, 3, 4],
    }
    for target, indices in derived.items():
        output[target] = _mean_keypoints(points, indices)
    direct = {
        1: 12,
        2: 14,
        3: 16,
        4: 11,
        5: 13,
        6: 15,
        9: 0,
        11: 5,
        12: 7,
        13: 9,
        14: 6,
        15: 8,
        16: 10,
    }
    for target, source in direct.items():
        if _valid_xy_score(points[source]):
            output[target] = points[source]
    return output


def _mean_coordinates(points: np.ndarray, indices: list[int]) -> np.ndarray:
    selected = [points[index] for index in indices if np.all(np.isfinite(points[index]))]
    if not selected:
        return np.full((3,), np.nan, dtype=np.float32)
    return np.mean(np.stack(selected).astype(np.float32), axis=0).astype(np.float32)


def _coco_metric_to_h36m(points: np.ndarray) -> np.ndarray:
    """Convert COCO camera-space points while preserving depth holes."""
    values = np.asarray(points, dtype=np.float32)
    if values.shape != (17, 3):
        raise ValueError("metric COCO points must have shape [17, 3]")
    output = np.full((17, 3), np.nan, dtype=np.float32)
    derived = {
        0: [11, 12],
        7: [5, 6, 11, 12],
        8: [5, 6],
        10: [0, 1, 2, 3, 4],
    }
    for target, indices in derived.items():
        output[target] = _mean_coordinates(values, indices)
    direct = {
        1: 12,
        2: 14,
        3: 16,
        4: 11,
        5: 13,
        6: 15,
        9: 0,
        11: 5,
        12: 7,
        13: 9,
        14: 6,
        15: 8,
        16: 10,
    }
    for target, source in direct.items():
        if np.all(np.isfinite(values[source])):
            output[target] = values[source]
    return output


def h36m_to_coco(points: Iterable) -> np.ndarray:
    """Convert H36M-17 points back to the project's COCO-17 order."""
    values = np.asarray(points, dtype=np.float32)
    if values.shape != (17, 3):
        raise ValueError("H36M points must have shape [17, 3]")
    output = np.full((17, 3), np.nan, dtype=np.float32)
    output[0] = values[9]
    for index in (1, 2, 3, 4):
        output[index] = values[10]
    mapping = {
        5: 11,
        6: 14,
        7: 12,
        8: 15,
        9: 13,
        10: 16,
        11: 4,
        12: 1,
        13: 5,
        14: 2,
        15: 6,
        16: 3,
    }
    for coco_index, h36m_index in mapping.items():
        output[coco_index] = values[h36m_index]
    return output


def similarity_align(
    source: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]] | None:
    """Align a normalized MotionBERT pose to metric camera-space points."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape [N, 3]")
    if mask.shape != (source.shape[0],):
        raise ValueError("valid must have one entry per point")
    if int(mask.sum()) < 4:
        return None
    source_valid = source[mask]
    target_valid = target[mask]
    source_center = source_valid.mean(axis=0)
    target_center = target_valid.mean(axis=0)
    source_centered = source_valid - source_center
    target_centered = target_valid - target_center
    source_energy = float(np.sum(source_centered * source_centered))
    if not np.isfinite(source_energy) or source_energy < 1e-10:
        return None
    covariance = np.zeros((3, 3), dtype=np.float64)
    for row in range(source_centered.shape[0]):
        for source_axis in range(3):
            for target_axis in range(3):
                covariance[source_axis, target_axis] += (
                    source_centered[row, source_axis]
                    * target_centered[row, target_axis]
                )
    # Horn's quaternion solution avoids NumPy's BLAS-backed SVD. This matters
    # on the Windows acquisition machine, where NumPy MKL and PyTorch can
    # otherwise load incompatible OpenMP runtimes in one process.
    xx, xy, xz = covariance[0]
    yx, yy, yz = covariance[1]
    zx, zy, zz = covariance[2]
    matrix = np.asarray(
        [
            [xx + yy + zz, yz - zy, zx - xz, xy - yx],
            [yz - zy, xx - yy - zz, xy + yx, zx + xz],
            [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
            [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
        ],
        dtype=np.float64,
    )
    quaternion = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for _ in range(64):
        updated = np.zeros((4,), dtype=np.float64)
        for row in range(4):
            for column in range(4):
                updated[row] += matrix[row, column] * quaternion[column]
        norm = math.sqrt(float(np.sum(updated * updated)))
        if norm < 1e-12 or not np.isfinite(norm):
            return None
        updated /= norm
        difference = updated - quaternion
        difference_norm = math.sqrt(float(np.sum(difference * difference)))
        if difference_norm < 1e-12:
            quaternion = updated
            break
        quaternion = updated
    qw, qx, qy, qz = quaternion
    row_rotation = np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy)],
            [2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx)],
            [2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    rotated = np.zeros_like(source_centered)
    for row in range(source_centered.shape[0]):
        for target_axis in range(3):
            for source_axis in range(3):
                rotated[row, target_axis] += (
                    source_centered[row, source_axis]
                    * row_rotation[source_axis, target_axis]
                )
    numerator = float(np.sum(rotated * target_centered))
    scale = numerator / source_energy
    if not np.isfinite(scale) or scale <= 1e-8:
        return None
    aligned = np.zeros_like(source)
    centered_source = source - source_center
    for row in range(source.shape[0]):
        for target_axis in range(3):
            for source_axis in range(3):
                aligned[row, target_axis] += (
                    centered_source[row, source_axis]
                    * row_rotation[source_axis, target_axis]
                )
            aligned[row, target_axis] = (
                aligned[row, target_axis] * scale + target_center[target_axis]
            )
    residual = aligned[mask] - target_valid
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return aligned.astype(np.float32), {
        "scale": float(scale),
        "rmse_m": rmse,
        "points": float(mask.sum()),
    }


def _crop_from_pose(
    keypoints: np.ndarray,
    bbox_xyxy: Iterable | None,
    padding: float,
) -> tuple[np.ndarray, float]:
    if bbox_xyxy is not None:
        box = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
        if box.shape == (4,) and np.all(np.isfinite(box)):
            lower, upper = box[:2], box[2:]
            size = np.maximum(upper - lower, 2.0)
            return (lower + upper) * 0.5, float(np.max(size) * (1.0 + padding))
    valid = keypoints[:, 2] > 0.0
    if not np.any(valid):
        return np.zeros((2,), dtype=np.float32), 2.0
    lower = np.min(keypoints[valid, :2], axis=0)
    upper = np.max(keypoints[valid, :2], axis=0)
    return (lower + upper) * 0.5, float(np.max(np.maximum(upper - lower, 2.0)) * (1.0 + padding))


class MotionBERTLiteRuntime:
    """Optional MotionBERT-Lite 3D prior for the live RGB-D pipeline.

    The public checkpoint predicts normalized H36M 3D pose from a temporal
    2D window. It is not a metric depth model. When enough strict RGB-D
    keypoint projections are available, a similarity transform anchors the
    prediction to the camera coordinate system. Only that aligned result may
    be used as a metric fallback; the raw prior remains an audit output.
    """

    CHECKPOINT_MAX_LENGTH = 243

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        source_root: str | Path,
        device: str = "cuda:0",
        clip_length: int = 243,
        min_frames: int = 9,
        min_metric_points: int = 4,
        crop_padding: float = 0.12,
    ) -> None:
        import torch

        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.source_root = str(Path(source_root).expanduser().resolve())
        if not Path(self.checkpoint).is_file():
            raise FileNotFoundError(f"MotionBERT checkpoint not found: {self.checkpoint}")
        if not Path(self.source_root).is_dir():
            raise FileNotFoundError(f"MotionBERT source tree not found: {self.source_root}")
        self.device = torch.device(
            device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        self.clip_length = min(
            max(int(clip_length), 2), self.CHECKPOINT_MAX_LENGTH
        )
        self.min_frames = min(max(int(min_frames), 1), self.clip_length)
        self.min_metric_points = max(int(min_metric_points), 4)
        self.crop_padding = max(float(crop_padding), 0.0)
        self._torch = torch
        if self.source_root not in sys.path:
            sys.path.insert(0, self.source_root)
        from lib.utils.learning import load_backbone

        model_args = SimpleNamespace(
            backbone="DSTformer",
            dim_feat=256,
            dim_rep=512,
            depth=5,
            num_heads=8,
            mlp_ratio=4,
            maxlen=self.CHECKPOINT_MAX_LENGTH,
            num_joints=17,
        )
        self.model = load_backbone(model_args)
        checkpoint_data = torch.load(
            self.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint_data.get("model_pos", checkpoint_data)
        normalized_state = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state.items()
        }
        self.model.load_state_dict(normalized_state, strict=True)
        self.model.to(self.device).eval()
        self._frames: deque[np.ndarray] = deque(maxlen=self.clip_length)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": "motionbert_lite_runtime.v1",
            "checkpoint": self.checkpoint,
            "source_root": self.source_root,
            "device": str(self.device),
            "clip_length": int(self.clip_length),
            "min_frames": int(self.min_frames),
            "min_metric_points": int(self.min_metric_points),
            "crop_padding": float(self.crop_padding),
        }

    def reset(self) -> None:
        self._frames.clear()

    def update(
        self,
        *,
        keypoints: Iterable,
        bbox_xyxy: Iterable | None = None,
        depth: np.ndarray | None = None,
        intrinsics: CameraIntrinsics | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        points = np.asarray(keypoints, dtype=np.float32)
        if points.shape != (17, 3):
            raise ValueError("MotionBERT input keypoints must have shape [17, 3]")
        h36m = coco_to_h36m(points)
        self._frames.append(h36m)
        crop_center, crop_size = _crop_from_pose(
            points, bbox_xyxy, self.crop_padding
        )
        normalized = np.stack(list(self._frames)).astype(np.float32)
        normalized[..., :2] = (
            normalized[..., :2] - crop_center.reshape(1, 1, 2)
        ) * (2.0 / max(crop_size, 1e-6))
        normalized[..., :2] = np.clip(normalized[..., :2], -1.0, 1.0)
        if len(self._frames) < self.min_frames:
            return {
                **self.metadata(),
                "ready": False,
                "history_length": len(self._frames),
                "crop_center_xy": crop_center.tolist(),
                "crop_size_px": float(crop_size),
                "metric_aligned": False,
                "metric_alignment_points": 0,
                "keypoints_3d_prior": None,
                "keypoints_3d_camera": None,
            }

        tensor = self._torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            raw_h36m = self.model(tensor)[0, -1].detach().cpu().numpy()
        raw_h36m = np.asarray(raw_h36m, dtype=np.float32)
        prior_coco = h36m_to_coco(raw_h36m)
        aligned_coco: np.ndarray | None = None
        alignment: dict[str, float] | None = None
        if depth is not None and intrinsics is not None:
            metric_coco = keypoints_to_camera(
                points,
                np.asarray(depth),
                intrinsics,
                image_size=image_size,
                reference_depth_m=None,
            )
            metric_h36m = _coco_metric_to_h36m(metric_coco)
            valid = np.all(np.isfinite(metric_h36m), axis=1) & np.all(
                np.isfinite(raw_h36m), axis=1
            )
            if int(valid.sum()) >= self.min_metric_points:
                aligned = similarity_align(raw_h36m, metric_h36m, valid)
                if aligned is not None:
                    aligned_h36m, alignment = aligned
                    aligned_coco = h36m_to_coco(aligned_h36m)
        return {
            **self.metadata(),
            "ready": True,
            "history_length": len(self._frames),
            "crop_center_xy": crop_center.tolist(),
            "crop_size_px": float(crop_size),
            "metric_aligned": aligned_coco is not None,
            "metric_alignment_points": int(alignment["points"]) if alignment else 0,
            "metric_alignment_scale": (
                float(alignment["scale"]) if alignment else None
            ),
            "metric_alignment_rmse_m": (
                float(alignment["rmse_m"]) if alignment else None
            ),
            "keypoints_3d_prior": prior_coco.tolist(),
            "keypoints_3d_camera": (
                aligned_coco.tolist() if aligned_coco is not None else None
            ),
        }
