from __future__ import annotations

import time
from typing import Iterable

import numpy as np

from .depth import CameraIntrinsics
from .regions import BodyRegionEstimator, _as_keypoints


class TemporalKeypointFilter:
    """Confidence-aware exponential filter for RGB-D video keypoints."""

    def __init__(
        self,
        tau_seconds: float = 0.10,
        min_alpha: float = 0.12,
        max_alpha: float = 0.80,
        max_jump_px: float = 180.0,
    ) -> None:
        self.tau_seconds = max(float(tau_seconds), 1e-3)
        self.min_alpha = float(np.clip(min_alpha, 0.0, 1.0))
        self.max_alpha = float(np.clip(max_alpha, self.min_alpha, 1.0))
        self.max_jump_px = max(float(max_jump_px), 1.0)
        self._state: np.ndarray | None = None
        self._timestamp: float | None = None

    def reset(self) -> None:
        self._state = None
        self._timestamp = None

    def update(self, keypoints: Iterable, timestamp: float | None = None) -> np.ndarray:
        current = _as_keypoints(keypoints).copy()
        now = time.monotonic() if timestamp is None else float(timestamp)
        if self._state is None or self._state.shape != current.shape:
            self._state = current
            self._timestamp = now
            return current.copy()

        dt = max(now - float(self._timestamp), 1e-3)
        base_alpha = 1.0 - np.exp(-dt / self.tau_seconds)
        base_alpha = float(np.clip(base_alpha, self.min_alpha, self.max_alpha))
        previous = self._state.copy()
        for index in range(current.shape[0]):
            score = float(np.clip(current[index, 2], 0.0, 1.0))
            previous_score = float(np.clip(previous[index, 2], 0.0, 1.0))
            if not np.all(np.isfinite(current[index, :2])) or score <= 0.0:
                current[index, :2] = previous[index, :2]
                current[index, 2] = previous_score * 0.92
                continue
            jump = float(np.linalg.norm(current[index, :2] - previous[index, :2]))
            if previous_score > 0.25 and jump > self.max_jump_px and score < 0.75:
                current[index, :2] = previous[index, :2]
                current[index, 2] = min(previous_score, score)
                continue
            alpha = base_alpha * (0.35 + 0.65 * score)
            current[index, :2] = previous[index, :2] + alpha * (
                current[index, :2] - previous[index, :2]
            )
            current[index, 2] = max(previous_score * (1.0 - alpha), score * alpha)
        self._state = current
        self._timestamp = now
        return current.copy()


class TemporalBodyRegionEstimator:
    """Pose smoothing plus the deterministic RGB-D region estimator."""

    def __init__(self, estimator: BodyRegionEstimator | None = None, **filter_kwargs):
        self.estimator = estimator or BodyRegionEstimator()
        self.filter = TemporalKeypointFilter(**filter_kwargs)

    def reset(self) -> None:
        self.filter.reset()

    def estimate(
        self,
        keypoints: Iterable,
        depth: np.ndarray | None = None,
        intrinsics: CameraIntrinsics | None = None,
        image_size: tuple[int, int] | None = None,
        timestamp: float | None = None,
        include_geometry: bool = False,
        geometry_stride: int = 3,
        min_depth_m: float = 0.2,
        max_depth_m: float = 10.0,
        reference_depth_m: float | None = None,
        max_depth_delta_m: float = 0.75,
        depth_mask: np.ndarray | None = None,
    ) -> dict:
        smoothed = self.filter.update(keypoints, timestamp=timestamp)
        result = self.estimator.estimate(
            smoothed,
            depth=depth,
            intrinsics=intrinsics,
            image_size=image_size,
            include_geometry=include_geometry,
            geometry_stride=geometry_stride,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            reference_depth_m=reference_depth_m,
            max_depth_delta_m=max_depth_delta_m,
            depth_mask=depth_mask,
        )
        result["temporal"] = {
            "enabled": True,
            "timestamp": None if timestamp is None else float(timestamp),
            "filter": "confidence_aware_ema",
            "smoothed_keypoints": smoothed.tolist(),
        }
        return result
