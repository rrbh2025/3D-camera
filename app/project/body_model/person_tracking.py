from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


TRACKING_SCHEMA = "nearest_person_tracking.v1"


def bbox_iou(first: Iterable[float], second: Iterable[float]) -> float:
    """Return intersection-over-union for two ``xyxy`` boxes."""
    a = np.asarray(first, dtype=np.float32).reshape(4)
    b = np.asarray(second, dtype=np.float32).reshape(4)
    lower = np.maximum(a[:2], b[:2])
    upper = np.minimum(a[2:], b[2:])
    size = np.maximum(upper - lower, 0.0)
    intersection = float(size[0] * size[1])
    area_a = float(np.prod(np.maximum(a[2:] - a[:2], 0.0)))
    area_b = float(np.prod(np.maximum(b[2:] - b[:2], 0.0)))
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _bbox_center(box: Iterable[float]) -> np.ndarray:
    values = np.asarray(box, dtype=np.float32).reshape(4)
    return (values[:2] + values[2:]) * 0.5


def _bbox_scale(box: Iterable[float]) -> float:
    values = np.asarray(box, dtype=np.float32).reshape(4)
    width = max(float(values[2] - values[0]), 1.0)
    height = max(float(values[3] - values[1]), 1.0)
    return max(float(np.sqrt(width * height)), 1.0)


def _bbox_center_distance(first: Iterable[float], second: Iterable[float]) -> float:
    return float(np.linalg.norm(_bbox_center(first) - _bbox_center(second)))


def _bbox_size_similarity(first: Iterable[float], second: Iterable[float]) -> float:
    """Compare box dimensions without making the association depend on area."""
    first_values = np.asarray(first, dtype=np.float32).reshape(4)
    second_values = np.asarray(second, dtype=np.float32).reshape(4)
    first_size = np.maximum(first_values[2:] - first_values[:2], 1.0)
    second_size = np.maximum(second_values[2:] - second_values[:2], 1.0)
    log_ratio = np.abs(np.log(first_size / second_size))
    return float(np.exp(-float(log_ratio.sum())))


def robust_person_depth(
    depth: np.ndarray,
    bbox_xyxy: Iterable[float],
    depth_scale: float,
    *,
    central_fraction: float = 0.60,
    min_depth_m: float = 0.2,
    max_depth_m: float = 10.0,
    min_valid_pixels: int = 64,
    selection_depth_percentile: float = 50.0,
) -> dict[str, float | int | None]:
    """Estimate median and near-surface distance from a detection.

    The center crop reduces background leakage from loose detector boxes. A
    median over valid metric depth remains stable around holes and isolated
    foreground pixels. The lower depth percentile is kept separately for
    nearest-person ranking, so a closer visible surface is not hidden by
    background pixels in a loose box.
    """
    values = np.asarray(depth)
    if values.ndim != 2:
        raise ValueError("depth must have shape [H, W]")
    scale = float(depth_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("depth_scale must be positive")
    fraction = float(np.clip(central_fraction, 0.1, 1.0))
    box = np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)
    center = (box[:2] + box[2:]) * 0.5
    size = np.maximum(box[2:] - box[:2], 1.0) * fraction
    lower = np.floor(center - size * 0.5).astype(np.int64)
    upper = np.ceil(center + size * 0.5).astype(np.int64)
    height, width = values.shape
    x0 = int(np.clip(lower[0], 0, width))
    y0 = int(np.clip(lower[1], 0, height))
    x1 = int(np.clip(upper[0], 0, width))
    y1 = int(np.clip(upper[1], 0, height))
    if x1 <= x0 or y1 <= y0:
        return {
            "depth_m": None,
            "depth_valid_fraction": 0.0,
            "depth_valid_pixels": 0,
            "depth_roi_pixels": 0,
        }
    roi = values[y0:y1, x0:x1].astype(np.float32, copy=False) * scale
    valid = np.isfinite(roi) & (roi >= float(min_depth_m)) & (
        roi <= float(max_depth_m)
    )
    valid_values = roi[valid]
    minimum = max(int(min_valid_pixels), 1)
    if valid_values.size < minimum:
        return {
            "depth_m": None,
            "selection_depth_m": None,
            "depth_iqr_m": None,
            "depth_valid_fraction": float(valid.mean()) if valid.size else 0.0,
            "depth_valid_pixels": int(valid_values.size),
            "depth_roi_pixels": int(valid.size),
        }
    percentile = float(np.clip(selection_depth_percentile, 0.0, 50.0))
    return {
        "depth_m": float(np.median(valid_values)),
        "selection_depth_m": float(np.percentile(valid_values, percentile)),
        "depth_iqr_m": float(
            np.percentile(valid_values, 75.0)
            - np.percentile(valid_values, 25.0)
        ),
        "depth_valid_fraction": float(valid.mean()) if valid.size else 0.0,
        "depth_valid_pixels": int(valid_values.size),
        "depth_roi_pixels": int(valid.size),
    }


@dataclass
class _TrackState:
    bbox: np.ndarray
    depth_m: float | None
    track_id: int
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32)
    )
    missing_frames: int = 0


class NearestPersonTracker:
    """Select and temporally lock the nearest RGB-D person detection."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.25,
        switch_margin_m: float = 0.20,
        switch_confirm_frames: int = 3,
        central_fraction: float = 0.60,
        min_depth_m: float = 0.2,
        max_depth_m: float = 10.0,
        min_valid_pixels: int = 64,
        max_missing_frames: int = 5,
        selection_depth_percentile: float = 50.0,
    ) -> None:
        self.iou_threshold = float(np.clip(iou_threshold, 0.0, 1.0))
        self.switch_margin_m = max(float(switch_margin_m), 0.0)
        self.switch_confirm_frames = max(int(switch_confirm_frames), 1)
        self.central_fraction = float(np.clip(central_fraction, 0.1, 1.0))
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.min_valid_pixels = max(int(min_valid_pixels), 1)
        self.max_missing_frames = max(int(max_missing_frames), 0)
        self.selection_depth_percentile = float(
            np.clip(selection_depth_percentile, 0.0, 50.0)
        )
        self._state: _TrackState | None = None
        self._next_track_id = 1
        self._pending_bbox: np.ndarray | None = None
        self._pending_frames = 0

    def reset(self) -> None:
        self._state = None
        self._next_track_id = 1
        self._clear_pending()

    def _clear_pending(self) -> None:
        self._pending_bbox = None
        self._pending_frames = 0

    @staticmethod
    def _fallback_index(candidates: list[dict]) -> int:
        # Depth-less operation is explicitly a fallback. Favor a confident,
        # large person rather than silently claiming geometric nearest-person
        # selection.
        return max(
            range(len(candidates)),
            key=lambda index: (
                candidates[index]["detection_score"]
                * np.sqrt(max(candidates[index]["area_px"], 1.0))
            ),
        )

    def _nearest_index(self, candidates: list[dict]) -> tuple[int, str]:
        metric = [
            index
            for index, candidate in enumerate(candidates)
            if candidate["selection_depth_m"] is not None
        ]
        if metric:
            return min(
                metric,
                key=lambda index: candidates[index]["selection_depth_m"],
            ), "nearest_depth"
        return self._fallback_index(candidates), "depth_unavailable_fallback"

    def _start_track(self, candidate: dict) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        self._state = _TrackState(
            bbox=np.asarray(candidate["bbox_xyxy"], dtype=np.float32),
            depth_m=candidate["depth_m"],
            track_id=track_id,
            velocity=np.zeros(4, dtype=np.float32),
            missing_frames=0,
        )
        self._clear_pending()
        return track_id

    def _predict_bbox(self) -> np.ndarray | None:
        if self._state is None:
            return None
        steps = max(self._state.missing_frames + 1, 1)
        return np.asarray(
            self._state.bbox + self._state.velocity * float(steps),
            dtype=np.float32,
        )

    def _association_metrics(
        self,
        candidate_box: Iterable[float],
        predicted_box: Iterable[float] | None,
    ) -> tuple[float | None, float | None, float | None, bool]:
        if predicted_box is None:
            return None, None, None, False
        overlap = bbox_iou(predicted_box, candidate_box)
        distance = _bbox_center_distance(predicted_box, candidate_box)
        scale = _bbox_scale(predicted_box)
        center_similarity = float(
            np.exp(-distance / max(scale * 1.5, 1.0))
        )
        size_similarity = _bbox_size_similarity(predicted_box, candidate_box)
        score = 0.60 * overlap + 0.25 * center_similarity + 0.15 * size_similarity
        valid = overlap >= self.iou_threshold or distance <= max(scale * 1.5, 24.0)
        return float(overlap), float(distance), float(score), bool(valid)

    def _same_pending_target(
        self, first: Iterable[float], second: Iterable[float]
    ) -> bool:
        overlap = bbox_iou(first, second)
        distance = _bbox_center_distance(first, second)
        return overlap >= self.iou_threshold or distance <= max(
            _bbox_scale(first) * 1.5, 24.0
        )

    def update(
        self,
        bboxes_xyxy: Iterable[Iterable[float]],
        scores: Iterable[float],
        depth: np.ndarray,
        depth_scale: float,
    ) -> dict:
        boxes = np.asarray(list(bboxes_xyxy), dtype=np.float32)
        if boxes.size == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("bboxes_xyxy must have shape [N, 4]")
        confidence = np.asarray(list(scores), dtype=np.float32).reshape(-1)
        if confidence.shape[0] != boxes.shape[0]:
            raise ValueError("scores must contain one value per bounding box")

        predicted_bbox = self._predict_bbox()
        candidates: list[dict] = []
        for index, (box, score) in enumerate(zip(boxes, confidence)):
            geometry = robust_person_depth(
                depth,
                box,
                depth_scale,
                central_fraction=self.central_fraction,
                min_depth_m=self.min_depth_m,
                max_depth_m=self.max_depth_m,
                min_valid_pixels=self.min_valid_pixels,
                selection_depth_percentile=self.selection_depth_percentile,
            )
            width = max(float(box[2] - box[0]), 0.0)
            height = max(float(box[3] - box[1]), 0.0)
            association = self._association_metrics(box, predicted_bbox)
            candidates.append(
                {
                    "index": index,
                    "bbox_xyxy": box.tolist(),
                    "detection_score": float(np.clip(score, 0.0, 1.0)),
                    "area_px": width * height,
                    **geometry,
                    "iou_with_previous": (
                        None
                        if self._state is None
                        else bbox_iou(self._state.bbox, box)
                    ),
                    "iou_with_prediction": association[0],
                    "center_distance_to_prediction_px": association[1],
                    "association_score": association[2],
                    "association_valid": association[3],
                }
            )

        if not candidates:
            self._clear_pending()
            if self._state is not None:
                self._state.missing_frames += 1
                if self._state.missing_frames > self.max_missing_frames:
                    self._state = None
                    return {
                        "schema": TRACKING_SCHEMA,
                        "selected_index": None,
                        "target_track_id": None,
                        "target_changed": False,
                        "selection_reason": "track_lost_reset",
                        "depth_selection_available": False,
                        "missing_frames": self.max_missing_frames + 1,
                        "predicted_bbox_xyxy": None,
                        "association_method": "none",
                        "candidates": [],
                    }
            return {
                "schema": TRACKING_SCHEMA,
                "selected_index": None,
                "target_track_id": None if self._state is None else self._state.track_id,
                "target_changed": False,
                "selection_reason": "no_person_detected",
                "depth_selection_available": False,
                "missing_frames": (
                    0 if self._state is None else self._state.missing_frames
                ),
                "predicted_bbox_xyxy": (
                    None
                    if self._state is None
                    else self._predict_bbox().tolist()
                ),
                "association_method": "none",
                "candidates": [],
            }

        nearest_index, nearest_reason = self._nearest_index(candidates)
        selected_index = nearest_index
        changed = False
        reason = nearest_reason
        association_method = "nearest_depth"

        if self._state is None:
            self._start_track(candidates[selected_index])
            changed = True
            reason = f"initial_{nearest_reason}"
            association_method = "initial"
        else:
            associated = [
                candidate
                for candidate in candidates
                if candidate["association_valid"]
                and candidate["association_score"] is not None
            ]
            if not associated:
                self._start_track(candidates[selected_index])
                changed = True
                reason = f"reacquired_{nearest_reason}"
                association_method = "nearest_reacquire"
            else:
                current_index = int(
                    max(
                        associated,
                        key=lambda candidate: candidate["association_score"],
                    )["index"]
                )
                if current_index == nearest_index:
                    selected_index = current_index
                    self._clear_pending()
                    reason = "tracked_nearest_person"
                    association_method = "predicted_box"
                else:
                    current_depth = candidates[current_index]["selection_depth_m"]
                    challenger_depth = candidates[nearest_index]["selection_depth_m"]
                    challenger_is_closer = (
                        challenger_depth is not None
                        and (
                            current_depth is None
                            or challenger_depth + self.switch_margin_m < current_depth
                        )
                    )
                    if challenger_is_closer:
                        challenger_box = np.asarray(
                            candidates[nearest_index]["bbox_xyxy"], dtype=np.float32
                        )
                        if (
                            self._pending_bbox is not None
                            and self._same_pending_target(
                                self._pending_bbox, challenger_box
                            )
                        ):
                            self._pending_frames += 1
                        else:
                            self._pending_bbox = challenger_box
                            self._pending_frames = 1
                        if self._pending_frames >= self.switch_confirm_frames:
                            selected_index = nearest_index
                            self._start_track(candidates[selected_index])
                            changed = True
                            reason = "confirmed_closer_person"
                            association_method = "nearest_switch"
                        else:
                            selected_index = current_index
                            reason = "closer_person_pending_hysteresis"
                            association_method = "predicted_box"
                    else:
                        selected_index = current_index
                        self._clear_pending()
                        reason = "kept_tracked_person_within_depth_margin"
                        association_method = "predicted_box"

        selected = candidates[selected_index]
        if self._state is None:
            raise RuntimeError("tracker failed to initialize state")
        previous_bbox = self._state.bbox.copy()
        elapsed_frames = max(self._state.missing_frames + 1, 1)
        observed_bbox = np.asarray(selected["bbox_xyxy"], dtype=np.float32)
        instantaneous_velocity = (observed_bbox - previous_bbox) / float(
            elapsed_frames
        )
        self._state.velocity = (
            0.5 * self._state.velocity + 0.5 * instantaneous_velocity
        )
        self._state.missing_frames = 0
        self._state.bbox = observed_bbox
        self._state.depth_m = selected["depth_m"]
        return {
            "schema": TRACKING_SCHEMA,
            "selected_index": int(selected_index),
            "target_track_id": int(self._state.track_id),
            "target_changed": bool(changed),
            "selection_reason": reason,
            "depth_selection_available": any(
                candidate["selection_depth_m"] is not None
                for candidate in candidates
            ),
            "missing_frames": 0,
            "switch_pending_frames": int(self._pending_frames),
            "predicted_bbox_xyxy": (
                None if predicted_bbox is None else predicted_bbox.tolist()
            ),
            "association_method": association_method,
            "candidates": candidates,
        }
