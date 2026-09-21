from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from body_model.mmpose_compat import prepare_mmdet_inference
from body_model.rtmo import extract_rtmo_people


POSE_BACKENDS = ("rtmo-bottomup", "rtmw-x-topdown")
DEFAULT_RTMX_CONFIG = (
    "/home/u25600009140214/body_model_link/configs/"
    "rtmw_x_cocktail14_384x288.py"
)
DEFAULT_RTMX_CHECKPOINT = (
    "/home/u25600009140214/data/RTMW_X_Cocktail14/"
    "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
)
DEFAULT_DETECTOR_CONFIG = (
    "/home/u25600009140214/data/MMPose/source/demo/mmdetection_cfg/"
    "rtmdet_nano_320-8xb32_coco-person.py"
)
DEFAULT_DETECTOR_CHECKPOINT = (
    "/home/u25600009140214/data/RTMPose_m_COCO/rtmdet-nano-person.pth"
)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _squeeze_singleton(value: Any, expected_ndim: int) -> np.ndarray:
    array = _numpy(value)
    while array.ndim > expected_ndim and array.shape[0] == 1:
        array = array[0]
    if array.ndim != expected_ndim:
        raise ValueError(
            f"expected {expected_ndim} dimensions, got {array.shape}"
        )
    return array


def _normalize_scores(scores: Any, count: int) -> np.ndarray:
    values = np.asarray(_numpy(scores), dtype=np.float32).reshape(-1)
    if values.shape[0] != count:
        raise ValueError(
            f"score count {values.shape[0]} does not match detection count {count}"
        )
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(values, 0.0, 1.0)


def extract_person_detections(
    detection_result: Any,
    *,
    score_threshold: float = 0.30,
    person_label: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract COCO-person boxes from an MMDetection result.

    The adapter accepts both ``DetDataSample`` and its ``pred_instances``
    member. Returned boxes are ``xyxy`` float32 arrays and scores are in
    ``[0, 1]``. Filtering happens before top-down pose inference.
    """
    instances = getattr(detection_result, "pred_instances", detection_result)
    raw_boxes = getattr(instances, "bboxes", None)
    raw_scores = getattr(instances, "scores", None)
    raw_labels = getattr(instances, "labels", None)
    if raw_boxes is None or raw_scores is None:
        raise ValueError("person detector result must expose bboxes and scores")

    boxes = _squeeze_singleton(raw_boxes, 2)
    if boxes.shape[1] < 4:
        raise ValueError(f"person detector boxes must have 4 columns, got {boxes.shape}")
    boxes = boxes[:, :4].astype(np.float32, copy=False)
    scores = _normalize_scores(raw_scores, boxes.shape[0])
    if raw_labels is None:
        labels = np.zeros((boxes.shape[0],), dtype=np.int64)
    else:
        labels = _squeeze_singleton(raw_labels, 1).astype(np.int64, copy=False)
        if labels.shape[0] != boxes.shape[0]:
            raise ValueError("person detector labels do not match boxes")

    finite = np.isfinite(boxes).all(axis=1)
    nonempty = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    keep = (
        finite
        & nonempty
        & (labels == int(person_label))
        & (scores >= float(score_threshold))
    )
    return boxes[keep], scores[keep]


def _extract_topdown_instance(
    sample: Any,
    *,
    body_keypoint_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    instances = getattr(sample, "pred_instances", sample)
    raw_keypoints = getattr(instances, "keypoints", None)
    if raw_keypoints is None:
        raise ValueError("top-down pose result must expose keypoints")
    keypoints = _squeeze_singleton(raw_keypoints, 2)
    if keypoints.shape[1] < 2 or keypoints.shape[0] < body_keypoint_count:
        raise ValueError(
            "top-down pose result returned too few keypoints: "
            f"{keypoints.shape}"
        )
    keypoints = keypoints[:body_keypoint_count, :2].astype(np.float32, copy=False)

    raw_scores = getattr(instances, "keypoint_scores", None)
    if raw_scores is None:
        scores = np.ones((body_keypoint_count,), dtype=np.float32)
    else:
        scores = _squeeze_singleton(raw_scores, 1).astype(np.float32, copy=False)
        if scores.shape[0] < body_keypoint_count:
            raise ValueError("top-down keypoint scores are shorter than keypoints")
        scores = scores[:body_keypoint_count]
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        if scores.size and (float(scores.min()) < 0.0 or float(scores.max()) > 1.0):
            scores = np.clip(scores, 0.0, None)
            maximum = float(scores.max())
            if maximum > 0.0:
                scores /= maximum
        scores = np.clip(scores, 0.0, 1.0)
    return keypoints, scores


def extract_topdown_people(
    pose_results: list[Any],
    boxes: Any,
    box_scores: Any,
    *,
    body_keypoint_count: int = 17,
) -> dict[str, np.ndarray]:
    """Convert RTMW-X top-down results to the project pose schema."""
    boxes_array = _squeeze_singleton(boxes, 2).astype(np.float32, copy=False)
    if boxes_array.shape[1] < 4:
        raise ValueError(f"top-down boxes must have 4 columns, got {boxes_array.shape}")
    boxes_array = boxes_array[:, :4]
    scores_array = _normalize_scores(box_scores, boxes_array.shape[0])
    if len(pose_results) != boxes_array.shape[0]:
        raise ValueError(
            "top-down pose result count does not match detector boxes: "
            f"{len(pose_results)} != {boxes_array.shape[0]}"
        )

    keypoint_rows: list[np.ndarray] = []
    score_rows: list[np.ndarray] = []
    for sample in pose_results:
        keypoints, scores = _extract_topdown_instance(
            sample,
            body_keypoint_count=body_keypoint_count,
        )
        keypoint_rows.append(keypoints)
        score_rows.append(scores)
    if not keypoint_rows:
        return {
            "keypoints": np.empty((0, body_keypoint_count, 3), dtype=np.float32),
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
            "keypoint_scores": np.empty(
                (0, body_keypoint_count), dtype=np.float32
            ),
        }
    keypoint_scores = np.stack(score_rows).astype(np.float32, copy=False)
    return {
        "keypoints": np.concatenate(
            [np.stack(keypoint_rows), keypoint_scores[..., None]], axis=-1
        ).astype(np.float32, copy=False),
        "bboxes": boxes_array,
        "scores": scores_array,
        "keypoint_scores": keypoint_scores,
    }


def _load_person_detector(
    config: str,
    checkpoint: str,
    *,
    mmpose_root: str,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    mmpose_path = Path(mmpose_root).expanduser().resolve()
    if not mmpose_path.is_dir():
        raise RuntimeError(
            f"MMPose source tree was not found at {mmpose_path}; "
            "RTMW-X top-down mode also needs the detector config tree."
        )
    sys.path.insert(0, str(mmpose_path))
    try:
        import torch
        compatibility = prepare_mmdet_inference()
        from mmdet.apis import init_detector
    except (ImportError, ModuleNotFoundError, OSError) as error:
        raise RuntimeError(
            "MMDetection is required for RTMW-X top-down mode. "
            "Install the pinned mmdet runtime in the existing environment. "
            f"Original error: {error}"
        ) from error

    actual_device = (
        device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    previous_cwd = Path.cwd()
    try:
        os.chdir(mmpose_path)
        detector = init_detector(config, checkpoint, device=actual_device)
    finally:
        os.chdir(previous_cwd)
    return detector, {
        "detector_config": str(config),
        "detector_checkpoint": str(checkpoint),
        "detector_device": actual_device,
        "mmdet_compatibility": compatibility,
    }


def _boxes_from_keypoints(
    keypoints: np.ndarray,
    keypoint_scores: np.ndarray,
    fallback: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Update cached top-down boxes from the latest pose without detection."""
    boxes = np.asarray(fallback, dtype=np.float32).copy()
    points = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(keypoint_scores, dtype=np.float32)
    if points.ndim != 3 or points.shape[2] < 2 or scores.ndim != 2:
        return boxes
    for index in range(min(points.shape[0], boxes.shape[0], scores.shape[0])):
        valid = (
            np.isfinite(points[index, :, :2]).all(axis=1)
            & np.isfinite(scores[index])
            & (scores[index] >= 0.10)
        )
        if int(valid.sum()) < 4:
            continue
        xy = points[index, valid, :2]
        lower = xy.min(axis=0)
        upper = xy.max(axis=0)
        size = np.maximum(upper - lower, 1.0)
        padding = np.maximum(size * np.asarray([0.18, 0.14], dtype=np.float32), 8.0)
        box = np.asarray(
            [lower[0] - padding[0], lower[1] - padding[1], upper[0] + padding[0], upper[1] + padding[1]],
            dtype=np.float32,
        )
        box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(image_width))
        box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(image_height))
        if box[2] > box[0] and box[3] > box[1]:
            boxes[index] = box
    return boxes


@dataclass
class PoseFrontend:
    """Unified multi-person pose adapter used by the RGB-D pipeline."""

    backend: str
    pose_model: Any
    device: str
    pose_config: str
    pose_checkpoint: str
    body_keypoint_count: int = 17
    detector: Any | None = None
    detector_config: str | None = None
    detector_checkpoint: str | None = None
    detector_threshold: float = 0.30
    compatibility: dict[str, Any] | None = None
    detector_interval: int = 1
    _cached_boxes: np.ndarray | None = field(default=None, init=False, repr=False)
    _cached_scores: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_detector_frame: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.backend = normalize_pose_backend(self.backend)
        if self.backend == "rtmw-x-topdown" and self.detector is None:
            raise ValueError("RTMW-X top-down mode requires a person detector")
        self.detector_interval = max(int(self.detector_interval), 1)

    def reset(self) -> None:
        """Clear short-term detection reuse state between camera sessions."""
        self._cached_boxes = None
        self._cached_scores = None
        self._last_detector_frame = None

    @property
    def label(self) -> str:
        return self.backend

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "pose_config": self.pose_config,
            "pose_checkpoint": self.pose_checkpoint,
            "device": self.device,
            "body_keypoint_count": int(self.body_keypoint_count),
            "detector_config": self.detector_config,
            "detector_checkpoint": self.detector_checkpoint,
            "detector_threshold": float(self.detector_threshold),
            "detector_interval": int(self.detector_interval),
            "compatibility": self.compatibility or {},
        }

    def infer(
        self,
        image: np.ndarray,
        *,
        pose_threshold: float = 0.30,
        keypoint_threshold: float = 0.10,
        frame_index: int | None = None,
    ) -> dict[str, np.ndarray]:
        from mmengine.registry import init_default_scope

        if self.backend == "rtmo-bottomup":
            from mmpose.apis import inference_bottomup

            init_default_scope("mmpose")
            pose_result = inference_bottomup(self.pose_model, image)[0]
            return extract_rtmo_people(
                pose_result.pred_instances,
                body_keypoint_count=self.body_keypoint_count,
                keypoint_threshold=keypoint_threshold,
                min_bbox_score=pose_threshold,
            )

        from mmdet.apis import inference_detector
        from mmpose.apis import inference_topdown

        reuse_detection = (
            self.detector_interval > 1
            and frame_index is not None
            and self._cached_boxes is not None
            and self._cached_scores is not None
            and self._last_detector_frame is not None
            and int(frame_index) - int(self._last_detector_frame) < self.detector_interval
        )
        if reuse_detection:
            boxes = self._cached_boxes.copy()
            box_scores = self._cached_scores.copy()
        else:
            init_default_scope("mmdet")
            detection_result = inference_detector(self.detector, image)
            boxes, box_scores = extract_person_detections(
                detection_result,
                score_threshold=self.detector_threshold,
            )
            self._last_detector_frame = (
                None if frame_index is None else int(frame_index)
            )
            if boxes.shape[0] == 0:
                self._cached_boxes = None
                self._cached_scores = None
        if boxes.shape[0] == 0:
            output = {
                "keypoints": np.empty(
                    (0, self.body_keypoint_count, 3), dtype=np.float32
                ),
                "bboxes": boxes,
                "scores": box_scores,
                "keypoint_scores": np.empty(
                    (0, self.body_keypoint_count), dtype=np.float32
                ),
            }
            output["detector_reused"] = bool(reuse_detection)
            return output
        init_default_scope("mmpose")
        pose_results = inference_topdown(
            self.pose_model,
            image,
            bboxes=boxes,
            bbox_format="xyxy",
        )
        output = extract_topdown_people(
            pose_results,
            boxes,
            box_scores,
            body_keypoint_count=self.body_keypoint_count,
        )
        self._cached_boxes = _boxes_from_keypoints(
            output["keypoints"],
            output["keypoint_scores"],
            boxes,
            image_width=int(image.shape[1]),
            image_height=int(image.shape[0]),
        )
        self._cached_scores = output["scores"].copy()
        output["detector_reused"] = bool(reuse_detection)
        return output


def normalize_pose_backend(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "rtmo": "rtmo-bottomup",
        "rtmo-l": "rtmo-bottomup",
        "bottomup": "rtmo-bottomup",
        "rtmw-x": "rtmw-x-topdown",
        "rtmw": "rtmw-x-topdown",
        "topdown": "rtmw-x-topdown",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in POSE_BACKENDS:
        raise ValueError(
            f"unknown pose backend {value!r}; expected one of {POSE_BACKENDS}"
        )
    return normalized


def validate_pose_assets(
    backend: str,
    pose_config: str,
    pose_checkpoint: str,
) -> None:
    """Fail early when a top-down/bottom-up backend is paired with wrong weights."""
    normalized = normalize_pose_backend(backend)
    identifiers = f"{pose_config} {pose_checkpoint}".lower()
    has_rtmo_hint = "rtmo" in identifiers
    has_rtmw_hint = "rtmw" in identifiers or "rtmpose" in identifiers
    if normalized == "rtmw-x-topdown" and has_rtmo_hint and not has_rtmw_hint:
        raise ValueError(
            "rtmw-x-topdown requires RTMW-X/RTMPose weights, but RTMO assets "
            f"were supplied: config={pose_config!r}, checkpoint={pose_checkpoint!r}"
        )
    if normalized == "rtmo-bottomup" and has_rtmw_hint and not has_rtmo_hint:
        raise ValueError(
            "rtmo-bottomup requires RTMO weights, but RTMW-X/RTMPose assets "
            f"were supplied: config={pose_config!r}, checkpoint={pose_checkpoint!r}"
        )


def build_pose_frontend(
    backend: str,
    *,
    pose_model: Any,
    device: str,
    pose_config: str,
    pose_checkpoint: str,
    mmpose_root: str,
    body_keypoint_count: int = 17,
    detector_config: str | None = None,
    detector_checkpoint: str | None = None,
    detector_threshold: float = 0.30,
    detector_interval: int = 1,
    compatibility: dict[str, Any] | None = None,
) -> PoseFrontend:
    normalized = normalize_pose_backend(backend)
    validate_pose_assets(normalized, pose_config, pose_checkpoint)
    detector = None
    detector_metadata: dict[str, Any] = {}
    if normalized == "rtmw-x-topdown":
        if not detector_config or not detector_checkpoint:
            raise ValueError(
                "RTMW-X top-down mode requires --detector-config and "
                "--detector-checkpoint"
            )
        detector, detector_metadata = _load_person_detector(
            detector_config,
            detector_checkpoint,
            mmpose_root=mmpose_root,
            device=device,
        )
    merged_compatibility = dict(compatibility or {})
    merged_compatibility.update(detector_metadata)
    return PoseFrontend(
        backend=normalized,
        pose_model=pose_model,
        device=device,
        pose_config=pose_config,
        pose_checkpoint=pose_checkpoint,
        body_keypoint_count=body_keypoint_count,
        detector=detector,
        detector_config=detector_config,
        detector_checkpoint=detector_checkpoint,
        detector_threshold=detector_threshold,
        detector_interval=detector_interval,
        compatibility=merged_compatibility,
    )
