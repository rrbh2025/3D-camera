from __future__ import annotations

from typing import Any

import numpy as np


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _first_instance_axis(value: Any, *, ndim: int) -> np.ndarray:
    array = _numpy(value)
    if array.ndim == ndim + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != ndim:
        raise ValueError(f"expected an array with {ndim} dimensions, got {array.shape}")
    return array


def _normalize_scores(scores: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.shape[0] != count:
        raise ValueError("score count does not match the number of people")
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    if values.size and (float(values.min()) < 0.0 or float(values.max()) > 1.0):
        values = np.clip(values, 0.0, None)
        maximum = float(values.max())
        if maximum > 0.0:
            values /= maximum
    return np.clip(values, 0.0, 1.0)


def _derive_boxes(
    keypoints: np.ndarray,
    scores: np.ndarray,
    *,
    keypoint_threshold: float,
    padding: float,
) -> np.ndarray:
    boxes = np.zeros((keypoints.shape[0], 4), dtype=np.float32)
    for index, points in enumerate(keypoints):
        visible = (
            np.isfinite(points[:, :2]).all(axis=1)
            & (scores[index] >= float(keypoint_threshold))
        )
        if not np.any(visible):
            continue
        xy = points[visible, :2]
        lower = xy.min(axis=0)
        upper = xy.max(axis=0)
        margin = np.maximum(upper - lower, 2.0) * max(float(padding), 0.0)
        boxes[index] = np.concatenate((lower - margin, upper + margin))
    return boxes


def extract_rtmo_people(
    instances: Any,
    *,
    body_keypoint_count: int = 17,
    keypoint_threshold: float = 0.10,
    bbox_padding: float = 0.10,
    min_bbox_score: float = 0.0,
) -> dict[str, np.ndarray]:
    """Convert an MMPose RTMO ``pred_instances`` object to project arrays.

    RTMO is a bottom-up model, so all people are retained. The caller is
    responsible for depth-based nearest-person selection and temporal locking.
    The adapter also accepts an empty prediction and derives boxes when a
    backend does not expose ``bboxes``.
    """
    count = int(body_keypoint_count)
    if count <= 0:
        raise ValueError("body_keypoint_count must be positive")
    raw_keypoints = _first_instance_axis(instances.keypoints, ndim=3)
    if raw_keypoints.ndim != 3 or raw_keypoints.shape[-1] < 2:
        raise ValueError(f"RTMO keypoints must have shape [N, K, 2], got {raw_keypoints.shape}")
    if raw_keypoints.shape[1] < count:
        raise ValueError(
            f"RTMO returned {raw_keypoints.shape[1]} keypoints; {count} are required"
        )
    keypoints = raw_keypoints[:, :count, :2].astype(np.float32, copy=False)
    person_count = keypoints.shape[0]

    raw_keypoint_scores = getattr(instances, "keypoint_scores", None)
    if raw_keypoint_scores is None:
        keypoint_scores = np.ones((person_count, count), dtype=np.float32)
    else:
        keypoint_scores = _first_instance_axis(raw_keypoint_scores, ndim=2)
        if keypoint_scores.shape[0] != person_count or keypoint_scores.shape[1] < count:
            raise ValueError("RTMO keypoint scores do not match keypoints")
        keypoint_scores = _normalize_scores(
            keypoint_scores[:, :count], person_count * count
        ).reshape(person_count, count)

    raw_boxes = getattr(instances, "bboxes", None)
    if raw_boxes is None:
        boxes = _derive_boxes(
            keypoints,
            keypoint_scores,
            keypoint_threshold=keypoint_threshold,
            padding=bbox_padding,
        )
    else:
        boxes = _first_instance_axis(raw_boxes, ndim=2)
        if boxes.shape[0] != person_count or boxes.shape[1] < 4:
            raise ValueError("RTMO boxes do not match keypoints")
        boxes = boxes[:, :4].astype(np.float32, copy=False)

    raw_bbox_scores = getattr(instances, "bbox_scores", None)
    if raw_bbox_scores is None:
        raw_bbox_scores = getattr(instances, "scores", None)
    if raw_bbox_scores is None:
        bbox_scores = keypoint_scores.mean(axis=1)
    else:
        bbox_scores = _first_instance_axis(raw_bbox_scores, ndim=1)
        bbox_scores = _normalize_scores(bbox_scores, person_count)

    finite_boxes = np.isfinite(boxes).all(axis=1)
    nonempty_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    keep = finite_boxes & nonempty_boxes & (bbox_scores >= float(min_bbox_score))
    return {
        "keypoints": np.concatenate(
            (keypoints, keypoint_scores[..., None]), axis=-1
        )[keep].astype(np.float32, copy=False),
        "bboxes": boxes[keep].astype(np.float32, copy=False),
        "scores": bbox_scores[keep].astype(np.float32, copy=False),
        "keypoint_scores": keypoint_scores[keep].astype(np.float32, copy=False),
    }
