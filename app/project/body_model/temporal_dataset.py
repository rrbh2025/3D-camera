from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .regions import COCO_KEYPOINT_NAMES, REGION_NAMES


TEMPORAL_FEATURE_DIM = (
    len(COCO_KEYPOINT_NAMES) * 3
    + len(REGION_NAMES) * 2
    + len(REGION_NAMES)
    + len(REGION_NAMES) * 3
    + len(REGION_NAMES)
)


def _finite(value: object, size: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        return None
    return array


def _record_frame_index(record: dict[str, Any], fallback: int) -> int:
    for key in ("source_frame", "frame_index", "frame_number"):
        value = record.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    identifier = str(record.get("id", ""))
    match = re.search(r"(\d+)(?:[^0-9]*)$", identifier)
    return int(match.group(1)) if match else int(fallback)


def sequence_group_key(record: dict[str, Any]) -> str:
    """Return a stable subject/sequence/camera key for leakage-free windows."""
    subject = record.get("subject_id", record.get("source_subject", "unknown"))
    sequence = record.get(
        "source_sequence",
        record.get("sequence_id", record.get("session_id", "sequence")),
    )
    camera = record.get("source_camera", record.get("device_id", "camera"))
    return f"{subject}|{sequence}|{camera}"


def load_temporal_records(manifest: str | Path) -> list[dict[str, Any]]:
    path = Path(manifest)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_sequence_windows(
    records: Iterable[dict[str, Any]],
    sequence_length: int,
    stride: int = 1,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Build contiguous windows without crossing subject or source-sequence boundaries."""
    length = max(int(sequence_length), 2)
    step = max(int(stride), 1)
    grouped: defaultdict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[sequence_group_key(record)].append(
            (_record_frame_index(record, index), index, record)
        )

    windows: list[tuple[str, list[dict[str, Any]]]] = []
    for group, values in sorted(grouped.items()):
        values.sort(key=lambda item: (item[0], item[1]))
        frame_indices = [item[0] for item in values]
        differences = [
            current - previous
            for previous, current in zip(frame_indices, frame_indices[1:])
            if current > previous
        ]
        expected_step = Counter(differences).most_common(1)[0][0] if differences else 1
        chunks: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        previous_frame: int | None = None
        for frame_index, _, record in values:
            if (
                previous_frame is not None
                and frame_index - previous_frame != expected_step
            ):
                if current:
                    chunks.append(current)
                current = []
            current.append(record)
            previous_frame = frame_index
        if current:
            chunks.append(current)
        for chunk in chunks:
            if len(chunk) < length:
                continue
            for start in range(0, len(chunk) - length + 1, step):
                windows.append((group, chunk[start : start + length]))
    return windows


def _person_bbox(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        (keypoints[:, 2] > 0.0)
        & np.all(np.isfinite(keypoints[:, :2]), axis=1)
    )
    if not np.any(valid):
        return np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)
    points = keypoints[valid, :2]
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    size = np.maximum(upper - lower, 1.0)
    pad = size * 0.05
    lower = lower - pad
    size = np.maximum(size + 2.0 * pad, 1.0)
    return lower.astype(np.float32), size.astype(np.float32)


def _region_labels(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = record.get("regions", {})
    if isinstance(value, list):
        return {
            str(item.get("name")): item
            for item in value
            if isinstance(item, dict) and item.get("name") is not None
        }
    if isinstance(value, dict):
        return {
            str(name): item
            for name, item in value.items()
            if isinstance(item, dict)
        }
    return {}


def _frame_arrays(
    record: dict[str, Any],
    *,
    training: bool,
    observation_noise_2d: float,
    observation_noise_3d: float,
    observation_dropout: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    keypoints = np.asarray(record.get("keypoints", []), dtype=np.float32)
    if keypoints.shape != (len(COCO_KEYPOINT_NAMES), 3):
        raise ValueError("temporal records require COCO-17 keypoints")
    lower, size = _person_bbox(keypoints)
    keypoints_xy = (keypoints[:, :2] - lower) / size
    keypoints_xy = np.clip(keypoints_xy, -0.5, 1.5)
    keypoints_input = np.concatenate(
        [keypoints_xy, np.clip(keypoints[:, 2:3], 0.0, 1.0)], axis=1
    )
    labels = _region_labels(record)
    target_2d = np.zeros((len(REGION_NAMES), 2), dtype=np.float32)
    target_visible = np.zeros((len(REGION_NAMES),), dtype=np.float32)
    target_3d = np.zeros((len(REGION_NAMES), 3), dtype=np.float32)
    target_3d_valid = np.zeros((len(REGION_NAMES),), dtype=np.float32)
    raw_3d = np.full((len(REGION_NAMES), 3), np.nan, dtype=np.float32)

    for index, name in enumerate(REGION_NAMES):
        label = labels.get(name, {})
        center_2d = _finite(label.get("center_2d"), 2)
        if center_2d is not None:
            target_2d[index] = np.clip((center_2d - lower) / size, -0.5, 1.5)
        visible = label.get("visible")
        target_visible[index] = float(
            (bool(visible) if visible is not None else center_2d is not None)
            and center_2d is not None
        )
        center_3d = _finite(label.get("center_3d"), 3)
        if center_3d is not None:
            raw_3d[index] = center_3d

    valid_3d_points = np.all(np.isfinite(raw_3d), axis=1) & (target_visible > 0.0)
    if np.any(valid_3d_points):
        pelvis_index = REGION_NAMES.index("pelvis")
        if valid_3d_points[pelvis_index]:
            origin = raw_3d[pelvis_index].copy()
        else:
            origin = np.nanmedian(raw_3d[valid_3d_points], axis=0).astype(np.float32)
        distances = np.linalg.norm(raw_3d[valid_3d_points] - origin, axis=1)
        scale = float(np.percentile(distances, 75)) if distances.size else 1.0
        scale = max(scale, 0.15)
        target_3d[valid_3d_points] = (raw_3d[valid_3d_points] - origin) / scale
        target_3d_valid[valid_3d_points] = 1.0
    else:
        origin = np.zeros(3, dtype=np.float32)
        scale = 1.0

    observed_2d = target_2d.copy()
    observed_3d = target_3d.copy()
    observed_valid_2d = target_visible.copy()
    observed_valid_3d = target_3d_valid.copy()
    if training:
        if observation_noise_2d > 0.0:
            observed_2d += rng.normal(
                0.0, observation_noise_2d, size=observed_2d.shape
            ).astype(np.float32) * observed_valid_2d[:, None]
        if observation_noise_3d > 0.0:
            observed_3d += rng.normal(
                0.0, observation_noise_3d, size=observed_3d.shape
            ).astype(np.float32) * observed_valid_3d[:, None]
        if observation_dropout > 0.0:
            drop_2d = (
                rng.random(observed_valid_2d.shape[0]) < observation_dropout
            ) & (observed_valid_2d > 0.0)
            drop_3d = (
                rng.random(observed_valid_3d.shape[0]) < observation_dropout
            ) & (observed_valid_3d > 0.0)
            observed_valid_2d[drop_2d] = 0.0
            observed_valid_3d[drop_3d] = 0.0
            observed_2d[drop_2d] = 0.0
            observed_3d[drop_3d] = 0.0
    observed_2d = np.clip(observed_2d, -0.5, 1.5)
    features = np.concatenate(
        [
            keypoints_input.reshape(-1),
            observed_2d.reshape(-1),
            observed_valid_2d,
            observed_3d.reshape(-1),
            observed_valid_3d,
        ]
    ).astype(np.float32)
    return {
        "features": features,
        "observed_center_2d": observed_2d.astype(np.float32),
        "observed_valid_2d": observed_valid_2d.astype(np.float32),
        "observed_center_3d": observed_3d.astype(np.float32),
        "observed_valid_3d": observed_valid_3d.astype(np.float32),
        "target_center_2d": target_2d,
        "target_center_3d": target_3d,
        "target_visible": target_visible,
        "target_3d_valid": target_3d_valid,
        "person_lower": lower.astype(np.float32),
        "person_size": size.astype(np.float32),
        "metric_origin": origin.astype(np.float32),
        "metric_scale": np.asarray(scale, dtype=np.float32),
    }


class TemporalRegionSequenceDataset(Dataset):
    """Sequence windows for causal body-region denoising and motion learning."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        sequence_length: int = 9,
        stride: int = 3,
        training: bool = False,
        observation_noise_2d: float = 0.02,
        observation_noise_3d: float = 0.04,
        observation_dropout: float = 0.08,
        seed: int = 2026,
    ) -> None:
        self.manifest = Path(manifest)
        self.records = load_temporal_records(self.manifest)
        self.windows = build_sequence_windows(
            self.records,
            sequence_length=sequence_length,
            stride=stride,
        )
        if not self.windows:
            raise ValueError(
                f"no contiguous windows of length {sequence_length} in {self.manifest}"
            )
        self.sequence_length = max(int(sequence_length), 2)
        self.training = bool(training)
        self.observation_noise_2d = max(float(observation_noise_2d), 0.0)
        self.observation_noise_3d = max(float(observation_noise_3d), 0.0)
        self.observation_dropout = float(np.clip(observation_dropout, 0.0, 1.0))
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        group, records = self.windows[index]
        rng = np.random.default_rng(self.seed + int(index) + (1 if self.training else 0))
        frames = [
            _frame_arrays(
                record,
                training=self.training,
                observation_noise_2d=self.observation_noise_2d,
                observation_noise_3d=self.observation_noise_3d,
                observation_dropout=self.observation_dropout,
                rng=rng,
            )
            for record in records
        ]
        stacked: dict[str, Any] = {
            key: torch.from_numpy(np.stack([frame[key] for frame in frames]))
            for key in frames[0]
        }
        stacked["window_index"] = torch.tensor(index, dtype=torch.long)
        stacked["group"] = group
        return stacked
