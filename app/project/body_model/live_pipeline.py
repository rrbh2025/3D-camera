from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np

from .dataset import build_rgbd_inference_sample
from .depth import CameraIntrinsics
from .body_frame import estimate_body_frame
from .height_estimation import estimate_height
from .person_tracking import NearestPersonTracker
from .regions import BodyRegionEstimator, body_region_centers_3d
from .temporal import TemporalBodyRegionEstimator
from .temporal_regions import TemporalRegionFilter


LIVE_SCHEMA = "body_regions_rgbd_live.v1"


def _finite_value(value: Any, size: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        return None
    return array


def _learned_region_is_usable(
    region: dict[str, Any],
    *,
    min_confidence: float,
    min_presence: float,
) -> bool:
    confidence = float(region.get("confidence", 0.0) or 0.0)
    presence = float(region.get("presence_probability", 0.0) or 0.0)
    return (
        np.isfinite(confidence)
        and np.isfinite(presence)
        and confidence >= min_confidence
        and presence >= min_presence
    )


def _merge_learned_region_fallback(
    regions: list[dict[str, Any]],
    learned_regions: Any,
    *,
    min_confidence: float,
    min_presence: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fill only missing geometric fields from a learned RGB-D prediction.

    The deterministic pose/depth result remains authoritative whenever it is
    present. A learned 3D center is accepted only when it was obtained from
    aligned depth; a network-only raw 3D prediction is intentionally ignored.
    """
    learned_by_name: dict[str, dict[str, Any]] = {}
    if isinstance(learned_regions, list):
        learned_by_name = {
            str(region.get("name")): region
            for region in learned_regions
            if isinstance(region, dict) and region.get("name") is not None
        }

    merged: list[dict[str, Any]] = []
    learned_2d_count = 0
    learned_3d_count = 0
    geometry_2d_count = 0
    geometry_3d_count = 0

    for source in regions:
        region = dict(source)
        name = str(region.get("name", ""))
        learned = learned_by_name.get(name)
        geometry_center_2d = _finite_value(region.get("center_2d"), 2)
        geometry_center_3d = _finite_value(region.get("center_3d"), 3)
        if geometry_center_2d is not None:
            geometry_2d_count += 1
        if geometry_center_3d is not None and region.get("center_3d_source") != "motionbert_metric_prior":
            geometry_3d_count += 1

        used_2d_fallback = False
        used_3d_fallback = False
        if learned is not None and _learned_region_is_usable(
            learned,
            min_confidence=min_confidence,
            min_presence=min_presence,
        ):
            if geometry_center_2d is None:
                learned_center_2d = _finite_value(learned.get("center_2d"), 2)
                if learned_center_2d is not None:
                    region["center_2d"] = learned_center_2d.tolist()
                    used_2d_fallback = True
            geometry_bbox = _finite_value(region.get("bbox_2d"), 4)
            if geometry_bbox is None:
                learned_bbox = _finite_value(learned.get("bbox_2d"), 4)
                if learned_bbox is not None and learned_bbox[2] >= learned_bbox[0] and learned_bbox[3] >= learned_bbox[1]:
                    region["bbox_2d"] = learned_bbox.tolist()
                    used_2d_fallback = True

            # This guard separates calibrated aligned-depth inference from the
            # model's monocular/raw-Z fallback, which is not a measurement.
            if geometry_center_3d is None and learned.get("used_aligned_depth") is True:
                learned_center_3d = _finite_value(
                    learned.get("center_3d_camera"),
                    3,
                )
                if learned_center_3d is not None:
                    region["center_3d"] = learned_center_3d.tolist()
                    used_3d_fallback = True

        if used_2d_fallback:
            learned_2d_count += 1
        if used_3d_fallback:
            learned_3d_count += 1

        has_2d = _finite_value(region.get("center_2d"), 2) is not None
        has_3d = _finite_value(region.get("center_3d"), 3) is not None
        if used_2d_fallback or used_3d_fallback:
            region["region_source"] = "learned_rgbd_fallback"
        elif region.get("center_3d_source") == "motionbert_metric_prior":
            region["region_source"] = "motionbert_metric_prior"
        else:
            region["region_source"] = "pose_depth_geometry"
        region["center_2d_source"] = (
            "learned_rgbd_fallback" if used_2d_fallback else
            "pose_geometry" if geometry_center_2d is not None else "unavailable"
        )
        region["center_3d_source"] = (
            "learned_aligned_depth_fallback" if used_3d_fallback else
            "motionbert_metric_prior"
            if region.get("center_3d_source") == "motionbert_metric_prior"
            else "aligned_depth_geometry" if geometry_center_3d is not None else "unavailable"
        )
        region["used_learned_2d_fallback"] = bool(used_2d_fallback)
        region["used_learned_3d_fallback"] = bool(used_3d_fallback)
        region["center_3d_is_metric"] = bool(has_3d)
        merged.append(region)

    total = len(merged)
    final_2d_count = sum(_finite_value(region.get("center_2d"), 2) is not None for region in merged)
    final_3d_count = sum(_finite_value(region.get("center_3d"), 3) is not None for region in merged)
    coverage = {
        "total_regions": int(total),
        "geometry_2d_count": int(geometry_2d_count),
        "geometry_3d_count": int(geometry_3d_count),
        "final_2d_count": int(final_2d_count),
        "final_3d_count": int(final_3d_count),
        "learned_2d_fallback_count": int(learned_2d_count),
        "learned_3d_fallback_count": int(learned_3d_count),
        "motionbert_3d_prior_count": int(sum(
            region.get("center_3d_source") == "motionbert_metric_prior"
            for region in merged
        )),
        "geometry_2d_fraction": float(geometry_2d_count / total) if total else 0.0,
        "geometry_3d_fraction": float(geometry_3d_count / total) if total else 0.0,
        "final_2d_fraction": float(final_2d_count / total) if total else 0.0,
        "final_3d_fraction": float(final_3d_count / total) if total else 0.0,
    }
    return merged, coverage


def summarize_region_coverage(values: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Aggregate per-frame region coverage without hiding missing summaries."""
    records = [value for value in values if isinstance(value, dict)]
    if not records:
        return None

    def mean(key: str) -> float:
        numbers = [float(value.get(key, 0.0) or 0.0) for value in records]
        return float(np.mean(numbers)) if numbers else 0.0

    def total(key: str) -> int:
        return int(sum(int(value.get(key, 0) or 0) for value in records))

    return {
        "frames_with_coverage": int(len(records)),
        "mean_geometry_2d_count": mean("geometry_2d_count"),
        "mean_final_2d_count": mean("final_2d_count"),
        "mean_geometry_3d_count": mean("geometry_3d_count"),
        "mean_final_3d_count": mean("final_3d_count"),
        "mean_geometry_2d_fraction": mean("geometry_2d_fraction"),
        "mean_final_2d_fraction": mean("final_2d_fraction"),
        "mean_geometry_3d_fraction": mean("geometry_3d_fraction"),
        "mean_final_3d_fraction": mean("final_3d_fraction"),
        "total_learned_2d_fallback_count": total("learned_2d_fallback_count"),
        "total_learned_3d_fallback_count": total("learned_3d_fallback_count"),
        "total_motionbert_3d_prior_count": total("motionbert_3d_prior_count"),
    }


def _refresh_final_region_coverage(
    coverage: dict[str, Any],
    regions: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute final coverage after online temporal stages have run.

    Fallback and geometry counts describe the pre-filter merge. Final counts
    must instead reflect the fields returned to the caller, including a
    bounded temporal hold or a temporal refiner output.
    """
    output = dict(coverage)
    region_list = [region for region in regions if isinstance(region, dict)]
    output.setdefault("pre_temporal_final_2d_count", int(
        coverage.get("final_2d_count", 0) or 0
    ))
    output.setdefault("pre_temporal_final_3d_count", int(
        coverage.get("final_3d_count", 0) or 0
    ))
    total = len(region_list)
    output["total_regions"] = int(total)
    final_2d_count = sum(
        _finite_value(region.get("center_2d"), 2) is not None
        for region in region_list
    )
    final_3d_count = sum(
        _finite_value(region.get("center_3d"), 3) is not None
        for region in region_list
    )
    output["final_2d_count"] = int(final_2d_count)
    output["final_3d_count"] = int(final_3d_count)
    output["final_2d_fraction"] = (
        float(final_2d_count / total) if total else 0.0
    )
    output["final_3d_fraction"] = (
        float(final_3d_count / total) if total else 0.0
    )
    output["coverage_stage"] = "after_temporal_filters"
    return output


def padded_crop(
    bbox_xyxy: Any,
    width: int,
    height: int,
    padding: float = 0.12,
) -> list[int]:
    """Return a clipped crop that remains stable for a live frame."""
    box = np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)
    size = np.maximum(box[2:] - box[:2], 1.0)
    margin = size * max(float(padding), 0.0)
    crop = np.asarray(
        [box[0] - margin[0], box[1] - margin[1], box[2] + margin[0], box[3] + margin[1]],
        dtype=np.float32,
    )
    crop[[0, 2]] = np.clip(crop[[0, 2]], 0.0, float(width))
    crop[[1, 3]] = np.clip(crop[[1, 3]], 0.0, float(height))
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        crop = np.asarray([0.0, 0.0, float(width), float(height)])
    return [int(np.floor(value)) for value in crop]


def _selected_keypoints(
    people: dict[str, Any],
    index: int,
    count: int,
) -> np.ndarray:
    points = np.asarray(people.get("keypoints"), dtype=np.float32)
    if points.ndim != 3 or index < 0 or index >= points.shape[0]:
        raise ValueError("people.keypoints must have shape [N, K, 2 or 3]")
    selected = points[index, :count, :2]
    scores = people.get("keypoint_scores")
    if scores is not None:
        score_array = np.asarray(scores, dtype=np.float32)
        if score_array.ndim == 2 and index < score_array.shape[0]:
            selected_scores = score_array[index, :count]
        else:
            selected_scores = np.ones((selected.shape[0],), dtype=np.float32)
    elif points.shape[2] >= 3:
        selected_scores = points[index, :count, 2]
    else:
        selected_scores = np.ones((selected.shape[0],), dtype=np.float32)
    if selected.shape[0] != count:
        raise ValueError(f"pose backend returned fewer than {count} keypoints")
    selected_scores = np.nan_to_num(
        selected_scores.astype(np.float32),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    selected_scores = np.clip(selected_scores, 0.0, 1.0)
    return np.concatenate([selected.astype(np.float32), selected_scores[:, None]], axis=1)


def _body_coordinates(regions: list[dict], body_frame: dict | None) -> None:
    if body_frame is None:
        return
    origin = np.asarray(body_frame["origin_camera"], dtype=np.float32)
    axes = np.asarray(body_frame["axes_camera"], dtype=np.float32)
    if origin.shape != (3,) or axes.shape != (3, 3):
        return
    for region in regions:
        value = region.get("center_3d")
        if value is None:
            continue
        point = np.asarray(value, dtype=np.float32).reshape(-1)
        if point.shape == (3,) and np.all(np.isfinite(point)):
            region["center_3d_body"] = ((point - origin) @ axes).tolist()


def _prefer_region_geometry_centers(
    regions: list[dict],
    *,
    minimum_point_count: int = 12,
) -> int:
    """Replace single-keypoint depth with robust region point-cloud centers."""
    replaced = 0
    for region in regions:
        geometry = region.get("geometry_3d")
        if not isinstance(geometry, dict):
            continue
        center = _finite_value(geometry.get("center_3d"), 3)
        point_count = int(geometry.get("point_count", 0) or 0)
        if center is None or point_count < max(int(minimum_point_count), 1):
            continue
        keypoint_center = _finite_value(region.get("center_3d"), 3)
        if keypoint_center is not None:
            region["center_3d_keypoint_sample"] = keypoint_center.tolist()
        region["center_3d"] = center.tolist()
        region["center_3d_source"] = "aligned_depth_region_geometry"
        region["center_3d_geometry_valid"] = True
        replaced += 1
    return replaced


class LiveBodyRegionPipeline:
    """Frame-level RGB-D body-region pipeline shared by camera and replay paths.

    ``people`` is the normalized output of a multi-person pose backend and must
    contain ``bboxes``, ``scores``, ``keypoints`` and optionally
    ``keypoint_scores``. The tracker selects one nearest person; all anatomical
    regions are then calculated only for that selected track.
    """

    def __init__(
        self,
        *,
        tracker: NearestPersonTracker | None = None,
        temporal_estimator: TemporalBodyRegionEstimator | None = None,
        keypoint_threshold: float = 0.25,
        body_keypoint_count: int = 17,
        min_keypoints: int = 6,
        geometry_stride: int = 5,
        crop_padding: float = 0.12,
        include_region_geometry: bool = True,
        prefer_region_geometry_center: bool = True,
        learned_inferencer: Any | None = None,
        region_filter: TemporalRegionFilter | None = None,
        temporal_refiner: Any | None = None,
        motionbert_runtime: Any | None = None,
        motionbert_apply_metric_prior: bool = True,
        learned_fallback_min_confidence: float = 0.35,
        learned_fallback_min_presence: float = 0.35,
        profile_timing: bool = False,
    ) -> None:
        self.tracker = tracker or NearestPersonTracker()
        self.temporal_estimator = temporal_estimator or TemporalBodyRegionEstimator(
            BodyRegionEstimator(keypoint_threshold)
        )
        self.keypoint_threshold = float(keypoint_threshold)
        self.body_keypoint_count = int(body_keypoint_count)
        self.min_keypoints = max(int(min_keypoints), 1)
        self.geometry_stride = max(int(geometry_stride), 1)
        self.crop_padding = max(float(crop_padding), 0.0)
        self.include_region_geometry = bool(include_region_geometry)
        self.prefer_region_geometry_center = bool(prefer_region_geometry_center)
        self.learned_inferencer = learned_inferencer
        self.region_filter = region_filter
        self.temporal_refiner = temporal_refiner
        self.motionbert_runtime = motionbert_runtime
        self.motionbert_apply_metric_prior = bool(motionbert_apply_metric_prior)
        self.learned_fallback_min_confidence = float(
            np.clip(learned_fallback_min_confidence, 0.0, 1.0)
        )
        self.learned_fallback_min_presence = float(
            np.clip(learned_fallback_min_presence, 0.0, 1.0)
        )
        self.profile_timing = bool(profile_timing)
        self._has_output = False

    def reset(self) -> None:
        self.tracker.reset()
        self.temporal_estimator.reset()
        if self.region_filter is not None:
            self.region_filter.reset()
        if self.motionbert_runtime is not None:
            self.motionbert_runtime.reset()
        self._has_output = False

    def process(
        self,
        *,
        frame_index: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        depth_scale: float,
        intrinsics: CameraIntrinsics | dict[str, float],
        people: dict[str, Any],
        timestamp: float | None = None,
        foreground_mask: np.ndarray | None = None,
        frame_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pipeline_started = time.perf_counter() if self.profile_timing else 0.0
        pipeline_timing: dict[str, float] = {}

        def stage_started() -> float:
            return time.perf_counter() if self.profile_timing else 0.0

        def record_stage(name: str, started: float) -> None:
            if self.profile_timing:
                pipeline_timing[name] = float(
                    (time.perf_counter() - started) * 1000.0
                )

        def attach_timing(result: dict[str, Any]) -> None:
            if self.profile_timing:
                pipeline_timing["pipeline_total_ms"] = float(
                    (time.perf_counter() - pipeline_started) * 1000.0
                )
                result["pipeline_timing"] = dict(pipeline_timing)

        image = np.asarray(rgb)
        depth_array = np.asarray(depth)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("rgb must have shape [H, W, 3]")
        if depth_array.ndim != 2:
            raise ValueError("depth must have shape [H, W]")
        boxes = np.asarray(people.get("bboxes", []), dtype=np.float32)
        if boxes.size == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.asarray(people.get("scores", []), dtype=np.float32).reshape(-1)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("people.bboxes must have shape [N, 4]")
        tracking_started = stage_started()
        tracking = self.tracker.update(boxes, scores, depth_array, float(depth_scale))
        record_stage("tracking_ms", tracking_started)
        changed_after_initial = bool(tracking.get("target_changed") and self._has_output)
        if changed_after_initial:
            self.temporal_estimator.reset()
            if self.temporal_refiner is not None:
                self.temporal_refiner.reset()
            if self.motionbert_runtime is not None:
                self.motionbert_runtime.reset()

        result: dict[str, Any] = {
            "schema": LIVE_SCHEMA,
            "frame_index": int(frame_index),
            "frame_metadata": frame_metadata,
            "image_size": [int(image.shape[1]), int(image.shape[0])],
            "depth_scale": float(depth_scale),
            "tracking": tracking,
            "target_changed_after_initial": changed_after_initial,
            "pose_person_count": int(boxes.shape[0]),
                "status": "no_person_detected",
                "regions": [],
                "learned_regions": None,
                "region_coverage": None,
                "body_frame": None,
                "height_estimate": None,
        }
        selected_index = tracking.get("selected_index")
        if selected_index is None:
            attach_timing(result)
            return result

        selection_started = stage_started()
        keypoints = _selected_keypoints(
            people,
            int(selected_index),
            self.body_keypoint_count,
        )
        visible_count = int(np.sum(keypoints[:, 2] >= self.keypoint_threshold))
        result["selected_keypoints"] = keypoints.tolist()
        result["visible_keypoint_count"] = visible_count
        result["status"] = "insufficient_keypoints"
        record_stage("selection_ms", selection_started)
        if visible_count < self.min_keypoints:
            attach_timing(result)
            return result

        if isinstance(intrinsics, CameraIntrinsics):
            camera_intrinsics = intrinsics
        else:
            camera_intrinsics = CameraIntrinsics.from_mapping(
                {**intrinsics, "depth_scale": float(depth_scale)}
            )
        selected_candidate = tracking["candidates"][int(selected_index)]
        geometry_started = stage_started()
        region_result = self.temporal_estimator.estimate(
            keypoints,
            depth=depth_array,
            intrinsics=camera_intrinsics,
            image_size=(image.shape[1], image.shape[0]),
            timestamp=timestamp,
            include_geometry=self.include_region_geometry,
            geometry_stride=self.geometry_stride,
            reference_depth_m=selected_candidate.get("depth_m"),
            depth_mask=foreground_mask,
        )
        regions = [dict(region) for region in region_result.get("regions", [])]
        robust_geometry_count = 0
        if self.prefer_region_geometry_center:
            robust_geometry_count = _prefer_region_geometry_centers(regions)
        record_stage("region_geometry_ms", geometry_started)
        body_frame_started = stage_started()
        body_frame = estimate_body_frame(
            keypoints,
            depth_array,
            camera_intrinsics,
            image_size=(image.shape[1], image.shape[0]),
            reference_depth_m=selected_candidate.get("depth_m"),
        )
        _body_coordinates(regions, body_frame)
        record_stage("body_frame_ms", body_frame_started)
        result.update(
            {
                "status": "ok",
                "regions": regions,
                "body_frame": body_frame,
                "temporal": region_result.get("temporal"),
                "region_geometry": region_result.get("geometry"),
                "robust_geometry_center_count": int(robust_geometry_count),
                "selected_bbox_xyxy": selected_candidate.get("bbox_xyxy"),
            }
        )

        motionbert_started = stage_started()
        if self.motionbert_runtime is not None:
            try:
                motionbert = self.motionbert_runtime.update(
                    keypoints=keypoints,
                    bbox_xyxy=selected_candidate.get("bbox_xyxy"),
                    depth=depth_array,
                    intrinsics=camera_intrinsics,
                    image_size=(image.shape[1], image.shape[0]),
                )
                metric_prior = motionbert.get("keypoints_3d_camera")
                result["motionbert"] = {
                    key: value
                    for key, value in motionbert.items()
                    if key not in {"keypoints_3d_prior", "keypoints_3d_camera"}
                }
                result["motionbert_keypoints_3d_prior"] = motionbert.get(
                    "keypoints_3d_prior"
                )
                if metric_prior is not None:
                    prior_points = np.asarray(metric_prior, dtype=np.float32)
                    prior_centers = body_region_centers_3d(prior_points)
                    applied = 0
                    for region in regions:
                        name = str(region.get("name", ""))
                        prior = prior_centers.get(name)
                        if prior is None or not np.all(np.isfinite(prior)):
                            continue
                        region["center_3d_motionbert_prior"] = prior.tolist()
                        region["motionbert_prior_valid"] = True
                        if (
                            self.motionbert_apply_metric_prior
                            and _finite_value(region.get("center_3d"), 3) is None
                        ):
                            region["center_3d"] = prior.tolist()
                            region["center_3d_source"] = "motionbert_metric_prior"
                            region["used_motionbert_3d_prior"] = True
                            applied += 1
                    result["motionbert"]["applied_metric_region_count"] = int(applied)
                    _body_coordinates(regions, body_frame)
            except Exception as error:
                result["motionbert_error"] = f"{type(error).__name__}: {error}"
        record_stage("motionbert_ms", motionbert_started)

        learned_started = stage_started()
        if self.learned_inferencer is not None:
            crop = padded_crop(
                selected_candidate["bbox_xyxy"],
                int(image.shape[1]),
                int(image.shape[0]),
                self.crop_padding,
            )
            try:
                sample = build_rgbd_inference_sample(
                    image,
                    depth_array,
                    keypoints,
                    camera_intrinsics,
                    float(depth_scale),
                    crop_xyxy=crop,
                    foreground_mask=foreground_mask,
                    image_size=self.learned_inferencer.image_size,
                    record_id=f"live-{frame_index:06d}",
                )
                learned_report = self.learned_inferencer.infer(sample)
                result["learned_regions"] = learned_report.get("regions")
                result["learned_inference"] = learned_report
                regions, coverage = _merge_learned_region_fallback(
                    regions,
                    learned_report.get("regions"),
                    min_confidence=self.learned_fallback_min_confidence,
                    min_presence=self.learned_fallback_min_presence,
                )
                _body_coordinates(regions, body_frame)
                result["regions"] = regions
                result["region_coverage"] = coverage
            except Exception as error:
                result["learned_inference_error"] = f"{type(error).__name__}: {error}"
        record_stage("learned_stage_ms", learned_started)

        postprocess_started = stage_started()
        if result.get("region_coverage") is None:
            regions, coverage = _merge_learned_region_fallback(
                regions,
                None,
                min_confidence=self.learned_fallback_min_confidence,
                min_presence=self.learned_fallback_min_presence,
            )
            _body_coordinates(regions, body_frame)
            result["regions"] = regions
            result["region_coverage"] = coverage

        if self.region_filter is not None:
            regions = self.region_filter.update(
                result["regions"],
                timestamp=timestamp,
                target_changed=changed_after_initial,
            )
            _body_coordinates(regions, body_frame)
            result["regions"] = regions
            result["region_temporal"] = {
                "enabled": True,
                "learned_only": bool(self.region_filter.learned_only),
                "smooth_metric_geometry": bool(
                    self.region_filter.smooth_metric_geometry
                ),
                "use_velocity": bool(self.region_filter.use_velocity),
            }

        if self.temporal_refiner is not None:
            try:
                regions, temporal_refinement = self.temporal_refiner.update(
                    keypoints=keypoints,
                    regions=result["regions"],
                )
                _body_coordinates(regions, body_frame)
                result["regions"] = regions
                result["temporal_refiner"] = temporal_refinement
            except Exception as error:
                result["temporal_refiner_error"] = (
                    f"{type(error).__name__}: {error}"
                )

        if isinstance(result.get("region_coverage"), dict):
            for region in result["regions"]:
                region["center_3d_is_metric"] = bool(
                    _finite_value(region.get("center_3d"), 3) is not None
                )
            result["region_coverage"] = _refresh_final_region_coverage(
                result["region_coverage"],
                result["regions"],
            )

        height_started = stage_started()
        try:
            metadata = frame_metadata if isinstance(frame_metadata, dict) else {}
            result["height_estimate"] = estimate_height(
                keypoints=keypoints,
                depth=depth_array,
                depth_scale=float(depth_scale),
                intrinsics=camera_intrinsics,
                selected_bbox=result.get("selected_bbox_xyxy"),
                body_frame=body_frame,
                regions=result.get("regions"),
                posture=metadata.get("posture"),
            )
        except Exception as error:
            result["height_estimation_error"] = (
                f"{type(error).__name__}: {error}"
            )
        record_stage("height_estimation_ms", height_started)

        record_stage("postprocess_ms", postprocess_started)
        attach_timing(result)
        self._has_output = True
        return result
