from __future__ import annotations

import time
from typing import Iterable

import numpy as np


class TemporalRegionFilter:
    """Confidence-aware online smoothing for learned 2D/3D region outputs.

    A bounded constant-velocity prediction reduces lag during short, smooth
    movements while confidence gating prevents an isolated bad detection from
    pulling the tracked region away from the target.
    """

    def __init__(
        self,
        tau_seconds: float = 0.12,
        min_alpha: float = 0.12,
        max_alpha: float = 0.80,
        max_jump_px: float = 180.0,
        max_jump_m: float = 0.75,
        metric_max_speed_m_s: float = 4.0,
        metric_noise_floor_m: float = 0.05,
        sparse_metric_max_jump_m: float = 0.08,
        min_metric_geometry_points: int = 12,
        use_velocity: bool = False,
        velocity_momentum: float = 0.50,
        max_velocity_px: float = 2500.0,
        max_velocity_m: float = 5.0,
        learned_only: bool = False,
        smooth_metric_geometry: bool = False,
        max_hold_frames: int = 3,
    ) -> None:
        self.tau_seconds = max(float(tau_seconds), 1e-3)
        self.min_alpha = float(np.clip(min_alpha, 0.0, 1.0))
        self.max_alpha = float(np.clip(max_alpha, self.min_alpha, 1.0))
        self.max_jump_px = max(float(max_jump_px), 1.0)
        self.max_jump_m = max(float(max_jump_m), 1e-3)
        self.metric_max_speed_m_s = max(float(metric_max_speed_m_s), 0.0)
        self.metric_noise_floor_m = max(float(metric_noise_floor_m), 0.0)
        self.sparse_metric_max_jump_m = max(
            float(sparse_metric_max_jump_m), 1e-3
        )
        self.min_metric_geometry_points = max(
            int(min_metric_geometry_points), 1
        )
        self.use_velocity = bool(use_velocity)
        self.velocity_momentum = float(np.clip(velocity_momentum, 0.0, 1.0))
        self.max_velocity_px = max(float(max_velocity_px), 1.0)
        self.max_velocity_m = max(float(max_velocity_m), 1e-3)
        self.learned_only = bool(learned_only)
        self.smooth_metric_geometry = bool(smooth_metric_geometry)
        self.max_hold_frames = max(int(max_hold_frames), 0)
        self._state: dict[str, dict] = {}
        self._timestamp: float | None = None

    def reset(self) -> None:
        self._state = {}
        self._timestamp = None

    @staticmethod
    def _finite(value: object, size: int) -> np.ndarray | None:
        try:
            point = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if point.shape != (size,) or not np.all(np.isfinite(point)):
            return None
        return point

    @staticmethod
    def _clip_velocity(value: np.ndarray, maximum: float) -> np.ndarray:
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm <= float(maximum) or norm <= 1e-8:
            return value
        return value * (float(maximum) / norm)

    @staticmethod
    def _metric_geometry_point_count(region: dict) -> int | None:
        """Return point quality when the live region schema provides it."""
        if "geometry_3d" not in region:
            return None
        geometry = region.get("geometry_3d")
        if not isinstance(geometry, dict):
            return 0
        try:
            return max(int(geometry.get("point_count", 0) or 0), 0)
        except (TypeError, ValueError):
            return 0

    def _should_smooth_field(self, region: dict, key: str) -> bool:
        if not self.learned_only:
            return True
        source_key = (
            "center_2d_source"
            if key in {"center_2d", "bbox_2d"}
            else "center_3d_source"
        )
        return str(region.get(source_key, "")).startswith("learned_")

    def update(
        self,
        regions: Iterable[dict],
        *,
        timestamp: float | None = None,
        target_changed: bool = False,
    ) -> list[dict]:
        if target_changed:
            self.reset()
        now = time.monotonic() if timestamp is None else float(timestamp)
        if self._timestamp is None:
            base_alpha = 1.0
            dt = None
        else:
            dt = max(now - self._timestamp, 1e-3)
            base_alpha = 1.0 - np.exp(-dt / self.tau_seconds)
            base_alpha = float(np.clip(base_alpha, self.min_alpha, self.max_alpha))
        self._timestamp = now

        output: list[dict] = []
        for source in regions:
            region = dict(source)
            name = str(region.get("name", ""))
            confidence = float(np.clip(region.get("confidence", 0.0), 0.0, 1.0))
            alpha = base_alpha if name not in self._state else base_alpha * (
                0.35 + 0.65 * confidence
            )
            previous_state = self._state.get(name)
            previous = (
                None if previous_state is None else previous_state.get("region")
            )
            velocity_2d = np.asarray(
                (previous_state or {}).get("velocity_2d", [0.0, 0.0]),
                dtype=np.float32,
            )
            velocity_3d = np.asarray(
                (previous_state or {}).get("velocity_3d", [0.0, 0.0, 0.0]),
                dtype=np.float32,
            )
            velocity_2d = self._clip_velocity(
                velocity_2d.reshape(2), self.max_velocity_px
            )
            velocity_3d = self._clip_velocity(
                velocity_3d.reshape(3), self.max_velocity_m
            )
            predicted_only = False
            temporal_held_2d = False
            temporal_held_3d = False
            state_values: dict[str, list[float] | None] = {}
            state_missing: dict[str, int] = {}
            # The public live schema calls the metric center ``center_3d``.
            # ``center_3d_camera`` remains supported for older inferencer
            # records and direct unit-test inputs.
            for key, size, max_jump in (
                ("center_2d", 2, self.max_jump_px),
                ("center_3d", 3, self.max_jump_m),
            ):
                field = key
                if key == "center_3d" and field not in region:
                    field = "center_3d_camera"
                current = self._finite(region.get(field), size)
                state_key = "value_2d" if key == "center_2d" else "value_3d"
                missing_key = (
                    "missing_2d" if key == "center_2d" else "missing_3d"
                )
                previous_value = self._finite(
                    (previous_state or {}).get(state_key), size
                )
                if previous_value is None and previous is not None:
                    previous_value = self._finite(previous.get(field), size)
                missing_count = int((previous_state or {}).get(missing_key, 0))
                if key == "center_2d":
                    velocity = velocity_2d
                    velocity_limit = self.max_velocity_px
                else:
                    velocity = velocity_3d
                    velocity_limit = self.max_velocity_m
                predicted = previous_value
                if (
                    self.use_velocity
                    and previous_value is not None
                    and dt is not None
                ):
                    predicted = previous_value + velocity * float(dt)
                if current is None:
                    missing_count += 1
                    if (
                        predicted is not None
                        and missing_count <= self.max_hold_frames
                    ):
                        region[field] = predicted.tolist()
                        predicted_only = True
                        if key == "center_2d":
                            temporal_held_2d = True
                        else:
                            temporal_held_3d = True
                            region["center_3d_source"] = "temporal_hold_last_valid"
                    state_values[state_key] = (
                        None if previous_value is None else previous_value.tolist()
                    )
                    state_missing[missing_key] = missing_count
                    continue
                missing_count = 0
                metric_source = str(region.get("center_3d_source", ""))
                is_metric_geometry = (
                    key == "center_3d"
                    and metric_source in {
                        "metric_keypoint_geometry",
                        "aligned_depth_geometry",
                        "aligned_depth_region_geometry",
                    }
                    and not self.smooth_metric_geometry
                )
                if not self._should_smooth_field(region, key) and not is_metric_geometry:
                    region[field] = current.tolist()
                    state_values[state_key] = current.tolist()
                    state_missing[missing_key] = missing_count
                    continue
                if is_metric_geometry:
                    # Keep calibrated measurements latency-free when they are
                    # plausible. A sudden large jump is usually a foreground
                    # hole, background surface, or a pose point on an image
                    # boundary, so hold the last reliable metric value even
                    # when the current detector confidence is high.
                    if previous_value is not None:
                        jump = float(np.linalg.norm(current - previous_value))
                        metric_jump_limit = float(max_jump)
                        if dt is not None:
                            metric_jump_limit = min(
                                metric_jump_limit,
                                self.metric_noise_floor_m
                                + self.metric_max_speed_m_s * float(dt),
                            )
                        current_points = self._metric_geometry_point_count(region)
                        previous_points = self._metric_geometry_point_count(
                            previous or {}
                        )
                        quality_counts = [
                            value
                            for value in (current_points, previous_points)
                            if value is not None
                        ]
                        if quality_counts and any(
                            value < self.min_metric_geometry_points
                            for value in quality_counts
                        ):
                            metric_jump_limit = min(
                                metric_jump_limit,
                                self.sparse_metric_max_jump_m,
                            )
                        region["metric_jump_limit_m"] = float(metric_jump_limit)
                        if jump > metric_jump_limit:
                            current = predicted if predicted is not None else previous_value
                            predicted_only = True
                            temporal_held_3d = True
                            region["center_3d_source"] = (
                                "temporal_rejected_metric_jump"
                            )
                    region[field] = current.tolist()
                    state_values[state_key] = current.tolist()
                    state_missing[missing_key] = missing_count
                    continue
                if previous_value is not None:
                    reference = previous_value
                    jump = float(np.linalg.norm(current - reference))
                    learned_source = str(
                        region.get(
                            "center_2d_source"
                            if key == "center_2d"
                            else "center_3d_source",
                            "",
                        )
                    ).startswith("learned_")
                    if jump > max_jump and (
                        confidence < 0.75 or learned_source
                    ):
                        current = predicted if predicted is not None else reference
                        if key == "center_2d":
                            velocity_2d *= 1.0 - self.velocity_momentum
                        else:
                            velocity_3d *= 1.0 - self.velocity_momentum
                    else:
                        # Keep every valid observation on the original
                        # confidence-aware smoother. Velocity is reserved for
                        # dropouts and rejected low-confidence jumps.
                        current = previous_value + alpha * (current - previous_value)
                        if self.use_velocity and dt is not None and dt > 0.0:
                            measured = self._finite(region.get(field), size)
                            if measured is None:
                                measured = current
                            measured_velocity = self._clip_velocity(
                                (measured - previous_value)
                                / float(dt),
                                velocity_limit,
                            )
                            updated_velocity = (
                                (1.0 - self.velocity_momentum) * velocity
                                + self.velocity_momentum * measured_velocity
                            )
                            if key == "center_2d":
                                velocity_2d = updated_velocity
                            else:
                                velocity_3d = updated_velocity
                    region[field] = current.tolist()
                state_values[state_key] = current.tolist()
                state_missing[missing_key] = missing_count

            if previous is not None and "bbox_2d" in region and self._should_smooth_field(
                region, "bbox_2d"
            ):
                current_box = self._finite(region.get("bbox_2d"), 4)
                previous_box = self._finite(previous.get("bbox_2d"), 4)
                if previous_box is not None and current_box is None:
                    if self.use_velocity and dt is not None:
                        translation = np.asarray(
                            [velocity_2d[0], velocity_2d[1], velocity_2d[0], velocity_2d[1]],
                            dtype=np.float32,
                        ) * float(dt)
                        region["bbox_2d"] = (previous_box + translation).tolist()
                        predicted_only = True
                elif current_box is not None and previous_box is not None:
                    region["bbox_2d"] = (
                        previous_box + alpha * (current_box - previous_box)
                    ).tolist()
            region["temporal_confidence"] = confidence
            region["motion_prediction_only"] = bool(predicted_only)
            region["temporal_held_2d"] = bool(temporal_held_2d)
            region["temporal_held_3d"] = bool(temporal_held_3d)
            self._state[name] = {
                "region": dict(region),
                "velocity_2d": velocity_2d.tolist(),
                "velocity_3d": velocity_3d.tolist(),
                **state_values,
                **state_missing,
            }
            output.append(region)
        return output
