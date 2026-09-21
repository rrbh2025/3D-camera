from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _regions_by_name(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        return {str(name): item for name, item in value.items() if isinstance(item, dict)}
    if isinstance(value, list):
        return {
            str(item.get("name")): item
            for item in value
            if isinstance(item, dict) and item.get("name")
        }
    return {}


def _selection_depth(candidate: dict[str, Any]) -> float | None:
    """Use the surface-ranking depth, with compatibility for old events."""
    value = candidate.get("selection_depth_m")
    if value is None:
        value = candidate.get("depth_m")
    return None if value is None else float(value)


def evaluate_tracking(
    events: Iterable[dict[str, Any]],
    region_records: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize nearest-person stability without pretending it is accuracy."""
    event_list = list(events)
    selected_depths: list[float] = []
    depth_gaps: list[float] = []
    selected_valid_fractions: list[float] = []
    nearest_consistent = 0
    selected_frames = 0
    depth_available_frames = 0
    switches = 0
    initial_track_events = 0
    post_initial_switches = 0
    track_ids: list[int] = []
    reasons: defaultdict[str, int] = defaultdict(int)

    for event in event_list:
        tracking = event.get("tracking") or event
        candidates = tracking.get("candidates") or []
        selected_index = tracking.get("selected_index")
        selection_reason = str(tracking.get("selection_reason", "unknown"))
        reasons[selection_reason] += 1
        target_changed = bool(tracking.get("target_changed"))
        is_initial_track = target_changed and selection_reason.startswith("initial_")
        switches += int(target_changed)
        initial_track_events += int(is_initial_track)
        post_initial_switches += int(target_changed and not is_initial_track)
        track_id = tracking.get("target_track_id")
        if track_id is not None:
            try:
                track_ids.append(int(track_id))
            except (TypeError, ValueError):
                pass
        valid_candidates = [
            item for item in candidates if _selection_depth(item) is not None
        ]
        if valid_candidates:
            depth_available_frames += 1
        if selected_index is None or not candidates:
            continue
        try:
            selected = candidates[int(selected_index)]
        except (IndexError, TypeError, ValueError):
            continue
        selected_frames += 1
        selected_depth = _selection_depth(selected)
        if selected_depth is not None:
            selected_depths.append(float(selected_depth))
            if valid_candidates:
                nearest_depth = min(
                    _selection_depth(item) for item in valid_candidates
                )
                depth_gaps.append(float(selected_depth) - nearest_depth)
                nearest_consistent += int(abs(float(selected_depth) - nearest_depth) <= 1e-6)
        fraction = selected.get("depth_valid_fraction")
        if fraction is not None:
            selected_valid_fractions.append(float(fraction))

    report: dict[str, Any] = {
        "schema": "nearest_person_sequence_eval.v1",
        "frames": len(event_list),
        "selected_frames": selected_frames,
        "depth_available_fraction": depth_available_frames / max(len(event_list), 1),
        "nearest_selection_consistency": (
            nearest_consistent / max(len(depth_gaps), 1) if depth_gaps else None
        ),
        "target_switches": switches,
        "initial_track_events": initial_track_events,
        "post_initial_target_switches": post_initial_switches,
        "unique_track_ids": sorted(set(track_ids)),
        "selection_reasons": dict(sorted(reasons.items())),
        "selected_depth_m": {
            "median": _percentile(selected_depths, 50),
            "p05": _percentile(selected_depths, 5),
            "p95": _percentile(selected_depths, 95),
        },
        "selected_depth_valid_fraction": {
            "median": _percentile(selected_valid_fractions, 50),
            "p05": _percentile(selected_valid_fractions, 5),
            "p95": _percentile(selected_valid_fractions, 95),
        },
        "selected_minus_nearest_depth_m": {
            "median": _percentile(depth_gaps, 50),
            "p95": _percentile(depth_gaps, 95),
            "max": max(depth_gaps) if depth_gaps else None,
        },
    }

    if region_records is not None:
        previous: dict[str, np.ndarray] = {}
        jitter_2d: defaultdict[str, list[float]] = defaultdict(list)
        jitter_3d: defaultdict[str, list[float]] = defaultdict(list)
        for record in region_records:
            for name, region in _regions_by_name(record.get("regions")).items():
                center_2d = region.get("center_2d")
                if center_2d is not None:
                    point = np.asarray(center_2d, dtype=np.float64).reshape(-1)
                    if point.shape == (2,) and np.all(np.isfinite(point)) and name in previous:
                        jitter_2d[name].append(float(np.linalg.norm(point - previous[name])))
                    if point.shape == (2,) and np.all(np.isfinite(point)):
                        previous[name] = point
                center_3d = region.get("center_3d")
                if center_3d is not None:
                    point_3d = np.asarray(center_3d, dtype=np.float64).reshape(-1)
                    previous_key = f"3d:{name}"
                    if point_3d.shape == (3,) and np.all(np.isfinite(point_3d)):
                        if previous_key in previous:
                            jitter_3d[name].append(
                                float(np.linalg.norm(point_3d - previous[previous_key]))
                            )
                        previous[previous_key] = point_3d
        report["region_center_jitter"] = {
            "2d_px": {
                name: {
                    "median": _percentile(values, 50),
                    "p95": _percentile(values, 95),
                    "samples": len(values),
                }
                for name, values in sorted(jitter_2d.items())
            },
            "3d_m": {
                name: {
                    "median": _percentile(values, 50),
                    "p95": _percentile(values, 95),
                    "samples": len(values),
                }
                for name, values in sorted(jitter_3d.items())
            },
        }
    return report
