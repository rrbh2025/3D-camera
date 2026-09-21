from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .regions import REGION_NAMES


REVIEW_DECISIONS = {"accept", "correct", "reject"}
TRAIN_REVIEW_STATUSES = {"accepted", "corrected"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {source}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record must be an object at {source}:{line_number}")
        records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_record_path(manifest_path: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else manifest_path.parent / path


def _as_regions(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items() if isinstance(item, dict)}
    if isinstance(value, list):
        return {
            str(item.get("name")): item
            for item in value
            if isinstance(item, dict) and item.get("name")
        }
    return {}


def _keypoint_quality(value: Any) -> tuple[int, int]:
    try:
        points = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return 0, 0
    if points.shape != (17, 3):
        return 0, 0
    finite = np.isfinite(points).all(axis=1)
    visible = finite & (points[:, 2] > 0.0)
    return int(visible.sum()), int(finite.sum())


def _quality_summary(record: dict[str, Any]) -> dict[str, Any]:
    visible_keypoints, finite_keypoints = _keypoint_quality(record.get("keypoints"))
    regions = _as_regions(record.get("regions"))
    visible_region_items = [
        item for item in regions.values() if bool(item.get("visible"))
    ]
    confidences = []
    for item in visible_region_items:
        try:
            value = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            confidences.append(value)
    target = (record.get("target_tracking") or {}).get("selected_person") or {}
    depth_fraction = target.get("depth_valid_fraction")
    depth_m = target.get("depth_m")
    try:
        depth_fraction = float(depth_fraction)
    except (TypeError, ValueError):
        depth_fraction = None
    try:
        depth_m = float(depth_m)
    except (TypeError, ValueError):
        depth_m = None
    depth_iqrs = []
    for item in visible_region_items:
        geometry = item.get("geometry_3d") or {}
        try:
            value = float(geometry.get("depth_iqr"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            depth_iqrs.append(value)
    return {
        "visible_keypoints": visible_keypoints,
        "finite_keypoints": finite_keypoints,
        "visible_regions": len(visible_region_items),
        "mean_region_confidence": (
            float(np.mean(confidences)) if confidences else 0.0
        ),
        "max_region_depth_iqr_m": max(depth_iqrs, default=None),
        "depth_valid_fraction": depth_fraction,
        "selected_depth_m": depth_m,
    }


def _is_smoke_record(record: dict[str, Any]) -> bool:
    values = (
        str(record.get("subject_id", "")),
        str(record.get("session_id", "")),
        str(record.get("id", "")),
    )
    return any("smoke" in value.lower() for value in values)


def _frame_index(record: dict[str, Any], fallback: int) -> int:
    timing = record.get("frame_timing") or {}
    streams = timing.get("streams") or {}
    value = (streams.get("rgb") or {}).get("frame_index")
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def validate_body_region_record(
    record: dict[str, Any],
    manifest_path: str | Path,
    *,
    stage: str,
    min_keypoints: int = 6,
    min_depth_valid_fraction: float = 0.40,
    min_region_confidence: float = 0.35,
    max_region_depth_iqr_m: float = 0.50,
    allow_smoke: bool = False,
    reject_target_changes: bool = True,
    require_metric_geometry: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """Return quality-gate failures and a small auditable quality summary."""
    if stage not in {"review", "train"}:
        raise ValueError("stage must be 'review' or 'train'")
    manifest = Path(manifest_path).resolve()
    reasons: list[str] = []
    if record.get("schema") != "body_regions_rgbd_manifest.v1":
        reasons.append("unsupported_schema")
    for field in ("id", "subject_id", "session_id", "device_id"):
        if not str(record.get(field, "")).strip():
            reasons.append(f"missing_{field}")
    if not allow_smoke and _is_smoke_record(record):
        reasons.append("smoke_record_requires_explicit_opt_in")
    for field in ("rgb", "depth"):
        path = _resolve_record_path(manifest, record.get(field))
        if path is None or not path.is_file():
            reasons.append(f"missing_{field}_file")

    visible_keypoints, _ = _keypoint_quality(record.get("keypoints"))
    if visible_keypoints < int(min_keypoints):
        reasons.append("insufficient_keypoints")
    regions = _as_regions(record.get("regions"))
    missing_regions = [name for name in REGION_NAMES if name not in regions]
    if missing_regions:
        reasons.append("missing_regions")
    quality = _quality_summary(record)
    if quality["visible_regions"] == 0:
        reasons.append("no_visible_regions")
    if quality["mean_region_confidence"] < float(min_region_confidence):
        reasons.append("low_region_confidence")
    if quality["depth_valid_fraction"] is None:
        reasons.append("missing_selected_depth_quality")
    elif quality["depth_valid_fraction"] < float(min_depth_valid_fraction):
        reasons.append("low_depth_valid_fraction")
    if (
        quality["max_region_depth_iqr_m"] is not None
        and quality["max_region_depth_iqr_m"] > float(max_region_depth_iqr_m)
    ):
        reasons.append("unstable_region_depth")

    tracking = record.get("target_tracking") or {}
    selected_person = tracking.get("selected_person") or {}
    if not selected_person:
        reasons.append("missing_selected_person")
    selection_reason = str(tracking.get("selection_reason", ""))
    initial_acquisition = selection_reason.startswith("initial_")
    if (
        reject_target_changes
        and bool(tracking.get("target_changed"))
        and not initial_acquisition
    ):
        reasons.append("target_changed")
    if require_metric_geometry:
        geometry_missing = [
            name
            for name in REGION_NAMES
            if regions.get(name, {}).get("center_3d") is None
        ]
        if geometry_missing:
            reasons.append("missing_metric_geometry")

    annotation_status = str(record.get("annotation_status", ""))
    review_status = str(record.get("review_status", ""))
    if stage == "review":
        if annotation_status != "pseudo_labeled":
            reasons.append("not_pseudo_labeled")
        if review_status not in {"", "pending_human_review"}:
            reasons.append("already_reviewed")
    else:
        if review_status not in TRAIN_REVIEW_STATUSES:
            reasons.append("not_human_reviewed")
        if annotation_status not in {"reviewed", "corrected"}:
            reasons.append("annotation_not_reviewed")

    quality["frame_index"] = _frame_index(record, 0)
    return sorted(set(reasons)), quality


def curate_manifests(
    manifests: Iterable[str | Path],
    *,
    stage: str,
    frame_stride: int = 1,
    max_records_per_session: int = 0,
    min_keypoints: int = 6,
    min_depth_valid_fraction: float = 0.40,
    min_region_confidence: float = 0.35,
    max_region_depth_iqr_m: float = 0.50,
    allow_smoke: bool = False,
    reject_target_changes: bool = True,
    require_metric_geometry: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Curate pseudo/reviewed manifests without mixing adjacent frames in splits."""
    stride = max(int(frame_stride), 1)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    valid_session_counts: defaultdict[str, int] = defaultdict(int)
    selected_session_counts: defaultdict[str, int] = defaultdict(int)
    reason_counts: Counter[str] = Counter()
    input_records = 0
    source_manifests = [str(Path(path).resolve()) for path in manifests]

    for manifest_string in source_manifests:
        manifest_path = Path(manifest_string)
        for source_line, record in enumerate(read_jsonl(manifest_path), start=1):
            input_records += 1
            record_id = str(record.get("id", ""))
            if record_id and record_id in seen_ids:
                rejected_record = copy.deepcopy(record)
                rejected_record["curation"] = {
                    "status": "rejected",
                    "reasons": ["duplicate_id"],
                    "source_manifest": str(manifest_path),
                    "source_line": source_line,
                }
                rejected.append(rejected_record)
                reason_counts["duplicate_id"] += 1
                continue
            if record_id:
                seen_ids.add(record_id)
            reasons, quality = validate_body_region_record(
                record,
                manifest_path,
                stage=stage,
                min_keypoints=min_keypoints,
                min_depth_valid_fraction=min_depth_valid_fraction,
                min_region_confidence=min_region_confidence,
                max_region_depth_iqr_m=max_region_depth_iqr_m,
                allow_smoke=allow_smoke,
                reject_target_changes=reject_target_changes,
                require_metric_geometry=require_metric_geometry,
            )
            session_id = str(record.get("session_id", record.get("id", "unknown")))
            if reasons:
                rejected_record = copy.deepcopy(record)
                rejected_record["curation"] = {
                    "status": "rejected",
                    "reasons": reasons,
                    "quality": quality,
                    "source_manifest": str(manifest_path),
                    "source_line": source_line,
                }
                rejected.append(rejected_record)
                reason_counts.update(reasons)
                continue
            valid_session_index = valid_session_counts[session_id]
            valid_session_counts[session_id] += 1
            if valid_session_index % stride != 0:
                reason_counts["frame_stride"] += 1
                continue
            if (
                max_records_per_session > 0
                and selected_session_counts[session_id] >= max_records_per_session
            ):
                reason_counts["max_records_per_session"] += 1
                continue
            selected_session_counts[session_id] += 1
            curated = copy.deepcopy(record)
            curated["curation"] = {
                "status": "pending_review" if stage == "review" else "ready_for_training",
                "quality": quality,
                "source_manifest": str(manifest_path),
                "source_line": source_line,
                "split_unit": "subject_id",
                "frame_stride": stride,
            }
            candidates.append(curated)

    subjects = sorted({str(record.get("subject_id")) for record in candidates})
    sessions = sorted({str(record.get("session_id")) for record in candidates})
    summary = {
        "schema": "gemini_body_dataset_curation.v1",
        "stage": stage,
        "source_manifests": source_manifests,
        "input_records": input_records,
        "selected_records": len(candidates),
        "rejected_records": len(rejected),
        "subjects": subjects,
        "sessions": sessions,
        "reason_counts": dict(sorted(reason_counts.items())),
        "split_unit": "subject_id",
        "leakage_guard": "adjacent frames from one subject must not cross splits",
    }
    return candidates, rejected, summary


def apply_review_decisions(
    records: Iterable[dict[str, Any]],
    decisions: Iterable[dict[str, Any]],
    *,
    reviewer: str,
    reviewed_at: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply auditable accept/correct/reject decisions to pseudo-label records."""
    reviewer = str(reviewer).strip()
    if not reviewer:
        raise ValueError("reviewer must not be empty")
    timestamp = reviewed_at or datetime.now(timezone.utc).isoformat()
    decision_map: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        record_id = str(decision.get("id", "")).strip()
        action = str(decision.get("decision", "")).strip().lower()
        if not record_id:
            raise ValueError("each decision requires an id")
        if action not in REVIEW_DECISIONS:
            raise ValueError(f"unsupported review decision: {action!r}")
        if record_id in decision_map:
            raise ValueError(f"duplicate review decision for {record_id}")
        decision_map[record_id] = decision

    output: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    record_ids: set[str] = set()
    for record in records:
        updated = copy.deepcopy(record)
        record_id = str(updated.get("id", ""))
        record_ids.add(record_id)
        decision = decision_map.get(record_id)
        if decision is None:
            output.append(updated)
            counts["unreviewed"] += 1
            continue
        action = str(decision["decision"]).strip().lower()
        if action == "accept":
            updated["review_status"] = "accepted"
            updated["annotation_status"] = "reviewed"
        elif action == "correct":
            updated["review_status"] = "corrected"
            updated["annotation_status"] = "corrected"
            if "regions" in decision:
                updated["regions"] = decision["regions"]
        else:
            updated["review_status"] = "rejected"
            updated["annotation_status"] = "reviewed"
        updated["review"] = {
            "reviewer": reviewer,
            "reviewed_at": timestamp,
            "decision": action,
            "notes": str(decision.get("notes", "")),
        }
        counts[action] += 1
        output.append(updated)

    unknown_ids = sorted(set(decision_map) - record_ids)
    if unknown_ids:
        raise ValueError(f"review decisions refer to unknown ids: {unknown_ids[:3]}")
    return output, dict(sorted(counts.items()))
