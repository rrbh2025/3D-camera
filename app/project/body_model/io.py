from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_keypoints(path: str | Path) -> Any:
    """Load a JSON payload containing one COCO-17 person."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_keypoints(payload: Any) -> Any:
    """Extract the first person's keypoints from common JSON layouts."""
    if isinstance(payload, dict):
        if "keypoints" in payload:
            return payload["keypoints"]
        if "people" in payload and payload["people"]:
            return extract_keypoints(payload["people"][0])
        if "predictions" in payload and payload["predictions"]:
            return extract_keypoints(payload["predictions"][0])
        if "pred_instances" in payload:
            return extract_keypoints(payload["pred_instances"])
        if "keypoints_xy" in payload:
            return payload["keypoints_xy"]
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            return extract_keypoints(payload[0])
        return payload
    raise ValueError("Could not find keypoints in the input JSON")
