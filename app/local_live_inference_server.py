#!/usr/bin/env python3
"""Local V12 body-region and V5 Gemini height inference service.

The service intentionally mirrors the remote HTTP contract used by the desktop
capture application. It accepts only one RGB-D request at a time so model
state, nearest-person tracking, and GPU memory remain bounded on the local
RTX 5060 workstation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, request


# Resolve paths from the cloned repository so the deployment package remains
# portable. The previous workstation-specific path is still valid when the
# repository is kept at D:\Gemini2L_LocalInference.
LOCAL_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = LOCAL_ROOT / "app"
PROJECT_ROOT = APP_ROOT / "project"
HEIGHT_ROOT = APP_ROOT / "height"
# Third-party OpenMMLab packages are intentionally isolated under the local
# deployment root so the system/Conda environments stay untouched.
VENDOR_ROOT = APP_ROOT / "vendor"
POSE_CONFIG = PROJECT_ROOT / "configs" / "rtmw_x_cocktail14_384x288.py"
POSE_CHECKPOINT = (
    LOCAL_ROOT
    / "models"
    / "pose"
    / "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
)
DETECTOR_CONFIG = PROJECT_ROOT / "configs" / "rtmdet_nano_320-8xb32_coco-person.py"
DETECTOR_CHECKPOINT = LOCAL_ROOT / "models" / "detector" / "rtmdet-nano-person.pth"
DEFAULT_BODY_CHECKPOINT = LOCAL_ROOT / "models" / "body_regions" / "v12" / "best.pt"
SERVICE_SCHEMA = "gemini_local_inference.v1"
HEIGHT_ESTIMATOR = "gemini_human_height_estimate.v5"


def _prepare_import_paths() -> None:
    for path in (VENDOR_ROOT, PROJECT_ROOT, HEIGHT_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_response(payload: dict[str, Any], status: int = 200) -> Response:
    return Response(
        json.dumps(payload, ensure_ascii=True, default=_json_default),
        status=status,
        mimetype="application/json",
    )


def _load_pose_model(config: Path, checkpoint: Path, device: str) -> tuple[Any, str, dict[str, Any]]:
    from body_model.mmpose_compat import (
        prepare_mmpose_import,
        prepare_torch_checkpoint_loading,
    )

    compatibility = prepare_mmpose_import()
    compatibility.update(prepare_torch_checkpoint_loading())
    import torch
    from mmpose.apis import init_model

    actual_device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    previous_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        model = init_model(str(config), str(checkpoint), device=actual_device)
    finally:
        os.chdir(previous_cwd)
    return model, actual_device, compatibility


class Runtime:
    def __init__(self, body_checkpoint: Path, detector_every: int, device: str) -> None:
        _prepare_import_paths()
        self._validate_assets(body_checkpoint)

        from body_model.inference import RGBDBodyRegionInferencer
        from body_model.live_pipeline import LiveBodyRegionPipeline
        from body_model.person_tracking import NearestPersonTracker
        from body_model.pose_frontend import build_pose_frontend
        from body_model.temporal_regions import TemporalRegionFilter
        from work.human_height_estimation_v5 import estimate_human_height

        pose_model, pose_device, compatibility = _load_pose_model(
            POSE_CONFIG,
            POSE_CHECKPOINT,
            device,
        )
        self.pose_frontend = build_pose_frontend(
            "rtmw-x-topdown",
            pose_model=pose_model,
            device=pose_device,
            pose_config=str(POSE_CONFIG),
            pose_checkpoint=str(POSE_CHECKPOINT),
            # The installed MMPose package supplies config resources. This
            # directory check only supports the existing pose adapter API.
            mmpose_root=str(PROJECT_ROOT),
            detector_config=str(DETECTOR_CONFIG),
            detector_checkpoint=str(DETECTOR_CHECKPOINT),
            detector_threshold=0.30,
            detector_interval=max(int(detector_every), 1),
            compatibility=compatibility,
        )
        learned_inferencer = RGBDBodyRegionInferencer(
            body_checkpoint,
            device=pose_device,
            amp=False,
        )
        self.pipeline = LiveBodyRegionPipeline(
            tracker=NearestPersonTracker(
                iou_threshold=0.25,
                switch_margin_m=0.20,
                switch_confirm_frames=3,
                max_missing_frames=5,
                selection_depth_percentile=50.0,
            ),
            keypoint_threshold=0.25,
            body_keypoint_count=17,
            min_keypoints=6,
            geometry_stride=5,
            crop_padding=0.12,
            include_region_geometry=True,
            learned_inferencer=learned_inferencer,
            region_filter=TemporalRegionFilter(
                tau_seconds=0.12,
                max_jump_m=0.75,
                metric_max_speed_m_s=0.8,
                metric_noise_floor_m=0.03,
                sparse_metric_max_jump_m=0.06,
                max_hold_frames=3,
                use_velocity=False,
                learned_only=True,
            ),
            profile_timing=True,
        )
        self.estimate_human_height = estimate_human_height
        self.body_checkpoint = str(body_checkpoint)
        self.pose_device = pose_device
        self.lock = threading.Lock()
        self.request_count = 0
        self.started_at = time.time()
        self.tracking_context: str | None = None
        self.last_height_diagnostic: dict[str, Any] | None = None

    @staticmethod
    def _validate_assets(body_checkpoint: Path) -> None:
        missing = [
            path
            for path in (
                body_checkpoint,
                POSE_CONFIG,
                POSE_CHECKPOINT,
                DETECTOR_CONFIG,
                DETECTOR_CHECKPOINT,
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing local inference asset(s): " + ", ".join(map(str, missing))
            )

    def infer(
        self,
        *,
        color_bgr: np.ndarray,
        depth_raw: np.ndarray,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        import torch

        frame_index = int(metadata["frame_index"])
        started = time.perf_counter()
        with self.lock, torch.inference_mode():
            tracking_context = str(
                metadata.get("tracking_context") or metadata.get("session_id") or "default"
            )
            if tracking_context != self.tracking_context:
                self.pose_frontend.reset()
                self.pipeline.reset()
                self.tracking_context = tracking_context

            pose_started = time.perf_counter()
            people = self.pose_frontend.infer(
                color_bgr,
                pose_threshold=0.30,
                keypoint_threshold=0.125,
                frame_index=frame_index,
            )
            pose_ms = (time.perf_counter() - pose_started) * 1000.0

            pipeline_started = time.perf_counter()
            result = self.pipeline.process(
                frame_index=frame_index,
                rgb=cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB),
                depth=depth_raw,
                depth_scale=float(metadata["depth_scale_m"]),
                intrinsics=metadata["intrinsics"],
                people=people,
                timestamp=float(metadata.get("timestamp", time.monotonic())),
                frame_metadata={
                    "capture_wall_time_ns": metadata.get("capture_wall_time_ns"),
                    "local_request_id": metadata.get("request_id"),
                    "posture": metadata.get("posture"),
                    "view": metadata.get("view"),
                },
            )
            # The desktop client can request deferred V5 height processing so
            # the expensive CPU geometry pass cannot delay the V12 overlay.
            # Generic API users keep the synchronous, self-contained result.
            defer_height = bool(metadata.get("defer_height_estimation", False))
            height_ms = 0.0
            height: dict[str, Any] | None = None
            if not defer_height:
                height_started = time.perf_counter()
                height = self.estimate_human_height(
                    depth_raw=depth_raw,
                    depth_scale_m=float(metadata["depth_scale_m"]),
                    intrinsics=metadata["intrinsics"],
                    result=result,
                    posture=metadata.get("posture"),
                    color_bgr=color_bgr,
                    imu=metadata.get("imu"),
                    depth_correction=metadata.get("height_depth_correction"),
                )
                height.update(
                    {
                        "algorithm": HEIGHT_ESTIMATOR,
                        "estimator_scope": "local_service",
                        "source_frame_index": frame_index,
                    }
                )
                result["height_estimate"] = height
                result["height_estimator"] = HEIGHT_ESTIMATOR
                result["height_estimate_source"] = "local_service_v5"
                height_ms = (time.perf_counter() - height_started) * 1000.0
            else:
                result["height_estimator"] = HEIGHT_ESTIMATOR
                result["height_estimate_source"] = "client_local_v5_pending"
            pipeline_ms = (time.perf_counter() - pipeline_started) * 1000.0
            self.request_count += 1
            if height is not None:
                self.last_height_diagnostic = {
                    "request_id": metadata.get("request_id"),
                    "frame_index": frame_index,
                    "captured_at": metadata.get("timestamp"),
                    "processed_at": time.time(),
                    "height_estimate": height,
                    "image_retained": False,
                }

        result["local_runtime"] = {
            "schema": SERVICE_SCHEMA,
            "service_scope": "local",
            "request_id": metadata.get("request_id"),
            "body_checkpoint": self.body_checkpoint,
            "pose_backend": self.pose_frontend.label,
            "pose_device": self.pose_device,
            "pose_ms": float(pose_ms),
            "body_pipeline_ms": float(pipeline_ms),
            "height_estimation_ms": float(height_ms),
            "height_deferred_to_client": bool(defer_height),
            "server_total_ms": float((time.perf_counter() - started) * 1000.0),
            "request_count": int(self.request_count),
            "tracking_context": self.tracking_context,
            "image_retained": False,
            "clinical_use": False,
            "requires_human_review": True,
        }
        return result


def build_app(runtime: Runtime) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health() -> Response:
        return _json_response(
            {
                "status": "ok",
                "service_scope": "local",
                "model": "v12-local",
                "body_checkpoint": runtime.body_checkpoint,
                "pose_backend": runtime.pose_frontend.label,
                "pose_device": runtime.pose_device,
                "height_estimator": HEIGHT_ESTIMATOR,
                "height_measurement_source": "gemini_aligned_depth_and_intrinsics",
                "height_depth_correction_schema": "gemini_height_depth_correction_request.v1",
                "height_depth_correction_scope": "height_estimator_only",
                "manual_reference_height_required": False,
                "request_count": runtime.request_count,
                "uptime_seconds": time.time() - runtime.started_at,
                "clinical_use": False,
                "requires_human_review": True,
            }
        )

    @app.get("/debug/last-height")
    def last_height() -> Response:
        with runtime.lock:
            diagnostic = runtime.last_height_diagnostic
            return _json_response(
                {
                    "status": "ok" if diagnostic is not None else "no_height_estimate_yet",
                    "last_height": diagnostic,
                    "image_retained": False,
                    "clinical_use": False,
                    "requires_human_review": True,
                }
            )

    @app.post("/infer")
    def infer() -> Response:
        try:
            metadata = json.loads(request.form["metadata"])
            color_bytes = np.frombuffer(request.files["rgb"].read(), dtype=np.uint8)
            depth_bytes = np.frombuffer(request.files["depth"].read(), dtype=np.uint8)
            color_bgr = cv2.imdecode(color_bytes, cv2.IMREAD_COLOR)
            depth_raw = cv2.imdecode(depth_bytes, cv2.IMREAD_UNCHANGED)
            if color_bgr is None or depth_raw is None:
                raise ValueError("failed to decode RGB or depth payload")
            if depth_raw.dtype != np.uint16 or depth_raw.ndim != 2:
                raise ValueError(
                    f"depth must be uint16 HxW, got {depth_raw.dtype} {depth_raw.shape}"
                )
            if color_bgr.shape[:2] != depth_raw.shape:
                raise ValueError("RGB and depth sizes do not match")
            return _json_response(
                runtime.infer(
                    color_bgr=color_bgr,
                    depth_raw=depth_raw,
                    metadata=metadata,
                )
            )
        except Exception as error:
            return _json_response(
                {"status": "server_error", "error": f"{type(error).__name__}: {error}"},
                status=500,
            )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Gemini V12 + V5 inference")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument("--body-checkpoint", type=Path, default=DEFAULT_BODY_CHECKPOINT)
    parser.add_argument("--detector-every", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    runtime = Runtime(args.body_checkpoint, args.detector_every, args.device)
    build_app(runtime).run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
