from __future__ import annotations

from pathlib import Path

import numpy as np


class RGBDBodyRegionInferencer:
    """Load one checkpoint once and run it on RGB-D dataset samples."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cuda:0",
        width: int | None = None,
        image_size: tuple[int, int] | None = None,
        amp: bool = False,
    ) -> None:
        import torch

        from .rgbd_model import RGBDBodyRegionNet

        requested = str(device)
        self.device = torch.device(
            requested
            if requested.startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        self.amp = bool(amp and self.device.type == "cuda")
        self.checkpoint_path = Path(checkpoint)
        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        checkpoint_args = self.checkpoint.get("args", {})
        self.width = int(width or checkpoint_args.get("width", 32))
        checkpoint_size = checkpoint_args.get("image_size", [256, 256])
        chosen_size = image_size or checkpoint_size
        self.image_size = (int(chosen_size[0]), int(chosen_size[1]))
        self.model = RGBDBodyRegionNet(
            width=self.width,
            use_metric_keypoints=bool(checkpoint_args.get("metric_keypoints", False)),
            use_metric_geometry=bool(checkpoint_args.get("metric_geometry", False)),
            use_prior_residual=bool(checkpoint_args.get("prior_residual", False)),
            use_prior_residual_depth_only=bool(
                checkpoint_args.get("prior_residual_depth_only", False)
            ),
            use_metric_region_prior=bool(
                checkpoint_args.get("metric_region_prior", False)
            ),
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint.get("model", self.checkpoint))
        self.model.eval()

    def predict(self, sample: dict) -> dict:
        import torch

        def tensor(name: str):
            return torch.from_numpy(sample[name]).unsqueeze(0).to(self.device)

        metric_available = torch.from_numpy(
            np.asarray([sample["metric_available"]], dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        metric_region_centers = torch.from_numpy(
            np.asarray(sample["metric_region_centers"], dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        metric_region_available = torch.from_numpy(
            np.asarray(sample["metric_region_available"], dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp,
        ):
            return self.model(
                tensor("rgbd"),
                tensor("keypoints"),
                tensor("modality_mask"),
                tensor("camera_intrinsics"),
                tensor("metric_keypoints"),
                tensor("metric_context"),
                metric_available,
                metric_region_centers,
                metric_region_available,
            )

    def report(self, sample: dict, prediction: dict) -> tuple[dict, np.ndarray]:
        import torch

        from .regions import REGION_NAMES

        crop = sample["meta"].get("crop_xyxy")
        crop_applied = bool(sample["meta"].get("crop_applied", False))
        offset_x = float(crop[0]) if crop_applied else 0.0
        offset_y = float(crop[1]) if crop_applied else 0.0
        width_px = float(sample["meta"]["width"])
        height_px = float(sample["meta"]["height"])
        center = prediction["center_2d"][0].cpu().numpy()
        heatmap_center = prediction["heatmap_center_2d"][0].cpu().numpy()
        boxes = prediction["bbox_2d"][0].cpu().numpy()
        center_3d = prediction["center_3d"][0].cpu().numpy()
        center_3d_raw = prediction["center_3d_raw"][0].cpu().numpy()
        local_geometry = prediction.get("region_geometry")
        if local_geometry is None:
            local_geometry = np.zeros(
                (1, len(REGION_NAMES), 8), dtype=np.float32
            )
        else:
            local_geometry = local_geometry.cpu().numpy()
        local_geometry_valid = prediction.get("region_geometry_valid")
        if local_geometry_valid is None:
            local_geometry_valid = np.zeros(
                (1, len(REGION_NAMES)), dtype=bool
            )
        else:
            local_geometry_valid = local_geometry_valid.cpu().numpy()
        metric_region_centers = np.asarray(
            sample.get(
                "metric_region_centers",
                np.zeros_like(center_3d, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        metric_region_available = np.asarray(
            sample.get(
                "metric_region_available",
                np.zeros((len(REGION_NAMES),), dtype=np.float32),
            ),
            dtype=np.float32,
        ).reshape(-1)
        metric_geometry_used = np.zeros((len(REGION_NAMES),), dtype=bool)
        if (
            metric_region_centers.shape == center_3d.shape
            and metric_region_available.shape == (len(REGION_NAMES),)
        ):
            metric_geometry_used = metric_region_available > 0.5
        reported_center_3d = center_3d.copy()
        reported_center_3d[metric_geometry_used] = metric_region_centers[
            metric_geometry_used
        ]
        center_depth_m = prediction["center_depth_m"][0].cpu().numpy()
        center_depth_valid = prediction["center_depth_valid"][0].cpu().numpy()
        center_depth_used = prediction["center_depth_used"][0].cpu().numpy()
        visible = torch.sigmoid(prediction["visible_logits"][0]).cpu().numpy()
        confidence = torch.sigmoid(prediction["confidence_logits"][0]).cpu().numpy()
        presence = torch.sigmoid(
            prediction["region_presence_logits"][0]
        ).cpu().numpy()
        regions = []
        for index, name in enumerate(REGION_NAMES):
            regions.append(
                {
                    "name": name,
                    "center_2d": [
                        float(center[index, 0] * width_px + offset_x),
                        float(center[index, 1] * height_px + offset_y),
                    ],
                    "heatmap_center_2d": [
                        float(heatmap_center[index, 0] * width_px + offset_x),
                        float(heatmap_center[index, 1] * height_px + offset_y),
                    ],
                    "bbox_2d": [
                        float(boxes[index, 0] * width_px + offset_x),
                        float(boxes[index, 1] * height_px + offset_y),
                        float(boxes[index, 2] * width_px + offset_x),
                        float(boxes[index, 3] * height_px + offset_y),
                    ],
                    "center_3d_camera": reported_center_3d[index].tolist(),
                    "center_3d_metric_geometry": metric_region_centers[
                        index
                    ].tolist(),
                    "center_3d_source": (
                        "metric_keypoint_geometry"
                        if metric_geometry_used[index]
                        else "network_depth_or_raw"
                    ),
                    "center_3d_raw": center_3d_raw[index].tolist(),
                    "local_metric_geometry": local_geometry[0][index].tolist(),
                    "local_metric_geometry_valid": bool(
                        local_geometry_valid[0][index]
                    ),
                    "sampled_aligned_depth_m": (
                        float(center_depth_m[index])
                        if center_depth_valid[index]
                        else None
                    ),
                    "used_aligned_depth": bool(center_depth_used[index]),
                    "visible_probability": float(visible[index]),
                    "confidence": float(confidence[index]),
                    "presence_probability": float(presence[index]),
                }
            )
        heatmaps = torch.sigmoid(
            prediction["region_heatmap_logits"][0]
        ).cpu().numpy()
        report = {
            "schema": "body_regions_rgbd_inference.v2",
            "record_id": sample["meta"]["id"],
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_epoch": self.checkpoint.get("epoch"),
            "device": str(self.device),
            "amp": bool(self.amp),
            "modalities": {
                "rgb": bool(sample["modality_mask"][0]),
                "depth": bool(sample["modality_mask"][1]),
            },
            "center_3d_calibrated": bool(sample["camera_intrinsics"][4] > 0.5),
            "camera_intrinsics_applied": bool(sample["camera_intrinsics"][4] > 0.5),
            "metric_region_geometry_used": int(metric_geometry_used.sum()),
            "metric_region_geometry_available_fraction": float(
                metric_region_available.mean()
                if metric_region_available.size
                else 0.0
            ),
            "center_3d_depth_source": (
                "metric_keypoint_geometry_with_network_fallback"
                if bool(metric_geometry_used.any())
                else "aligned_depth_with_network_fallback"
            ),
            "local_metric_geometry_available_fraction": float(
                local_geometry_valid[0].mean()
                if local_geometry_valid.size
                else 0.0
            ),
            "regions": regions,
        }
        return report, heatmaps.astype(np.float32)

    def infer(self, sample: dict, heatmaps_output: str | Path | None = None) -> dict:
        report, heatmaps = self.report(sample, self.predict(sample))
        if heatmaps_output is not None:
            output = Path(heatmaps_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            from .regions import REGION_NAMES

            np.savez_compressed(
                output,
                heatmaps=heatmaps,
                region_names=np.asarray(REGION_NAMES),
            )
        return report


class DualPathRGBDBodyRegionInferencer:
    """Route metric regions through a primary model and missing regions through a fallback.

    The fallback checkpoint is evaluated with all recorded 3D inputs masked,
    so it remains an independent RGB/2D/depth path instead of silently
    reusing a ground-truth-derived metric prior.
    """

    def __init__(
        self,
        primary_checkpoint: str | Path,
        fallback_checkpoint: str | Path,
        *,
        device: str = "cuda:0",
        image_size: tuple[int, int] | None = None,
        amp: bool = False,
    ) -> None:
        self.primary = RGBDBodyRegionInferencer(
            primary_checkpoint,
            device=device,
            image_size=image_size,
            amp=amp,
        )
        self.fallback = RGBDBodyRegionInferencer(
            fallback_checkpoint,
            device=device,
            image_size=image_size or self.primary.image_size,
            amp=amp,
        )
        if self.primary.image_size != self.fallback.image_size:
            raise ValueError(
                "primary and fallback checkpoints must use the same image size"
            )
        self.image_size = self.primary.image_size
        self.device = self.primary.device
        self.primary_checkpoint = self.primary.checkpoint_path
        self.fallback_checkpoint = self.fallback.checkpoint_path

    @staticmethod
    def _mask_metric_inputs(sample: dict) -> dict:
        masked = dict(sample)
        for name in (
            "metric_keypoints",
            "metric_context",
            "metric_region_centers",
            "metric_region_available",
        ):
            if name in sample:
                masked[name] = np.zeros_like(np.asarray(sample[name]))
        if "metric_available" in sample:
            masked["metric_available"] = np.asarray(0.0, dtype=np.float32)
        return masked

    @staticmethod
    def _finite(value: object, size: int) -> bool:
        try:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return False
        return array.shape == (size,) and bool(np.all(np.isfinite(array)))

    def infer(self, sample: dict, heatmaps_output: str | Path | None = None) -> dict:
        from .regions import REGION_NAMES

        primary_report = self.primary.infer(sample, heatmaps_output)
        metric_available = np.asarray(
            sample.get(
                "metric_region_available",
                np.zeros((len(REGION_NAMES),), dtype=np.float32),
            ),
            dtype=np.float32,
        ).reshape(-1)
        missing = np.zeros((len(REGION_NAMES),), dtype=bool)
        if metric_available.shape == (len(REGION_NAMES),):
            missing = metric_available <= 0.5
        fallback_report = None
        used_names: list[str] = []
        if bool(missing.any()):
            fallback_report = self.fallback.infer(
                self._mask_metric_inputs(sample)
            )
            primary_by_name = {
                str(region.get("name")): region
                for region in primary_report.get("regions", [])
            }
            fallback_by_name = {
                str(region.get("name")): region
                for region in fallback_report.get("regions", [])
            }
            for index, name in enumerate(REGION_NAMES):
                if not missing[index]:
                    continue
                primary_region = primary_by_name.get(name)
                fallback_region = fallback_by_name.get(name)
                if primary_region is None or fallback_region is None:
                    continue
                if not bool(fallback_region.get("used_aligned_depth")):
                    continue
                if not self._finite(
                    fallback_region.get("center_3d_camera"), 3
                ):
                    continue
                for key in (
                    "center_3d_camera",
                    "center_3d_raw",
                    "center_3d_metric_geometry",
                    "local_metric_geometry",
                    "local_metric_geometry_valid",
                    "sampled_aligned_depth_m",
                    "used_aligned_depth",
                ):
                    if key in fallback_region:
                        primary_region[key] = fallback_region[key]
                primary_region["center_3d_source"] = (
                    "fallback_checkpoint_aligned_depth"
                )
                primary_region["fallback_checkpoint_used"] = True
                used_names.append(name)

        routing = {
            "schema": "body_regions_dual_path_routing.v1",
            "primary_checkpoint": str(self.primary_checkpoint),
            "fallback_checkpoint": str(self.fallback_checkpoint),
            "missing_metric_region_count": int(missing.sum()),
            "fallback_region_count": int(len(used_names)),
            "fallback_regions": used_names,
            "fallback_metric_inputs_masked": bool(fallback_report is not None),
        }
        primary_report["routing"] = routing
        if fallback_report is not None:
            primary_report["fallback_inference"] = {
                "checkpoint": fallback_report.get("checkpoint"),
                "checkpoint_epoch": fallback_report.get("checkpoint_epoch"),
                "device": fallback_report.get("device"),
            }
        return primary_report
