from __future__ import annotations

from collections import deque
from typing import Any, Iterable

import numpy as np

from .temporal_dataset import _frame_arrays
from .temporal_refiner import TemporalRegionRefiner
from .regions import REGION_NAMES


class TemporalRegionRuntime:
    """Apply a trained causal refiner to one live target track."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cuda:0",
        learned_only: bool = True,
    ) -> None:
        import torch

        self.device = torch.device(
            device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        checkpoint_data = torch.load(
            checkpoint, map_location=self.device, weights_only=False
        )
        args = checkpoint_data.get("args", {})
        self.sequence_length = max(int(args.get("sequence_length", 9)), 2)
        self.model = TemporalRegionRefiner(
            hidden_dim=int(args.get("hidden_dim", 192)),
            layers=int(args.get("layers", 3)),
            heads=int(args.get("heads", 6)),
            max_length=self.sequence_length,
        ).to(self.device)
        self.model.load_state_dict(checkpoint_data["model"])
        self.model.eval()
        self.learned_only = bool(learned_only)
        self._frames: deque[dict[str, np.ndarray]] = deque(
            maxlen=self.sequence_length
        )

    def reset(self) -> None:
        self._frames.clear()

    @staticmethod
    def _is_learned(region: dict[str, Any]) -> bool:
        return str(region.get("region_source", "")).startswith("learned_") or (
            str(region.get("center_2d_source", "")).startswith("learned_")
            or str(region.get("center_3d_source", "")).startswith("learned_")
        )

    def update(
        self,
        *,
        keypoints: Iterable,
        regions: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        import torch

        record = {"keypoints": np.asarray(keypoints, dtype=np.float32).tolist(), "regions": regions}
        frame = _frame_arrays(
            record,
            training=False,
            observation_noise_2d=0.0,
            observation_noise_3d=0.0,
            observation_dropout=0.0,
            rng=np.random.default_rng(0),
        )
        self._frames.append(frame)
        features = torch.from_numpy(
            np.stack([item["features"] for item in self._frames])
        ).unsqueeze(0).to(self.device)
        observed_center_2d = torch.from_numpy(
            np.stack([item["observed_center_2d"] for item in self._frames])
        ).unsqueeze(0).to(self.device)
        observed_valid_2d = torch.from_numpy(
            np.stack([item["observed_valid_2d"] for item in self._frames])
        ).unsqueeze(0).to(self.device)
        observed_center_3d = torch.from_numpy(
            np.stack([item["observed_center_3d"] for item in self._frames])
        ).unsqueeze(0).to(self.device)
        observed_valid_3d = torch.from_numpy(
            np.stack([item["observed_valid_3d"] for item in self._frames])
        ).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            prediction = self.model(
                features,
                observed_center_2d,
                observed_valid_2d,
                observed_center_3d,
                observed_valid_3d,
            )
        center_2d = prediction["center_2d"][0, -1].cpu().numpy()
        center_3d = prediction["center_3d"][0, -1].cpu().numpy()
        person_lower = frame["person_lower"]
        person_size = frame["person_size"]
        metric_origin = frame["metric_origin"]
        metric_scale = float(frame["metric_scale"])
        refined: list[dict[str, Any]] = []
        applied_2d = 0
        applied_3d = 0
        for index, source in enumerate(regions):
            region = dict(source)
            name = str(region.get("name", ""))
            try:
                region_index = REGION_NAMES.index(name)
            except ValueError:
                refined.append(region)
                continue
            if self.learned_only and not self._is_learned(region):
                refined.append(region)
                continue
            if float(frame["observed_valid_2d"][region_index]) > 0.0:
                region["center_2d"] = (
                    center_2d[region_index] * person_size + person_lower
                ).tolist()
                region["temporal_refiner_applied_2d"] = True
                applied_2d += 1
            if float(frame["observed_valid_3d"][region_index]) > 0.0:
                region["center_3d"] = (
                    center_3d[region_index] * metric_scale + metric_origin
                ).tolist()
                region["temporal_refiner_applied_3d"] = True
                applied_3d += 1
            refined.append(region)
        return refined, {
            "enabled": True,
            "learned_only": self.learned_only,
            "sequence_length": len(self._frames),
            "applied_2d": applied_2d,
            "applied_3d": applied_3d,
        }
