from __future__ import annotations

import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _install_portalocker_fallback() -> None:
    """Keep inference usable when iopath was installed without portalocker.

    SAM2 imports iopath even though static inference does not acquire a file
    lock. The fallback is process-local and is intentionally not used for
    checkpoint downloads or multi-process coordination.
    """
    try:
        import portalocker  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    class _Lock:
        _locks: dict[str, threading.RLock] = {}
        _guard = threading.Lock()

        def __init__(self, path: str, **_: object) -> None:
            key = str(path)
            with self._guard:
                self._lock = self._locks.setdefault(key, threading.RLock())

        def acquire(self, *_: object, **__: object) -> bool:
            return self._lock.acquire()

        def release(self) -> None:
            self._lock.release()

        def __enter__(self) -> "_Lock":
            self.acquire()
            return self

        def __exit__(self, *_: object) -> None:
            self.release()

    module = types.ModuleType("portalocker")
    module.Lock = _Lock  # type: ignore[attr-defined]
    sys.modules["portalocker"] = module


def _as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _config_name(config: str | Path, repo_path: Path) -> str:
    """Convert a config path into the name expected by Hydra's sam2 module."""
    path = Path(config)
    if not path.is_absolute():
        return str(path).replace("\\", "/")
    candidates = [repo_path / "sam2", repo_path]
    for root in candidates:
        try:
            return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
        except ValueError:
            continue
    raise ValueError(
        "SAM2 config must be relative to the SAM2 repository or use a "
        "configs/... name"
    )


@dataclass(frozen=True)
class SAM2MaskResult:
    mask: np.ndarray
    score: float
    mask_index: int
    bbox_xyxy: list[float]
    mask_area_px: int
    bbox_area_px: int
    mask_inside_bbox_fraction: float
    mask_bbox_xyxy: list[int] | None
    inference_ms: float

    def metadata(self) -> dict:
        return {
            "model": "SAM2",
            "score": float(self.score),
            "mask_index": int(self.mask_index),
            "bbox_xyxy": [float(value) for value in self.bbox_xyxy],
            "mask_area_px": int(self.mask_area_px),
            "bbox_area_px": int(self.bbox_area_px),
            "mask_inside_bbox_fraction": float(self.mask_inside_bbox_fraction),
            "mask_bbox_xyxy": self.mask_bbox_xyxy,
            "inference_ms": float(self.inference_ms),
        }


class SAM2Segmenter:
    """Lazy bbox-prompt SAM2 image segmentation for one selected person."""

    def __init__(
        self,
        config: str | Path,
        checkpoint: str | Path,
        *,
        repo_path: str | Path,
        device: str = "cuda:0",
    ) -> None:
        repo = Path(repo_path).expanduser().resolve()
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")
        if not (repo / "sam2").is_dir():
            raise FileNotFoundError(f"SAM2 repository not found: {repo}")

        _install_portalocker_fallback()
        sys.path.insert(0, str(repo))
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.checkpoint = str(checkpoint_path)
        self.config = _config_name(config, repo)
        self._torch = torch
        self._predictor = SAM2ImagePredictor(
            build_sam2(self.config, self.checkpoint, device=self.device)
        )

    def _autocast(self):
        if self.device.startswith("cuda"):
            return self._torch.autocast("cuda", dtype=self._torch.bfloat16)
        return self._torch.autocast("cpu", enabled=False)

    @staticmethod
    def _box_mask(shape: tuple[int, int], bbox: np.ndarray) -> tuple[np.ndarray, list[int]]:
        height, width = shape
        x0, y0, x1, y1 = np.round(bbox).astype(int).tolist()
        x0 = int(np.clip(x0, 0, width))
        x1 = int(np.clip(x1, 0, width))
        y0 = int(np.clip(y0, 0, height))
        y1 = int(np.clip(y1, 0, height))
        box_mask = np.zeros(shape, dtype=bool)
        if x1 > x0 and y1 > y0:
            box_mask[y0:y1, x0:x1] = True
        return box_mask, [x0, y0, x1, y1]

    def predict(self, image_rgb: np.ndarray, bbox_xyxy: object) -> SAM2MaskResult:
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image_rgb must have shape [H, W, 3]")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)
        if not np.all(np.isfinite(bbox)):
            raise ValueError("bbox_xyxy must contain finite values")

        with self._torch.inference_mode(), self._autocast():
            started = time.perf_counter()
            self._predictor.set_image(image)
            masks, scores, _ = self._predictor.predict(
                box=bbox,
                multimask_output=True,
            )
            if self.device.startswith("cuda"):
                self._torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0

        mask_values = _as_numpy(masks).astype(bool)
        score_values = _as_numpy(scores).reshape(-1).astype(np.float32)
        if mask_values.ndim == 2:
            mask_values = mask_values[None, ...]
        if mask_values.ndim != 3 or mask_values.shape[0] == 0:
            raise RuntimeError("SAM2 returned no image masks")
        if score_values.shape[0] != mask_values.shape[0]:
            raise RuntimeError("SAM2 mask and score counts do not match")
        score_values = np.nan_to_num(score_values, nan=-np.inf)
        selected_index = int(np.argmax(score_values))
        mask = mask_values[selected_index]
        box_mask, _ = self._box_mask(mask.shape, bbox)
        ys, xs = np.where(mask)
        mask_bbox = (
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if xs.size
            else None
        )
        return SAM2MaskResult(
            mask=mask,
            score=float(score_values[selected_index]),
            mask_index=selected_index,
            bbox_xyxy=bbox.tolist(),
            mask_area_px=int(mask.sum()),
            bbox_area_px=int(box_mask.sum()),
            mask_inside_bbox_fraction=float((mask & box_mask).sum() / max(int(mask.sum()), 1)),
            mask_bbox_xyxy=mask_bbox,
            inference_ms=float(elapsed_ms),
        )
