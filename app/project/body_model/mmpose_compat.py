from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec


def _install_python_mmcv_extension() -> None:
    """Provide import-time stubs for MMCV's optional compiled extension.

    ``mmcv-lite`` contains the Python operator wrappers but not ``mmcv._ext``.
    MMPose and MMDetection import a broad registry of operators even when the
    selected RTMPose model does not execute them. The stubs keep those unused
    wrappers importable; they deliberately fail if a model actually requests
    an unavailable compiled operator.
    """

    extension = types.ModuleType("mmcv._ext")

    def _unavailable(*args, **kwargs):
        raise RuntimeError(
            "This model requested an MMCV compiled operator, but only "
            "mmcv-lite is installed. Use a full MMCV build for this operator."
        )

    def _extension_getattr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return _unavailable

    extension.__file__ = "<mmcv-lite-extension-stub>"
    extension.__package__ = "mmcv"
    extension.__spec__ = ModuleSpec("mmcv._ext", loader=None)
    extension.__getattr__ = _extension_getattr
    sys.modules["mmcv._ext"] = extension


def _clear_partial_mmcv_ops() -> None:
    for name in list(sys.modules):
        if name == "mmcv.ops" or name.startswith("mmcv.ops."):
            sys.modules.pop(name, None)


def _has_native_mmcv_extension() -> bool:
    extension = sys.modules.get("mmcv._ext")
    if extension is None:
        return False
    return not str(getattr(extension, "__file__", "")).startswith(
        "<mmcv-lite-extension-stub>"
    )


def _prepare_coco_api() -> str:
    """Make MMPose's optional ``xtcocotools`` import portable on Windows.

    The runtime only performs pose inference; it does not invoke COCO
    evaluation. MMPose nevertheless imports dataset and metric registries at
    startup. ``pycocotools`` exposes the compatible COCO/mask API and has a
    maintained CPython 3.13 Windows wheel, so alias it only when the legacy
    package is absent.
    """

    try:
        import xtcocotools  # noqa: F401
    except ModuleNotFoundError:
        import pycocotools
        from pycocotools import coco, cocoeval, mask

        sys.modules.setdefault("xtcocotools", pycocotools)
        sys.modules.setdefault("xtcocotools.coco", coco)
        sys.modules.setdefault("xtcocotools.cocoeval", cocoeval)
        sys.modules.setdefault("xtcocotools.mask", mask)
        return "pycocotools_compat"
    return "xtcocotools"


def prepare_torch_checkpoint_loading() -> dict[str, bool]:
    """Keep trusted OpenMMLab checkpoints loadable on PyTorch 2.6+.

    PyTorch changed ``torch.load`` to default to ``weights_only=True``. The
    bundled RTMPose checkpoint is a trusted local artifact from OpenMMLab and
    uses the older checkpoint metadata format, so the compatibility wrapper
    restores the pre-2.6 default for this process only.
    """

    import torch

    if getattr(torch.load, "_body_model_compat", False):
        return {"weights_only_false_for_trusted_checkpoint": True}

    original_load = torch.load

    def _load(*args, **kwargs):
        if "weights_only" in kwargs:
            return original_load(*args, **kwargs)
        try:
            return original_load(*args, weights_only=False, **kwargs)
        except TypeError as error:
            # PyTorch versions before 2.6 do not expose weights_only.
            # Retry only for that API difference; preserve other TypeErrors.
            if "weights_only" not in str(error):
                raise
            return original_load(*args, **kwargs)

    _load._body_model_compat = True
    torch.load = _load
    return {"weights_only_false_for_trusted_checkpoint": True}


def prepare_mmpose_import() -> dict[str, str | bool]:
    """Make MMPose importable when optional MMCV CUDA ops are unavailable.

    MMPose imports every registered head at package import time. The RTMPose
    path does not use EDPose or deformable attention, but ``mmcv-lite`` cannot
    provide those compiled operators. In that environment, install a narrow
    module shim for the unused transformer-head package. A full MMCV install
    keeps the native import path untouched.
    """

    coco_api = _prepare_coco_api()
    if "mmcv._ext" in sys.modules and not _has_native_mmcv_extension():
        return {
            "optional_mmcv_ops": False,
            "mode": "python_mmcv_ops_fallback",
            "coco_api": coco_api,
        }

    try:
        from mmcv.ops import MultiScaleDeformableAttention  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        _clear_partial_mmcv_ops()
        _install_python_mmcv_extension()
        try:
            from mmcv.ops import MultiScaleDeformableAttention  # noqa: F401
        except (ImportError, ModuleNotFoundError, OSError):
            _clear_partial_mmcv_ops()
            module_name = "mmpose.models.heads.transformer_heads"
            shim = types.ModuleType(module_name)
            shim.__package__ = "mmpose.models.heads"
            shim.EDPoseHead = None
            shim.__all__ = ["EDPoseHead"]
            sys.modules[module_name] = shim
            return {
                "optional_mmcv_ops": False,
                "mode": "rtmpose_without_optional_ops",
                "coco_api": coco_api,
            }

        return {
            "optional_mmcv_ops": False,
            "mode": "python_mmcv_ops_fallback",
            "coco_api": coco_api,
        }

    return {
        "optional_mmcv_ops": True,
        "mode": "native_mmcv_ops",
        "coco_api": coco_api,
    }


def prepare_mmdet_inference() -> dict[str, str | bool]:
    """Use torchvision NMS for RTMDet when the server has ``mmcv-lite``.

    RTMDet-Nano itself does not need custom MMCV kernels, but MMDetection's
    post-processing imports MMCV's compiled NMS wrapper. The existing server
    already has a matching CUDA-enabled torchvision build, so this narrow
    substitution preserves the environment and executes the same class-aware
    greedy NMS operation.
    """
    compatibility = prepare_mmpose_import()
    has_native_extension = _has_native_mmcv_extension()
    if compatibility.get("optional_mmcv_ops") and has_native_extension:
        return {**compatibility, "mmdet_nms": "native_mmcv"}

    import torch
    from torchvision.ops import batched_nms as torchvision_batched_nms

    def _batched_nms(
        boxes,
        scores,
        idxs,
        nms_cfg,
        class_agnostic: bool = False,
    ):
        if nms_cfg is None:
            sorted_scores, keep = scores.sort(descending=True)
            return torch.cat((boxes[keep], sorted_scores[:, None]), dim=1), keep
        options = dict(nms_cfg)
        nms_type = options.pop("type", "nms")
        if nms_type not in ("nms",):
            raise RuntimeError(
                f"torchvision fallback does not support NMS type {nms_type!r}"
            )
        options.pop("split_thr", None)
        options.pop("max_num", None)
        class_agnostic = bool(
            options.pop("class_agnostic", class_agnostic)
        )
        threshold = float(
            options.pop("iou_threshold", options.pop("iou_thr", 0.5))
        )
        if options:
            raise RuntimeError(
                f"unsupported torchvision NMS options: {sorted(options)}"
            )
        groups = torch.zeros_like(idxs) if class_agnostic else idxs
        keep = torchvision_batched_nms(boxes, scores, groups, threshold)
        detections = torch.cat((boxes[keep], scores[keep, None]), dim=1)
        return detections, keep

    # These modules bind batched_nms at import time. Patch the paths used by
    # RTMDet after the compatibility shim has made MMDetection importable.
    import mmdet.models.dense_heads.base_dense_head as base_dense_head
    import mmdet.models.layers.bbox_nms as bbox_nms

    base_dense_head.batched_nms = _batched_nms
    bbox_nms.batched_nms = _batched_nms
    return {**compatibility, "mmdet_nms": "torchvision"}
