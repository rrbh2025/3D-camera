from __future__ import annotations

import importlib.metadata
import os
import platform
import re
from pathlib import Path
from typing import Any


# This is the server matrix that has been tested with the current body model.
# It deliberately excludes PyTorch/CUDA from install requirements: the model
# must reuse the existing server runtime instead of replacing it.
TESTED_SERVER_MATRIX = {
    "python": "3.11.5",
    "torch": "2.7.1+cu118",
    "torchvision": "0.22.1+cu118",
    "numpy": "1.26.4",
    "mmengine": "0.10.7",
    "mmcv": "2.1.0",
    "mmdet": "3.3.0",
    "mmpose": "1.3.2",
    "opencv-python-headless": "4.8.1.78",
}


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", value.split("+", 1)[0])
    return tuple(int(number) for number in numbers[:4]) or (0,)


def _package_version(name: str) -> str | None:
    candidates = (name, "mmcv-lite") if name == "mmcv" else (name,)
    for candidate in candidates:
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _source_package_version(root: str | Path, package: str) -> str | None:
    version_file = Path(root) / package / "version.py"
    if not version_file.is_file():
        return None
    match = re.search(
        r"__version__\s*=\s*['\"]([^'\"]+)['\"]",
        version_file.read_text(encoding="utf-8"),
    )
    return match.group(1) if match else None


def collect_environment() -> dict[str, Any]:
    """Collect versions without importing optional OpenMMLab components."""
    packages = {
        name: _package_version(name)
        for name in (
            "numpy",
            "torch",
            "torchvision",
            "mmengine",
            "mmcv",
            "mmdet",
            "mmpose",
            "opencv-python-headless",
        )
    }
    package_sources: dict[str, str] = {}
    if packages["mmpose"] is None:
        mmpose_root = os.environ.get(
            "MMPPOSE_ROOT", "/home/u25600009140214/data/MMPose/source"
        )
        source_version = _source_package_version(mmpose_root, "mmpose")
        if source_version is not None:
            packages["mmpose"] = source_version
            package_sources["mmpose"] = str(Path(mmpose_root))
    try:
        import torch

        cuda: dict[str, Any] = {
            "available": bool(torch.cuda.is_available()),
            "version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
        }
    except Exception as error:  # pragma: no cover - diagnostic fallback
        cuda = {"error": f"{type(error).__name__}: {error}"}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "package_sources": package_sources,
        "cuda": cuda,
    }


def check_environment() -> dict[str, Any]:
    """Return compatibility findings; this function never changes the env."""
    environment = collect_environment()
    packages = environment["packages"]
    warnings: list[str] = []
    errors: list[str] = []

    python_version = _version_tuple(environment["python"])
    if not (3, 9) <= python_version < (3, 14):
        errors.append("Python must be >=3.9 and <3.14 for this project")

    numpy_version = packages.get("numpy")
    if numpy_version is None:
        errors.append("numpy is not installed")
    elif not (1, 24) <= _version_tuple(numpy_version) < (2, 0):
        errors.append("NumPy must stay in the >=1.24,<2.0 range")

    torch_version = packages.get("torch")
    if torch_version is None:
        errors.append("torch is not installed")
    elif not (2, 0) <= _version_tuple(torch_version) < (2, 8):
        errors.append("PyTorch must stay in the >=2.0,<2.8 range")

    # These ranges retain the known MMPose 1.x/MMDetection 3.x API contract.
    ranges = {
        "mmengine": ((0, 10), (0, 11)),
        "mmcv": ((2, 0), (2, 2)),
        "mmdet": ((3, 0), (3, 4)),
        "mmpose": ((1, 3), (1, 4)),
    }
    for package, (lower, upper) in ranges.items():
        value = packages.get(package)
        if value is None:
            warnings.append(f"{package} is not installed; only keypoint inference needs it")
            continue
        version = _version_tuple(value)[:2]
        if not lower <= version < upper:
            errors.append(
                f"{package} {value} is outside the tested range "
                f">={lower[0]}.{lower[1]},<{upper[0]}.{upper[1]}"
            )

    if packages.get("mmcv") == "2.2.0":
        errors.append(
            "mmcv 2.2.0 is incompatible with the server's mmdet 3.3.0; "
            "use mmcv-lite 2.1.0"
        )

    for package, expected in TESTED_SERVER_MATRIX.items():
        actual = environment["python"] if package == "python" else packages.get(package)
        if actual is not None and str(actual) != expected:
            warnings.append(
                f"{package} {actual} differs from tested server version {expected}"
            )

    environment["tested_server_matrix"] = TESTED_SERVER_MATRIX
    environment["warnings"] = warnings
    environment["errors"] = errors
    environment["compatible"] = not errors
    return environment
