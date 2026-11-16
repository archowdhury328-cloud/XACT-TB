"""Hashing, seeding, and small IO helpers shared across the package."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np

NORMAL, TB = 0, 1
LABEL_NAMES = {NORMAL: "Normal", TB: "TB"}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id: int) -> None:
    import torch

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def safe_join(root: str | Path, rel: str | Path) -> Path:
    """Join and refuse paths escaping ``root`` (path-traversal guard)."""
    root_p = Path(root).resolve()
    full = (root_p / rel).resolve()
    if root_p != full and root_p not in full.parents:
        raise ValueError(f"path escapes data root: {rel}")
    return full


def environment_record() -> dict[str, Any]:
    import torch

    rec: dict[str, Any] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rec.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 2**30, 2))
    for mod in ("torchvision", "timm", "numpy", "pandas", "scipy", "sklearn", "cv2", "bm3d"):
        try:
            m = __import__(mod)
            rec[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            rec[mod] = None
    return rec


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
