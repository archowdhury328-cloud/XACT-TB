"""Deterministic image preprocessing: grayscale -> [resize] -> CLAHE -> BM3D -> [resize] -> 3ch.

Value ranges: loaded uint8 [0,255]; CLAHE works on uint8; BM3D works on float [0,1] (sigma in
those units, so sigma=0.05 as in the paper); output float32 [0,1], shape (H,W). Backbone-specific
normalisation is NOT applied here (see models.normalize).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessConfig:
    size: int = 224
    clahe: bool = True
    clahe_clip: float = 2.0  # paper
    clahe_grid: int = 8  # paper (8x8)
    bm3d: bool = True
    bm3d_sigma: float = 0.05  # paper
    bm3d_lambda_thr3d: float = 2.7  # paper's "lambda_2D = 2.7 sigma" mapped to package lambda_thr3d
    resize_order: str = "before_denoise"  # ASSUMPTION: paper does not state resize position
    interpolation: str = "area"  # ASSUMPTION
    cache_dir: str | None = None

    def validate(self) -> None:
        if self.resize_order not in ("before_denoise", "after_denoise"):
            raise ValueError("resize_order must be before_denoise|after_denoise")
        if self.interpolation not in _INTERP:
            raise ValueError(f"interpolation must be one of {sorted(_INTERP)}")


_INTERP = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}


def load_gray_uint8(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def _resize(img: np.ndarray, size: int, interp: str) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=_INTERP[interp])


def apply_clahe(img_u8: np.ndarray, clip: float, grid: int) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(img_u8)


def apply_bm3d(img01: np.ndarray, sigma: float, lambda_thr3d: float) -> np.ndarray:
    try:
        import bm3d
    except ImportError as e:  # never silently substitute another denoiser
        raise RuntimeError("bm3d package missing; install it or set bm3d=false explicitly") from e
    prof = bm3d.BM3DProfile()
    prof.lambda_thr3d = lambda_thr3d
    out = bm3d.bm3d(img01.astype(np.float64), sigma_psd=sigma, profile=prof)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def preprocess_array(img_u8: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    cfg.validate()
    x = img_u8
    if cfg.resize_order == "before_denoise":
        x = _resize(x, cfg.size, cfg.interpolation)
    if cfg.clahe:
        x = apply_clahe(x, cfg.clahe_clip, cfg.clahe_grid)
    x = x.astype(np.float32) / 255.0
    if cfg.bm3d:
        x = apply_bm3d(x, cfg.bm3d_sigma, cfg.bm3d_lambda_thr3d)
    if cfg.resize_order == "after_denoise":
        x = np.clip(_resize(x, cfg.size, cfg.interpolation), 0, 1)
    return x.astype(np.float32)


def _versions() -> dict[str, str]:
    out = {"cv2": cv2.__version__}
    try:
        import importlib.metadata as md

        import bm3d

        out["bm3d"] = md.version("bm3d")
        del bm3d
    except Exception:
        out["bm3d"] = "none"
    return out


def cache_key(file_sha256: str, cfg: PreprocessConfig) -> str:
    d = asdict(cfg)
    d.pop("cache_dir")
    blob = json.dumps({"f": file_sha256, "c": d, "v": _versions()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def preprocess_file(path: str | Path, cfg: PreprocessConfig, file_sha256: str | None = None) -> np.ndarray:
    """Preprocess with optional on-disk cache keyed by (checksum, config, package versions)."""
    cache_path = None
    if cfg.cache_dir and file_sha256:
        cache_path = Path(cfg.cache_dir) / f"{cache_key(file_sha256, cfg)}.npy"
        if cache_path.exists():
            return np.load(cache_path)
    out = preprocess_array(load_gray_uint8(path), cfg)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, out)
    return out
