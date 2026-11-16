"""Measured computational cost of the implemented models (never copied from the paper)."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


def param_counts(model: nn.Module) -> dict[str, int]:
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": tot, "trainable": tr}


def flops_fvcore(model: nn.Module, size: int = 224) -> dict:
    """fvcore counts one fused multiply-add as ONE flop (i.e. MACs). Unsupported ops are listed."""
    from fvcore.nn import FlopCountAnalysis

    model.eval()
    x = torch.zeros(1, 3, size, size)
    fa = FlopCountAnalysis(model.cpu(), x)
    fa.unsupported_ops_warnings(False)
    fa.uncalled_modules_warnings(False)
    return {
        "macs_g": fa.total() / 1e9,
        "convention": "fvcore (1 MAC = 1 flop)",
        "unsupported_ops": dict(fa.unsupported_ops()),
    }


def serialized_size_mb(model: nn.Module) -> float:
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 2**20


@torch.no_grad()
def latency(model: nn.Module, device="cpu", batch: int = 1, size: int = 224, warmup: int = 5, iters: int = 30) -> dict:
    dev = torch.device(device)
    model = model.to(dev).eval()
    x = torch.rand(batch, 3, size, size, device=dev)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        model(x)
    sync()
    ts = []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        t = time.perf_counter()
        model(x)
        sync()
        ts.append((time.perf_counter() - t) * 1000)
    ts_a = np.array(ts)
    res = {
        "device": str(dev),
        "batch": batch,
        "median_ms": float(np.median(ts_a)),
        "p95_ms": float(np.percentile(ts_a, 95)),
        "throughput_img_s": float(batch / (np.median(ts_a) / 1000)),
        "iters": iters,
        "precision": "fp32",
    }
    if dev.type == "cuda":
        res["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return res


def svm_storage(svm) -> dict:
    import io

    import joblib

    buf = io.BytesIO()
    joblib.dump(svm, buf)
    return {"n_support_vectors": int(svm.n_support_.sum()), "serialized_kb": buf.getbuffer().nbytes / 1024}
