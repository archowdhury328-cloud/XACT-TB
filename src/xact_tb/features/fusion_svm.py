"""Embedding extraction, ID-aligned fusion (concatenation), train-only scaler, SVM fitting.

The SVM is fitted on FROZEN embeddings; nothing backpropagates through the networks.
Reproduction config: concatenate CNN(256) + Swin(D) -> StandardScaler (fit on train only, ASSUMPTION)
-> SVC. Kernel/C/gamma are not given in the paper: a modest predefined grid is searched using
VALIDATION data only (never test), identical for all comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class FeatureSet:
    ids: list[str]
    labels: np.ndarray
    blocks: dict[str, np.ndarray] = field(default_factory=dict)  # e.g. {"cnn": (N,256), "swin": (N,D)}


@torch.no_grad()
def extract_block(embed_fn, loader, device) -> FeatureSet:
    """Deterministic eval-mode embedding extraction. ``embed_fn(images)->(B,D)``."""
    ids, ys, feats = [], [], []
    for b in loader:
        feats.append(embed_fn(b["image"].to(device)).float().cpu())
        ys.append(b["label"])
        ids.extend(b["image_id"])
    return FeatureSet(ids, torch.cat(ys).numpy().astype(int), {"x": torch.cat(feats).numpy()})


def fuse(sets: dict[str, FeatureSet]) -> FeatureSet:
    """Concatenate per-branch embeddings after asserting identical IDs, order and labels."""
    names = list(sets)
    ref = sets[names[0]]
    for n in names[1:]:
        if sets[n].ids != ref.ids:
            raise ValueError(f"sample order/IDs differ between branches '{names[0]}' and '{n}'")
        if not np.array_equal(sets[n].labels, ref.labels):
            raise ValueError(f"labels differ between branches '{names[0]}' and '{n}'")
    blocks = {n: sets[n].blocks["x"] for n in names}
    return FeatureSet(ref.ids, ref.labels, blocks)


def concat_blocks(fs: FeatureSet, order: list[str]) -> np.ndarray:
    return np.concatenate([fs.blocks[k] for k in order], axis=1)


DEFAULT_GRID = {"kernel": ["linear", "rbf"], "C": [0.1, 1.0, 10.0], "gamma": ["scale"]}


def fit_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    grid: dict | None = None,
    seed: int = 42,
) -> tuple[StandardScaler, SVC, dict]:
    """Scaler fit on train only; hyperparameters chosen by val AUC (ties: first in grid order)."""
    grid = grid or DEFAULT_GRID
    scaler = StandardScaler().fit(x_train)
    xt, xv = scaler.transform(x_train), scaler.transform(x_val)
    best, best_auc, trials = None, -np.inf, []
    for kernel, C, gamma in product(grid["kernel"], grid["C"], grid["gamma"]):
        svm = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=None, random_state=seed)
        svm.fit(xt, y_train)
        auc = roc_auc_score(y_val, svm.decision_function(xv)) if len(np.unique(y_val)) > 1 else np.nan
        trials.append({"kernel": kernel, "C": C, "gamma": gamma, "val_auc": float(auc)})
        if best is None or (np.isfinite(auc) and auc > best_auc):
            best, best_auc = svm, auc
    info = {
        "selected": {"kernel": best.kernel, "C": best.C, "gamma": best.gamma},
        "n_support": int(best.n_support_.sum()),
        "trials": trials,
    }
    return scaler, best, info
