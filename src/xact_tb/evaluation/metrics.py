"""Metrics derived from sample-level scores. Positive class = TB (1); Normal = 0.

Conventions: hard label = score > threshold (default 0.0 for logits/SVM margins). ROC AUC and AP use
the continuous score. Undefined ratios (zero denominator) -> NaN, never 0. SVM margins are NOT
probabilities; calibration metrics are only computed when ``proba`` is explicitly supplied.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
    }


def _div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def metrics_from_confusion(c: dict[str, int]) -> dict[str, float]:
    tp, fn, tn, fp = c["tp"], c["fn"], c["tn"], c["fp"]
    sens, spec, prec = _div(tp, tp + fn), _div(tn, tn + fp), _div(tp, tp + fp)
    f1 = _div(2 * tp, 2 * tp + fp + fn)
    return {
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
    }


def compute_metrics(
    y_true: np.ndarray, score: np.ndarray, threshold: float = 0.0, proba: np.ndarray | None = None
) -> dict:
    y_true, score = np.asarray(y_true).astype(int), np.asarray(score, dtype=float)
    if not np.isfinite(score).all():
        raise ValueError("non-finite scores")
    c = confusion(y_true, (score > threshold).astype(int))
    out: dict = {**metrics_from_confusion(c), "confusion": c, "n": int(len(y_true))}
    single = len(np.unique(y_true)) < 2
    out["roc_auc"] = float("nan") if single else float(roc_auc_score(y_true, score))
    out["average_precision"] = float("nan") if single else float(average_precision_score(y_true, score))
    if proba is not None:
        p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1 - 1e-12)
        out["brier"] = float(np.mean((p - y_true) ** 2))
        out["log_loss"] = float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    return out
