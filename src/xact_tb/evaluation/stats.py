"""Uncertainty and paired tests. Never manufactures a finite p-value when assumptions fail."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def bootstrap_ci(
    y_true: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray | None = None,
    metric=roc_auc_score,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap. Resamples groups when given (group bootstrap), else images.

    The ``unit`` field states which was used. Single-class resamples are skipped and counted.
    """
    y, s = np.asarray(y_true), np.asarray(score)
    rng = np.random.default_rng(seed)
    if groups is None:
        unit, idx_sets = "image", None
    else:
        unit = "group"
        g = np.asarray(groups)
        uniq, inv = np.unique(g, return_inverse=True)
        idx_sets = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    vals, skipped = [], 0
    for _ in range(n_boot):
        if idx_sets is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            pick = rng.integers(0, len(idx_sets), len(idx_sets))
            idx = np.concatenate([idx_sets[k] for k in pick])
        if len(np.unique(y[idx])) < 2:
            skipped += 1
            continue
        vals.append(metric(y[idx], s[idx]))
    if not vals:
        return {
            "point": float(metric(y, s)),
            "lo": float("nan"),
            "hi": float("nan"),
            "unit": unit,
            "n_boot": 0,
            "skipped": skipped,
        }
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {
        "point": float(metric(y, s)),
        "lo": float(lo),
        "hi": float(hi),
        "unit": unit,
        "n_boot": len(vals),
        "skipped": skipped,
    }


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_paired(y_true: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict:
    """DeLong test for two correlated AUCs (same samples). p is NaN if variance is degenerate
    (e.g., both AUCs perfect), in which case the test is not valid and none is reported."""
    y = np.asarray(y_true).astype(int)
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m < 2 or n < 2:
        return {
            "auc1": float("nan"),
            "auc2": float("nan"),
            "z": float("nan"),
            "p": float("nan"),
            "note": "need >=2 samples per class",
        }
    aucs, v10, v01 = [], [], []
    for s in (np.asarray(s1, float), np.asarray(s2, float)):
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(s)
        aucs.append((tz[pos].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[pos] - tx) / n)
        v01.append(1 - (tz[neg] - ty) / m)
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    res = {"auc1": float(aucs[0]), "auc2": float(aucs[1])}
    if not np.isfinite(var) or var <= 1e-18:
        return {
            **res,
            "z": float("nan"),
            "p": float("nan"),
            "note": "degenerate variance (e.g., perfect/identical scores): test not valid",
        }
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return {**res, "z": float(z), "p": float(2 * stats.norm.sf(abs(z))), "note": ""}


def mcnemar_exact(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> dict:
    """Exact (binomial) McNemar on discordant pairs. p=NaN when there are no discordant pairs."""
    c1, c2 = np.asarray(pred1) == np.asarray(y_true), np.asarray(pred2) == np.asarray(y_true)
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    if b + c == 0:
        return {"b": b, "c": c, "p": float("nan"), "note": "no discordant pairs: test undefined"}
    return {"b": b, "c": c, "p": float(stats.binomtest(min(b, c), b + c, 0.5).pvalue), "note": ""}
