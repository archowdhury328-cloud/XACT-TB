"""Arithmetic-consistency audit of *reported* metrics (the paper's Tables 1/3).

Given class sizes and a dict of reported percentages (2-decimals), search all integer confusion
matrices and report whether any reproduces every reported value after rounding. Also reports
pairwise identities that hold regardless of cohort size (e.g., balanced test sets: accuracy equals
the mean of sensitivity and specificity). Nothing is adjusted; findings are only reported.
"""

from __future__ import annotations

KEYS = ("accuracy", "sensitivity", "specificity", "precision", "f1")


def _r(x: float, nd: int) -> float:
    return round(x * 100, nd)


def find_consistent_confusions(
    n_pos: int, n_neg: int, reported: dict[str, float], decimals: int = 2, tol: float = 0.0
) -> list[dict]:
    """All (tp,tn) with fn=n_pos-tp, fp=n_neg-tn matching every reported metric (percent)."""
    import numpy as np

    half = 0.5 * 10 ** (-decimals) + tol
    tp, tn = np.meshgrid(np.arange(n_pos + 1), np.arange(n_neg + 1), indexing="ij")
    tp, tn = tp.astype(float), tn.astype(float)
    fn, fp = n_pos - tp, n_neg - tn
    with np.errstate(divide="ignore", invalid="ignore"):
        m = {
            "accuracy": (tp + tn) / (n_pos + n_neg),
            "sensitivity": tp / n_pos,
            "specificity": tn / n_neg,
            "precision": tp / (tp + fp),
            "f1": 2 * tp / (2 * tp + fp + fn),
        }
    ok = np.ones(tp.shape, dtype=bool)
    for k, v in reported.items():
        if k in KEYS:
            ok &= np.isfinite(m[k]) & (np.abs(m[k] * 100 - v) <= half + 1e-9)
    return [
        {"tp": int(a), "fn": int(n_pos - a), "tn": int(b), "fp": int(n_neg - b)}
        for a, b in zip(tp[ok], tn[ok], strict=True)
    ]


def balanced_identity_gap(reported: dict[str, float]) -> float:
    """For equal class sizes: accuracy - (sens+spec)/2 must be 0 (percentage points)."""
    return reported["accuracy"] - (reported["sensitivity"] + reported["specificity"]) / 2


def f1_from_pr(reported: dict[str, float]) -> float:
    """F1 implied by reported precision & sensitivity (valid for single-pass counts)."""
    p, r = reported["precision"], reported["sensitivity"]
    return 2 * p * r / (p + r)


def audit_row(name: str, n_pos: int, n_neg: int, reported: dict[str, float]) -> dict:
    hits = find_consistent_confusions(n_pos, n_neg, reported)
    row = {
        "model": name,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_consistent_confusions": len(hits),
        "f1_implied_by_P_and_R": round(f1_from_pr(reported), 2),
        "f1_reported": reported["f1"],
    }
    if n_pos == n_neg:
        row["accuracy_minus_mean_sens_spec"] = round(balanced_identity_gap(reported), 3)
    return row
