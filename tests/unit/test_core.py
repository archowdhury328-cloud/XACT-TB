"""Unit tests. Synthetic data only; nothing here is a research result."""

import numpy as np
import pandas as pd
import pytest
import torch

from xact_tb.evaluation.audit import balanced_identity_gap, find_consistent_confusions
from xact_tb.evaluation.metrics import compute_metrics, confusion, metrics_from_confusion
from xact_tb.evaluation.stats import bootstrap_ci, delong_paired, mcnemar_exact
from xact_tb.features.fusion_svm import FeatureSet, concat_blocks, fit_svm, fuse
from xact_tb.models.cnn import CNNConfig, CustomCNN
from xact_tb.preprocessing.augment import (
    AugmentConfig,
    build_augmentation,
    build_epoch_plan,
    plan_counts,
)
from xact_tb.preprocessing.pipeline import PreprocessConfig, cache_key, preprocess_array
from xact_tb.training.loss import weighted_bce_with_logits
from xact_tb.utils.common import safe_join


# ---- loss -----------------------------------------------------------------------------------
def test_weighted_bce_matches_hand_computation():
    z = torch.tensor([0.0, 2.0])
    y = torch.tensor([1.0, 0.0])  # noqa: E702
    # sample1: TB, p=.5 -> -ln .5 *1.5 ; sample2: Normal, p=s(2) -> -ln(1-s(2)) *1.0
    s2 = 1 / (1 + np.exp(-2.0))
    expected = (1.5 * -np.log(0.5) + 1.0 * -np.log(1 - s2)) / 2
    assert weighted_bce_with_logits(z, y).item() == pytest.approx(expected, rel=1e-6)


def test_weighted_bce_stable_for_extreme_logits():
    v = weighted_bce_with_logits(torch.tensor([1e4, -1e4]), torch.tensor([0.0, 1.0]))
    assert torch.isfinite(v)


# ---- augmentation / balance -------------------------------------------------------------------
def test_epoch_plan_balances_to_1_to_1_and_augments_only_tb():
    labels = np.array([0] * 10 + [1] * 4)
    plan = build_epoch_plan(labels, np.random.default_rng(0))
    c = plan_counts(plan, labels)
    assert c["Normal"] == c["TB"] == 10 and c["TB_augmented_views"] == 6
    assert all(labels[i] == 1 for i, aug in plan if aug)  # never augment Normal
    assert sum(1 for _, aug in plan if not aug) == 14  # all originals kept un-augmented


def test_epoch_plan_no_extra_when_already_balanced():
    labels = np.array([0] * 5 + [1] * 5)
    assert plan_counts(build_epoch_plan(labels, np.random.default_rng(0)), labels)["TB_augmented_views"] == 0


def test_augmentation_has_no_flip_and_preserves_range():
    aug = build_augmentation(AugmentConfig())
    names = [type(t).__name__ for t in aug.transforms]
    assert not any("Flip" in n for n in names)
    out = aug(torch.rand(3, 64, 64))
    assert out.min() >= 0 and out.max() <= 1


# ---- preprocessing ---------------------------------------------------------------------------
def test_preprocess_range_shape_dtype_and_determinism():
    img = (np.random.default_rng(0).random((100, 80)) * 255).astype(np.uint8)
    cfg = PreprocessConfig(size=64, bm3d=False)
    a, b = preprocess_array(img, cfg), preprocess_array(img, cfg)
    assert a.shape == (64, 64) and a.dtype == np.float32 and 0 <= a.min() and a.max() <= 1
    assert np.array_equal(a, b)


def test_bm3d_runs_and_is_deterministic_small():
    img = (np.random.default_rng(1).random((32, 32)) * 255).astype(np.uint8)
    cfg = PreprocessConfig(size=32, bm3d=True)
    a = preprocess_array(img, cfg)
    assert a.shape == (32, 32) and np.isfinite(a).all()
    assert not np.allclose(a, preprocess_array(img, PreprocessConfig(size=32, bm3d=False)))


def test_cache_key_changes_with_config():
    assert cache_key("abc", PreprocessConfig()) != cache_key("abc", PreprocessConfig(clahe_clip=3.0))
    assert cache_key("abc", PreprocessConfig()) == cache_key("abc", PreprocessConfig(cache_dir="x"))


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        preprocess_array(np.zeros((8, 8), np.uint8), PreprocessConfig(resize_order="bad"))


# ---- model -----------------------------------------------------------------------------------
def test_cnn_shapes_and_embedding_dim():
    m = CustomCNN(CNNConfig()).eval()
    x = torch.rand(2, 3, 224, 224)
    assert m.feature_map(x).shape == (2, 256, 14, 14)
    assert m.embed(x).shape == (2, 256) and m(x).shape == (2,)


def test_cnn_projection_when_width_differs():
    m = CustomCNN(CNNConfig(channels=(8, 16), embed_dim=256)).eval()
    assert m.embed(torch.rand(1, 3, 32, 32)).shape == (1, 256)


# ---- fusion / svm ----------------------------------------------------------------------------
def _fs(ids, y, d, seed):
    return FeatureSet(ids, np.array(y), {"x": np.random.default_rng(seed).normal(size=(len(ids), d))})


def test_fusion_concat_and_alignment_checks():
    a, b = _fs(list("abc"), [0, 1, 0], 4, 0), _fs(list("abc"), [0, 1, 0], 6, 1)
    f = fuse({"cnn": a, "swin": b})
    assert concat_blocks(f, ["cnn", "swin"]).shape == (3, 10)
    with pytest.raises(ValueError):
        fuse({"cnn": a, "swin": _fs(list("acb"), [0, 1, 0], 6, 1)})
    with pytest.raises(ValueError):
        fuse({"cnn": a, "swin": _fs(list("abc"), [1, 1, 0], 6, 1)})


def test_svm_scaler_fit_on_train_only_and_separable():
    rng = np.random.default_rng(0)
    xt = np.vstack([rng.normal(0, 1, (40, 5)), rng.normal(3, 1, (40, 5))])
    yt = np.r_[np.zeros(40), np.ones(40)]  # noqa: E702
    xv = np.vstack([rng.normal(0, 1, (10, 5)), rng.normal(3, 1, (10, 5))]) + 100
    yv = np.r_[np.zeros(10), np.ones(10)]  # noqa: E702
    scaler, svm, info = fit_svm(xt, yt, xv, yv)
    assert np.allclose(scaler.mean_, xt.mean(0))  # val shift (+100) did not leak into scaler
    assert info["n_support"] > 0


# ---- metrics ---------------------------------------------------------------------------------
def test_metrics_known_confusion():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array([2, 1, 1, -1, -1, -1, 1, -2.0])  # noqa: E702
    m = compute_metrics(y, s)
    assert m["confusion"] == {"tp": 3, "fn": 1, "tn": 3, "fp": 1}
    assert m["sensitivity"] == 0.75 and m["specificity"] == 0.75 and m["precision"] == 0.75
    assert m["accuracy"] == 0.75 and m["f1"] == 0.75 and m["balanced_accuracy"] == 0.75


def test_roc_auc_uses_scores_not_labels():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])  # noqa: E702
    assert compute_metrics(y, s - 0.5)["roc_auc"] == pytest.approx(0.75)


def test_undefined_metrics_are_nan_not_zero():
    c = metrics_from_confusion({"tp": 0, "fn": 0, "tn": 5, "fp": 0})
    assert np.isnan(c["sensitivity"]) and np.isnan(c["precision"])
    assert np.isnan(compute_metrics(np.zeros(4, int), np.arange(4.0))["roc_auc"])


def test_nonfinite_scores_rejected():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1]), np.array([np.nan, 1.0]))


def test_confusion_counts():
    assert confusion(np.array([1, 0, 1]), np.array([1, 1, 0])) == {"tp": 1, "fn": 1, "tn": 0, "fp": 1}


# ---- stats -----------------------------------------------------------------------------------
def test_bootstrap_ci_brackets_point_and_labels_unit():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(60), np.ones(60)]
    s = np.r_[rng.normal(0, 1, 60), rng.normal(1, 1, 60)]  # noqa: E702
    r = bootstrap_ci(y, s, n_boot=300)
    assert r["unit"] == "image" and r["lo"] <= r["point"] <= r["hi"]
    g = bootstrap_ci(y, s, groups=np.arange(120) // 2, n_boot=300)
    assert g["unit"] == "group"


def test_bootstrap_perfect_scores_does_not_crash():
    y = np.r_[np.zeros(20), np.ones(20)]
    s = y * 2 - 1.0  # noqa: E702
    r = bootstrap_ci(y, s, n_boot=100)
    assert r["point"] == 1.0 and r["lo"] == 1.0


def test_bootstrap_single_class_returns_nan():
    r = bootstrap_ci(np.ones(10), np.arange(10.0), n_boot=20)
    assert np.isnan(r["lo"]) and r["n_boot"] == 0


def test_delong_perfect_scores_gives_nan_p():
    y = np.r_[np.zeros(20), np.ones(20)]
    r = delong_paired(y, y * 2.0, y * 3.0)
    assert np.isnan(r["p"]) and "degenerate" in r["note"]


def test_delong_detects_difference():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(200), np.ones(200)]
    good = y + rng.normal(0, 0.5, 400)
    bad = rng.normal(0, 1, 400)  # noqa: E702
    r = delong_paired(y, good, bad)
    assert r["auc1"] > r["auc2"] and r["p"] < 0.001


def test_mcnemar_undefined_without_discordant_pairs():
    y = np.array([0, 1, 1])
    p = np.array([0, 1, 0])  # noqa: E702
    assert np.isnan(mcnemar_exact(y, p, p)["p"])
    assert mcnemar_exact(np.r_[np.ones(10)], np.r_[np.ones(10)], np.zeros(10))["p"] < 0.01


# ---- audit / paths -----------------------------------------------------------------------------
def test_audit_finds_known_confusion():
    # tp=500,tn=520 of 525/525 -> compute reported pcts, then audit must recover it
    c = {"tp": 500, "fn": 25, "tn": 520, "fp": 5}
    m = metrics_from_confusion(c)
    rep = {k: round(m[k] * 100, 2) for k in ("accuracy", "sensitivity", "specificity", "precision", "f1")}
    assert c in find_consistent_confusions(525, 525, rep)


def test_balanced_identity_gap():
    assert balanced_identity_gap({"accuracy": 95, "sensitivity": 94, "specificity": 96}) == 0


def test_safe_join_blocks_traversal(tmp_path):
    with pytest.raises(ValueError):
        safe_join(tmp_path, "../x.png")
    assert safe_join(tmp_path, "a/b.png").parent.name == "a"


def test_pandas_import_sanity():
    assert pd.__version__
