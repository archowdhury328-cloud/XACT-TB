"""Integration tests on SYNTHETIC images (software correctness only)."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from xact_tb.data.manifest import build_manifest
from xact_tb.data.splits import make_splits, split_hash, verify_split
from xact_tb.explainability.methods import branch_attenuation, gradcam_cnn, swin_attention_map
from xact_tb.models.backbones import SMOKE, Backbone, BackboneConfig
from xact_tb.models.cnn import CustomCNN
from xact_tb.training.pipeline import make_synthetic_dataset


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    make_synthetic_dataset(root, n_per_class=20)
    return root


def test_manifest_labels_and_counts(synth):
    df, excl = build_manifest(synth)
    assert len(df) == 40 and len(excl) == 0
    assert set(df.label) == {0, 1} and (df.label == 1).sum() == 20
    assert df.image_id.is_unique


def test_corrupt_file_reported_not_dropped_silently(synth, tmp_path):
    import shutil

    shutil.copytree(synth, tmp_path / "d")
    (tmp_path / "d" / "Normal" / "bad.png").write_bytes(b"not an image")
    df, excl = build_manifest(tmp_path / "d")
    assert len(df) == 40 and excl["reason"].str.startswith("corrupt").sum() == 1


def test_duplicates_share_group_and_stay_in_one_split(tmp_path):
    (tmp_path / "Normal").mkdir()
    (tmp_path / "Tuberculosis").mkdir()  # noqa: E702
    rng = np.random.default_rng(0)
    for i in range(30):
        a = (rng.random((16, 16)) * 255).astype(np.uint8)
        Image.fromarray(a).save(tmp_path / "Normal" / f"n{i}.png")
        Image.fromarray(a).save(tmp_path / "Normal" / f"n{i}_copy.png")  # exact pixel duplicate
        Image.fromarray((rng.random((16, 16)) * 255).astype(np.uint8)).save(tmp_path / "Tuberculosis" / f"t{i}.png")
    df, _ = build_manifest(tmp_path)
    assert df.groupby("group_id").size().max() == 2
    sp = make_splits(df, seed=1)
    verify_split(sp)
    assert split_hash(sp) == split_hash(make_splits(df, seed=1))  # frozen/reproducible
    assert split_hash(sp) != split_hash(make_splits(df, seed=2))


def test_label_conflict_group_excluded(tmp_path):
    (tmp_path / "Normal").mkdir()
    (tmp_path / "Tuberculosis").mkdir()  # noqa: E702
    a = (np.random.default_rng(0).random((16, 16)) * 255).astype(np.uint8)
    Image.fromarray(a).save(tmp_path / "Normal" / "x.png")
    Image.fromarray(a).save(tmp_path / "Tuberculosis" / "x.png")
    df, excl = build_manifest(tmp_path)
    assert len(df) == 0 and (excl.reason == "label_conflict_duplicate_group").sum() == 2


def test_split_ratios_and_stratification(synth):
    df, _ = build_manifest(synth)
    sp = make_splits(df)
    t = sp.groupby(["split", "label"]).size().unstack()
    assert (t.loc["train"] == 14).all() and (t.loc["val"] == 3).all() and (t.loc["test"] == 3).all()


def test_gradcam_shape_range_finite():
    cnn = CustomCNN()
    cam = gradcam_cnn(cnn, torch.rand(2, 3, 224, 224))
    assert cam.shape == (2, 1, 224, 224) and torch.isfinite(cam).all()
    assert cam.min() >= 0 and cam.max() <= 1


def test_swin_attention_is_real_attention_and_shaped():
    bb = Backbone(BackboneConfig(name=SMOKE, pretrained=False)).eval()
    x = torch.rand(1, 3, 224, 224)
    m = swin_attention_map(bb, x, stage=2)  # 14x14 tokens, 4 windows, shifted block present
    assert m.shape == (1, 1, 224, 224) and torch.isfinite(m).all()
    assert m.min() == 0 and m.max() == pytest.approx(1.0)
    assert len(bb.trunk.layers[2].blocks) == 2 and any(bb.trunk.layers[2].blocks[1].shift_size)


def test_branch_attenuation_records_baseline_and_flips():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 6))
    w = rng.normal(size=6)  # noqa: E702
    rows = branch_attenuation(x, {"cnn": slice(0, 3), "swin": slice(3, 6)}, lambda z: z @ w)
    r1 = [r for r in rows if r["alpha"] == 1.0]
    assert all(r["mean_abs_score_change"] == 0 and r["n_class_flips"] == 0 for r in r1)
    assert any(r["n_class_flips"] > 0 for r in rows if r["alpha"] == 0.0)


def test_manifest_is_dataframe_roundtrip(synth, tmp_path):
    df, _ = build_manifest(synth)
    p = tmp_path / "m.csv"
    make_splits(df).to_csv(p, index=False)  # noqa: E702
    back = pd.read_csv(p, keep_default_na=False)
    verify_split(back)
