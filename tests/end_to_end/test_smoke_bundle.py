"""End-to-end on SYNTHETIC data: train 1 epoch -> SVM -> bundle -> reload -> infer; plus resume."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from xact_tb.data.dataset import ManifestDataset
from xact_tb.data.manifest import build_manifest
from xact_tb.data.splits import make_splits
from xact_tb.inference.bundle import load_bundle, predict_files
from xact_tb.models.backbones import SMOKE, BackboneConfig
from xact_tb.models.cnn import CNNConfig, CustomCNN
from xact_tb.preprocessing.augment import AugmentConfig
from xact_tb.preprocessing.pipeline import PreprocessConfig
from xact_tb.training.pipeline import RunSpec, make_synthetic_dataset, run_proposed
from xact_tb.training.trainer import TrainConfig, train_network


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    t = tmp_path_factory.mktemp("e2e")
    make_synthetic_dataset(t / "d", n_per_class=16, size=48)
    df, _ = build_manifest(t / "d")
    make_splits(df).to_csv(t / "m.csv", index=False)
    return t


def _spec(t, out):
    return RunSpec(
        str(t / "m.csv"),
        str(t / "d"),
        str(t / out),
        pre=PreprocessConfig(size=224, bm3d=False),
        train=TrainConfig(epochs=1, batch_size=8),
        swin=BackboneConfig(name=SMOKE, pretrained=False),
    )


def test_end_to_end_and_bundle_roundtrip(env):
    r = run_proposed(_spec(env, "run"))
    assert {"train", "val", "test"} <= set(r) and r["branches"] == ["cnn", "swin"]
    pred = pd.read_csv(env / "run" / "predictions" / "test.csv")
    assert set(pred.columns) >= {"image_id", "y_true", "score", "split_hash", "seed"}
    # metrics must be reproducible from the saved sample-level predictions
    from xact_tb.evaluation.metrics import compute_metrics

    assert compute_metrics(pred.y_true, pred.score)["accuracy"] == pytest.approx(r["test"]["accuracy"])
    b = load_bundle(env / "run" / "bundle")
    img = next((env / "d" / "Tuberculosis").glob("*.png"))
    out = predict_files(b, [img])[0]
    assert out["predicted_class"] in ("Normal", "TB") and out["calibrated_probability"] is None
    # bundle score must equal the score from the pipeline run for the same image
    rel = img.relative_to(env / "d").as_posix()
    m = pd.read_csv(env / "m.csv")
    pid = m[m.rel_path == rel].image_id.iloc[0]
    allp = pd.concat([pd.read_csv(env / "run" / "predictions" / f"{s}.csv") for s in ("train", "val", "test")])
    assert out["decision_score"] == pytest.approx(float(allp[allp.image_id == pid].score.iloc[0]), abs=1e-3)


def test_bundle_rejects_tampering(env):
    run_proposed(_spec(env, "run2"))
    w = env / "run2" / "bundle" / "weights.pt"
    w.write_bytes(w.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(env / "run2" / "bundle")


def test_predict_rejects_bad_extension(env, tmp_path):
    run_proposed(_spec(env, "run3"))
    b = load_bundle(env / "run3" / "bundle")
    bad = tmp_path / "x.txt"
    bad.write_text("hi")  # noqa: E702
    with pytest.raises(ValueError):
        predict_files(b, [bad])


def test_checkpoint_resume_continues_history(env):
    df = pd.read_csv(env / "m.csv", keep_default_na=False)
    pre = PreprocessConfig(size=64, bm3d=False)
    tr = ManifestDataset(df[df.split == "train"], str(env / "d"), pre, AugmentConfig(), 0)
    va = DataLoader(ManifestDataset(df[df.split == "val"], str(env / "d"), pre), batch_size=8)
    small = CNNConfig(channels=(4, 8), embed_dim=8)
    out = env / "resume"
    train_network(CustomCNN(small), tr, va, TrainConfig(epochs=1, batch_size=8), out, tag="cnn")
    h2 = train_network(CustomCNN(small), tr, va, TrainConfig(epochs=2, batch_size=8), out, tag="cnn")
    assert [r["epoch"] for r in h2] == [0, 1]
    lines = (out / "history.jsonl").read_text().strip().splitlines()
    assert [json.loads(x)["epoch"] for x in lines] == [0, 1]
    with pytest.raises(ValueError):  # incompatible checkpoint tag is a clear error
        train_network(CustomCNN(small), tr, va, TrainConfig(epochs=2, batch_size=8), out, tag="other")


def test_effective_counts_logged_each_epoch(env):
    h = [json.loads(x) for x in (env / "resume" / "history.jsonl").read_text().splitlines()]
    assert all("effective_counts" in r and "lr" in r for r in h)
    assert torch.cuda.is_available() in (True, False) and np.isfinite(h[0]["val_loss"])
