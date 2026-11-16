import json

import numpy as np
import pandas as pd

from xact_tb.models.backbones import SMOKE, Backbone, BackboneConfig
from xact_tb.models.cnn import CustomCNN
from xact_tb.reporting import costs, figures
from xact_tb.reporting.paper_reference import PAPER_TABLE3


def test_param_counts_and_size():
    m = CustomCNN()
    pc = costs.param_counts(m)
    assert pc["total"] == pc["trainable"] > 0
    assert costs.serialized_size_mb(m) > 0


def test_frozen_backbone_trainable_less_than_total():
    bb = Backbone(BackboneConfig(name=SMOKE, pretrained=False, freeze=True))
    pc = costs.param_counts(bb)
    assert pc["trainable"] < pc["total"]


def test_latency_reports_median_and_p95():
    r = costs.latency(CustomCNN(), "cpu", batch=1, size=64, warmup=1, iters=3)
    assert r["median_ms"] > 0 and r["p95_ms"] >= r["median_ms"]


def test_flops_fvcore_runs_on_small_model():
    r = costs.flops_fvcore(CustomCNN(), size=64)
    assert r["macs_g"] > 0 and "convention" in r


def test_figures_from_artifacts(tmp_path):
    h = tmp_path / "history.jsonl"
    h.write_text(
        "\n".join(
            json.dumps({"epoch": i, "train_loss": 1 / (i + 1), "val_loss": 1 / (i + 2), "val_acc": 0.5 + i / 10})
            for i in range(3)
        )
    )
    figures.learning_curves(h, tmp_path / "lc.png", "t")
    p = tmp_path / "p.csv"
    pd.DataFrame({"y_true": [0, 0, 1, 1], "score": [-1, 0.2, 0.1, 2]}).to_csv(p, index=False)
    figures.roc_plot(p, tmp_path / "roc.png", "t")
    figures.confusion_plot({"tp": 1, "fn": 1, "tn": 2, "fp": 0}, tmp_path / "cm.png", "t")
    assert all((tmp_path / f).stat().st_size > 0 for f in ("lc.png", "roc.png", "cm.png"))


def test_paper_reference_values_are_frozen_reference():
    assert PAPER_TABLE3["XACT-TB (proposed)"]["accuracy"] == 99.44
    assert np.isfinite(list(PAPER_TABLE3["MaxViT"].values())).all()
