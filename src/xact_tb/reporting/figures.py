"""Figures drawn only from saved artifacts (history.jsonl, prediction CSVs). No smoothing."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402


def read_history(path: str | Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows)


def learning_curves(history_jsonl: str | Path, out_png: str | Path, title: str) -> None:
    h = read_history(history_jsonl)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(h["epoch"], h["train_loss"], label="train (weighted BCE)")
    ax[0].plot(h["epoch"], h["val_loss"], label="val (weighted BCE)")
    ax[0].set(xlabel="epoch", ylabel="loss", title=title + " - loss")
    ax[1].plot(h["epoch"], h["val_acc"], label="val accuracy (aux. classifier)")
    ax[1].set(xlabel="epoch", ylabel="accuracy", title=title + " - val accuracy")
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def roc_plot(pred_csv: str | Path, out_png: str | Path, title: str) -> None:
    d = pd.read_csv(pred_csv)
    fpr, tpr, _ = roc_curve(d["y_true"], d["score"])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.step(fpr, tpr, where="post")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def confusion_plot(conf: dict, out_png: str | Path, title: str) -> None:
    m = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]])
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.imshow(m, cmap="Blues")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, int(v), ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal", "TB"],
        yticklabels=["Normal", "TB"],
        xlabel="predicted",
        ylabel="true",
        title=title,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
