"""End-to-end runs: proposed model (CNN+Swin -> SVM) and standalone baselines.

Order: train networks (val used for selection) -> freeze -> deterministic embeddings (train
originals only, no augmented views) -> train-only scaler -> SVM grid on VAL -> predictions.
Test predictions are produced once, at the end, and are never used for any selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from xact_tb.data.dataset import ManifestDataset
from xact_tb.data.splits import split_hash, verify_split
from xact_tb.evaluation.metrics import compute_metrics
from xact_tb.features.fusion_svm import concat_blocks, extract_block, fit_svm, fuse
from xact_tb.inference.bundle import FEATURE_ORDER, save_bundle
from xact_tb.models.backbones import Backbone, BackboneConfig, build_baseline
from xact_tb.models.cnn import CNNConfig, CustomCNN
from xact_tb.preprocessing.augment import AugmentConfig
from xact_tb.preprocessing.pipeline import PreprocessConfig
from xact_tb.training.trainer import TrainConfig, load_best, predict_logits, train_network
from xact_tb.utils.common import environment_record, write_json


@dataclass
class RunSpec:
    manifest_csv: str
    data_root: str
    out_dir: str
    pre: PreprocessConfig = PreprocessConfig()
    train: TrainConfig = TrainConfig()
    aug: AugmentConfig = AugmentConfig()
    cnn: CNNConfig = CNNConfig()
    swin: BackboneConfig = BackboneConfig()
    device: str = "cpu"
    evaluate_test: bool = True
    branches: tuple[str, ...] = ("cnn", "swin")  # SVM feature blocks (post-hoc ablation if subset)
    run_name: str = "proposed"


def _datasets(spec: RunSpec):
    df = pd.read_csv(spec.manifest_csv, keep_default_na=False)
    verify_split(df)
    mk = lambda s, aug=None: ManifestDataset(df[df.split == s], spec.data_root, spec.pre, aug, spec.train.seed)  # noqa: E731
    return df, mk("train", spec.aug), mk("train"), mk("val"), mk("test")


def _loader(ds, spec: RunSpec) -> DataLoader:
    return DataLoader(ds, batch_size=spec.train.batch_size, shuffle=False, num_workers=spec.train.num_workers)


def _pred_frame(ids, y, z, split, name, spec, shash) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ids,
            "y_true": y,
            "score": z,
            "y_pred": (z > 0).astype(int),
            "split": split,
            "model": name,
            "seed": spec.train.seed,
            "split_hash": shash,
        }
    )


def run_proposed(spec: RunSpec) -> dict:
    out = Path(spec.out_dir)
    df, tr_aug, tr_plain, va, te = _datasets(spec)
    shash = split_hash(df)
    dev = torch.device(spec.device)
    write_json(out / "run_config.json", {"spec": asdict(spec), "split_hash": shash, "env": environment_record()})
    cnn = CustomCNN(spec.cnn)
    swin = Backbone(spec.swin)
    for net, tag in ((cnn, "cnn"), (swin, "swin")):
        train_network(net, tr_aug, _loader(va, spec), spec.train, out / tag, dev, tag=tag)
        load_best(net, out / tag, dev)
        net.eval().to(dev)
    embed = {"cnn": cnn.embed, "swin": swin.embed}
    feats: dict[str, dict] = {s: {} for s in ("train", "val", "test")}
    for s, ds in (("train", tr_plain), ("val", va), ("test", te)):
        if s == "test" and not spec.evaluate_test:
            continue
        for b in ("cnn", "swin"):
            feats[s][b] = extract_block(embed[b], _loader(ds, spec), dev)
    fused = {s: fuse(feats[s]) for s in feats if feats[s]}
    order = [b for b in FEATURE_ORDER if b in spec.branches]
    xs = {s: concat_blocks(fused[s], order) for s in fused}
    scaler, svm, info = fit_svm(
        xs["train"], fused["train"].labels, xs["val"], fused["val"].labels, seed=spec.train.seed
    )
    result: dict = {"svm": info, "branches": order, "split_hash": shash}
    for s in fused:
        z = svm.decision_function(scaler.transform(xs[s]))
        pf = _pred_frame(fused[s].ids, fused[s].labels, z, s, spec.run_name, spec, shash)
        (out / "predictions").mkdir(parents=True, exist_ok=True)
        pf.to_csv(out / "predictions" / f"{s}.csv", index=False)
        result[s] = compute_metrics(fused[s].labels, z)
    write_json(out / "metrics.json", result)
    if order == FEATURE_ORDER:
        save_bundle(
            out / "bundle", cnn.cpu(), swin.cpu(), scaler, svm, spec.pre, {"split_hash": shash, "svm": info["selected"]}
        )
    return result


def run_baseline(key: str, spec: RunSpec, pretrained: bool = True) -> dict:
    out = Path(spec.out_dir)
    df, tr_aug, _, va, te = _datasets(spec)
    shash = split_hash(df)
    dev = torch.device(spec.device)
    net = build_baseline(key, pretrained)
    write_json(
        out / "run_config.json",
        {"spec": asdict(spec), "baseline": key, "split_hash": shash, "env": environment_record()},
    )
    train_network(net, tr_aug, _loader(va, spec), spec.train, out, dev, tag=key)
    load_best(net, out, dev)
    result: dict = {"baseline": key, "split_hash": shash}
    for s, ds in (("val", va), ("test", te)):
        if s == "test" and not spec.evaluate_test:
            continue
        ids, y, z = predict_logits(net.to(dev), _loader(ds, spec), dev)
        (out / "predictions").mkdir(parents=True, exist_ok=True)
        _pred_frame(ids, y, z, s, key, spec, shash).to_csv(out / "predictions" / f"{s}.csv", index=False)
        result[s] = compute_metrics(y, z)
    write_json(out / "metrics.json", result)
    return result


def make_synthetic_dataset(root: str | Path, n_per_class: int = 24, size: int = 64, seed: int = 0) -> None:
    """SYNTHETIC images for software tests ONLY. Written under <root>/Normal and <root>/Tuberculosis.
    Never use for research results; a SYNTHETIC_DATA_DO_NOT_REPORT marker file is added."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    root = Path(root)
    for name, bright in (("Normal", 0.35), ("Tuberculosis", 0.55)):
        (root / name).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            a = np.clip(rng.normal(bright, 0.15, (size, size)), 0, 1)
            Image.fromarray((a * 255).astype(np.uint8)).save(root / name / f"{name}-{i}.png")
    (root / "SYNTHETIC_DATA_DO_NOT_REPORT").write_text("software tests only\n")
