"""Versioned inference bundle: CNN + Swin weights, scaler, SVM, label map, preprocessing config.

Security: bundles contain joblib/pickle and torch state, i.e. executable-capable formats. Load only
bundles you built yourself or whose SHA-256 you verified out-of-band (``expected_sha256``).
Torch weights are loaded with ``weights_only=True``. Nothing is trained or downloaded at inference.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import torch

from xact_tb.features.fusion_svm import concat_blocks
from xact_tb.models.backbones import Backbone, BackboneConfig
from xact_tb.models.cnn import CNNConfig, CustomCNN
from xact_tb.preprocessing.pipeline import PreprocessConfig, preprocess_file
from xact_tb.utils.common import LABEL_NAMES, sha256_file

BUNDLE_VERSION = "1"
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FEATURE_ORDER = ["cnn", "swin"]


def save_bundle(
    path, cnn: CustomCNN, swin: Backbone, scaler, svm, pre: PreprocessConfig, extra: dict | None = None
) -> Path:
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"cnn": cnn.state_dict(), "swin": swin.state_dict()}, d / "weights.pt")
    joblib.dump({"scaler": scaler, "svm": svm}, d / "classifier.joblib")
    pre_d = asdict(pre)
    pre_d["cache_dir"] = None
    meta = {
        "bundle_version": BUNDLE_VERSION,
        "feature_order": FEATURE_ORDER,
        "label_map": LABEL_NAMES,
        "preprocess": pre_d,
        "cnn": asdict(CNNConfig(**{**asdict(cnn.cfg)})),
        "swin": asdict(swin.cfg),
        "sha256": {f: sha256_file(d / f) for f in ("weights.pt", "classifier.joblib")},
        "research_use_only": True,
        **(extra or {}),
    }
    (d / "bundle.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return d


def load_bundle(path, expected_sha256: dict[str, str] | None = None, device="cpu"):
    d = Path(path)
    meta = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
    if meta.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(f"unsupported bundle version {meta.get('bundle_version')}")
    for f, h in meta["sha256"].items():
        if sha256_file(d / f) != h:
            raise ValueError(f"{f}: checksum mismatch with bundle.json")
        if expected_sha256 and f in expected_sha256 and expected_sha256[f] != h:
            raise ValueError(f"{f}: checksum differs from externally supplied value")
    c = meta["cnn"]
    cnn = CustomCNN(CNNConfig(**{**c, "channels": tuple(c["channels"])}))
    swin = Backbone(BackboneConfig(**{**meta["swin"], "pretrained": False}))
    w = torch.load(d / "weights.pt", map_location=device, weights_only=True)
    cnn.load_state_dict(w["cnn"])
    swin.load_state_dict(w["swin"])
    clf = joblib.load(d / "classifier.joblib")
    pre = PreprocessConfig(**meta["preprocess"])
    return {
        "meta": meta,
        "cnn": cnn.eval().to(device),
        "swin": swin.eval().to(device),
        "scaler": clf["scaler"],
        "svm": clf["svm"],
        "pre": pre,
        "device": device,
    }


@torch.no_grad()
def predict_files(bundle: dict, paths: list[str | Path]) -> list[dict]:
    out = []
    for p in paths:
        p = Path(p)
        if p.suffix.lower() not in ALLOWED_EXT or not p.is_file():
            raise ValueError(f"unsupported or missing image file: {p.name}")
        arr = preprocess_file(p, bundle["pre"])
        x = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(bundle["device"])
        feats = {"cnn": bundle["cnn"].embed(x).cpu().numpy(), "swin": bundle["swin"].embed(x).cpu().numpy()}
        z = concat_blocks(type("F", (), {"blocks": feats})(), bundle["meta"]["feature_order"])
        score = float(bundle["svm"].decision_function(bundle["scaler"].transform(z))[0])
        pred = int(score > 0)
        out.append(
            {
                "file": p.name,
                "predicted_class": LABEL_NAMES[pred],
                "decision_score": score,
                "calibrated_probability": None,
                "model_version": bundle["meta"]["bundle_version"],
                "note": "research software; decision_score is an SVM margin, not a probability",
            }
        )
    return out
