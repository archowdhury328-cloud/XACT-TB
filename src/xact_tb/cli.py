"""xact-tb command line. See docs/command_reference.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _cmd_audit_data(a):
    from xact_tb.data.manifest import build_manifest
    from xact_tb.data.splits import make_splits, split_hash, verify_split

    df, excl = build_manifest(a.root)
    if df.empty:
        sys.exit(f"no labelled images found under {a.root} (expect Normal/ and Tuberculosis/ dirs)")
    df = make_splits(df, seed=a.seed)
    verify_split(df)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "manifest.csv", index=False)
    excl.to_csv(out / "exclusions.csv", index=False)
    summary = {
        "n_images": len(df),
        "n_excluded": len(excl),
        "split_hash": split_hash(df),
        "per_split_class": df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(),
        "patient_ids_available": bool((df["patient_id"].astype(str).str.len() > 0).any()),
        "note": "image-level split: NOT patient-independent unless patient_id is populated",
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


def _spec(a):
    import yaml

    from xact_tb.models.backbones import BackboneConfig
    from xact_tb.preprocessing.pipeline import PreprocessConfig
    from xact_tb.training.pipeline import RunSpec
    from xact_tb.training.trainer import TrainConfig

    cfg = yaml.safe_load(Path(a.config).read_text()) if a.config else {}
    unknown = set(cfg) - {"pre", "train", "swin", "run_name"}
    if unknown:
        sys.exit(f"unknown config keys: {sorted(unknown)}")
    return RunSpec(
        manifest_csv=a.manifest,
        data_root=a.data_root,
        out_dir=a.out,
        device=a.device,
        pre=PreprocessConfig(**cfg.get("pre", {})),
        train=TrainConfig(**cfg.get("train", {})),
        swin=BackboneConfig(**cfg.get("swin", {})),
        run_name=cfg.get("run_name", "proposed"),
        evaluate_test=not a.no_test,
    )


def _cmd_train_proposed(a):
    from xact_tb.training.pipeline import run_proposed

    print(json.dumps(run_proposed(_spec(a)), indent=2, default=str))


def _cmd_train_baseline(a):
    from xact_tb.training.pipeline import run_baseline

    print(json.dumps(run_baseline(a.baseline, _spec(a), pretrained=not a.no_pretrained), indent=2, default=str))


def _cmd_audit_metrics(a):
    from xact_tb.evaluation.audit import audit_row
    from xact_tb.reporting.paper_reference import PAPER_TABLE3, TEST_PER_CLASS

    rows = [audit_row(m, TEST_PER_CLASS, TEST_PER_CLASS, v) for m, v in PAPER_TABLE3.items()]
    print(pd.DataFrame(rows).to_string(index=False))


def _cmd_predict(a):
    from xact_tb.inference.bundle import load_bundle, predict_files

    b = load_bundle(a.bundle, device=a.device)
    paths = sorted(p for p in Path(a.input).iterdir()) if Path(a.input).is_dir() else [Path(a.input)]
    print(json.dumps(predict_files(b, paths), indent=2))


def _cmd_benchmark(a):
    from xact_tb.models.backbones import BASELINES, build_baseline
    from xact_tb.models.cnn import CustomCNN
    from xact_tb.reporting import costs

    models = {"custom_cnn": CustomCNN()} | {k: build_baseline(k, pretrained=False) for k in BASELINES}
    rows = []
    for k, m in models.items():
        r = {"model": k, **costs.param_counts(m), "size_mb_fp32": round(costs.serialized_size_mb(m), 1)}
        if a.flops:
            r["macs_g"] = round(costs.flops_fvcore(m)["macs_g"], 2)
        r.update({f"b1_{k2}": v for k2, v in costs.latency(m, a.device, 1).items() if k2 in ("median_ms", "p95_ms")})
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))


def _cmd_smoke(a):
    """Synthetic end-to-end run (software test only; results are meaningless)."""
    import tempfile

    from xact_tb.data.manifest import build_manifest
    from xact_tb.data.splits import make_splits
    from xact_tb.models.backbones import SMOKE, BackboneConfig
    from xact_tb.preprocessing.pipeline import PreprocessConfig
    from xact_tb.training.pipeline import RunSpec, make_synthetic_dataset, run_proposed
    from xact_tb.training.trainer import TrainConfig

    with tempfile.TemporaryDirectory() as t:
        make_synthetic_dataset(Path(t) / "d")
        df, _ = build_manifest(Path(t) / "d")
        make_splits(df).to_csv(Path(t) / "m.csv", index=False)
        spec = RunSpec(
            str(Path(t) / "m.csv"),
            str(Path(t) / "d"),
            str(Path(t) / "o"),
            pre=PreprocessConfig(size=224, bm3d=False),
            train=TrainConfig(epochs=1, batch_size=8),
            swin=BackboneConfig(name=SMOKE, pretrained=False),
        )
        r = run_proposed(spec)
        print("SYNTHETIC SMOKE OK; test acc (meaningless):", r["test"]["accuracy"])


def main(argv=None):
    p = argparse.ArgumentParser(prog="xact-tb", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("audit-data", help="build manifest, exclusions, frozen split")
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)  # noqa: E702
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(f=_cmd_audit_data)  # noqa: E702
    for name, fn in (("train-proposed", _cmd_train_proposed), ("train-baseline", _cmd_train_baseline)):
        s = sub.add_parser(name)
        s.add_argument("--manifest", required=True)
        s.add_argument("--data-root", required=True)  # noqa: E702
        s.add_argument("--out", required=True)
        s.add_argument("--config")  # noqa: E702
        s.add_argument("--device", default="cpu")
        s.add_argument("--no-test", action="store_true", help="skip test evaluation (dev runs)")
        if name == "train-baseline":
            s.add_argument("--baseline", required=True, choices=["swin", "coatnet3", "maxvit", "smoke_swin"])
            s.add_argument("--no-pretrained", action="store_true")
        s.set_defaults(f=fn)
    s = sub.add_parser("audit-metrics", help="arithmetic audit of the paper's reported Table 3")
    s.set_defaults(f=_cmd_audit_metrics)
    s = sub.add_parser("predict")
    s.add_argument("--bundle", required=True)  # noqa: E702
    s.add_argument("--input", required=True)
    s.add_argument("--device", default="cpu")  # noqa: E702
    s.set_defaults(f=_cmd_predict)
    s = sub.add_parser("benchmark")
    s.add_argument("--device", default="cpu")  # noqa: E702
    s.add_argument("--flops", action="store_true")
    s.set_defaults(f=_cmd_benchmark)  # noqa: E702
    s = sub.add_parser("smoke", help="synthetic end-to-end software check")
    s.set_defaults(f=_cmd_smoke)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
