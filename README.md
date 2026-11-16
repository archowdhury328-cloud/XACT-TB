# XACT-TB — documented reconstruction

Research software reconstructing the pipeline of

> Sikder T.R., Chy M.A.R., Hossain M.J., Uddin S.M.M., Faruk M.I., Manik M.M.T.G., Adnan M.
> *XACT-TB an explainable hybrid CNN–swin transformer framework for tuberculosis screening from chest X-ray images.*
> Discover Artificial Intelligence 6:1030 (2026). doi:10.1007/s44163-026-01509-4

Repository maintainer: Md Asikur Rahman Chy (Dept. of Business Administration, International American University, Los Angeles, CA, USA).

> **Status: partial implementation / documented reconstruction.** This is *not* the authors' code
> (none is linked in the paper). **No model has been trained on real data and no accuracy/AUC is claimed.**
> Not a medical device; not clinically validated; binary TB-vs-Normal research task only.
> See [docs/reproducibility_report.md](docs/reproducibility_report.md).

## What is implemented
Custom CNN (+ Grad-CAM) and Swin branch (+ real window-attention maps) → frozen embeddings → concatenation →
train-only scaler → SVM; direct baselines (Swin, CoAtNet-3, MaxViT via timm); CLAHE + BM3D preprocessing;
leakage-safe training-only TB augmentation; weighted BCE; resumable training; group-aware frozen splits;
metrics from saved sample-level predictions; bootstrap / DeLong / McNemar; arithmetic audit of the paper's tables;
cost benchmark; versioned inference bundle; CLI. Every unstated paper detail is a labelled, configurable
assumption: [docs/assumptions_and_deviations.md](docs/assumptions_and_deviations.md).

## Install
```bash
python -m venv .venv && . .venv/Scripts/activate      # Python 3.10-3.12
pip install -e ".[dev]"
# GPU: install a CUDA build of torch first, e.g. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pytest --cov && xact-tb smoke
```

## Data
Kaggle *Tuberculosis (TB) Chest X-ray Database* (tawsifurrahman). Accept its terms and download with your own
Kaggle credentials (`~/.kaggle/kaggle.json`). The public archive holds ~700 TB images; the paper's 3,500 TB
require NIAID agreement data — runs on the public archive alone are `public_only` and not comparable to Table 1.
Patient IDs are unavailable, so splits are **image-level** (exact-duplicate grouped), never patient-independent.
Details: [docs/data_statement.md](docs/data_statement.md), [docs/artifact_policy.md](docs/artifact_policy.md).

## Use
See [docs/command_reference.md](docs/command_reference.md) (audit data, train proposed model, baselines, predict, benchmark).

## Documentation
[paper spec](docs/paper_specification.md) · [assumptions/deviations](docs/assumptions_and_deviations.md) ·
[traceability checklist](docs/reproduction_checklist.md) · [experiment plan](docs/experiment_plan.md) ·
[literature comparisons](docs/literature_comparisons.md) · [measured cost](docs/measured_cost_cpu.md) ·
[model card](docs/model_card.md)

## Hardware
CPU smoke tests need ~4 GB RAM. Full runs: GPU ≥ 12 GB recommended (CoAtNet-3 at batch 16 will not fit in 6 GB without changes that must be recorded).

## Licence
Code: MIT (see LICENSE). Does not cover the dataset, the paper's text/figures (CC BY-NC-ND 4.0) or third-party pretrained weights.
Cite via [CITATION.cff](CITATION.cff) (cites the paper; software citation is secondary).
