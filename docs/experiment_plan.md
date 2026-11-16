# Experiment plan (prespecified; none executed on real data yet)

Principles: one frozen split (hash recorded) shared by every compared model; test set touched once
per model at the end (`--no-test` for development); all selection on validation; seeds
{42, 43, 44, 45, 46} predefined, all reported, none cherry-picked; every run stores config, split hash,
environment and git commit.

| # | Experiment | Type | Command stem | Notes |
|---|---|---|---|---|
| E0 | Env + paper audit | — | `xact-tb audit-metrics`, `benchmark` | done |
| E1 | Dataset audit + split | — | `xact-tb audit-data` | needs data |
| E2 | Proposed: CNN+Swin → SVM | **primary reconstruction** | `train-proposed` | |
| E3 | Baseline Swin / CoAtNet-3 / MaxViT | direct baselines | `train-baseline --baseline …` | same split/preproc/aug/budget |
| A1 | CNN-only SVM, Swin-only SVM (reuse E2 networks) | **new**, post-hoc head swap | `branches` config | not retraining |
| A2 | Fused + neural head | **new** | planned | |
| A3 | no-CLAHE / no-BM3D / no-aug / no-class-weight | **new**, retraining | config `pre.*`, `train.w_tb=1` | |
| A4 | SVM kernels (linear/rbf/poly) | **new** | grid | |
| A5 | SVM on augmented views | **new** | planned | |
| X1 | Branch attenuation, Grad-CAM, Swin attention | post-hoc | `explainability` API | sample rule: first N correct + first N incorrect by image_id hash |
| S1 | Stats: group bootstrap if patient IDs else image bootstrap (labelled), DeLong, McNemar | | `evaluation/stats.py` | NaN p when degenerate |

Resource note: 6 GB GPU; CoAtNet-3 (~168 M) batch 16 may OOM → document any batch/accumulation change.
Not allowed: tuning on test, reporting best seed, hard-coding paper numbers.
