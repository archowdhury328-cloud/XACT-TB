# Reproducibility report (current state)

**Classification: partial implementation / documented reconstruction. No accuracy/AUC has been
measured on real data; no trained model exists.** Reasons: Kaggle credentials absent in the execution
environment (dataset not downloaded), torch build is CPU-only, and 3,500 TB images of the paper's
Table 1 are partly agreement-controlled.

## What was actually executed
| Check | Result |
|---|---|
| `pytest --cov` (synthetic data only) | 48 tests passed (4 min on CPU), 92 % line coverage of `xact_tb` (cli excluded) — see CI for latest |
| `ruff check` / `ruff format` | clean |
| `xact-tb benchmark --flops` | `docs/measured_cost_cpu.md` |
| `xact-tb audit-metrics` | below |

## Arithmetic audit of the paper's Table 3 (test set 525 TB + 525 Normal, Table 1)
Searched all (TP, TN) integer pairs; a row is "consistent" if one pair reproduces accuracy, sensitivity,
specificity, precision and F1 to two decimals.

| Model | consistent confusion matrices | F1 implied by reported P & R | F1 reported | acc − mean(sens, spec) (pp) |
|---|---|---|---|---|
| Swin Transformer | **0** | 90.74 | 90.67 | +0.090 |
| CoAtNet-3 | **0** | 97.03 | 97.14 | +0.540 |
| MaxViT | **0** | 97.94 | 97.79 | +0.625 |
| XACT-TB | **0** | 99.39 | 99.38 | −0.030 |

For a perfectly balanced test set accuracy must equal (sens+spec)/2; it does not in any row, and no
integer confusion matrix on 525+525 images reproduces any row. Possible explanations (all unverified):
a different test-set size/composition than Table 1, averaging over runs/folds, or transcription issues.
This is reported as found; nothing is adjusted. The paper values are therefore *not* usable as exact targets.

## Unresolved methodological questions
* Q1 Parameter count 75 M vs reconstruction ≈ 50 M: what are the CNN widths / Swin variant?
* Q2 Which split was evaluated (Table 1 vs a much smaller sample suggested by ROC step sizes and CI widths)?
* Q3 Patient-level separation: claimed (§3) vs denied (§3.1, §5.4).
* Q4 Which network produced Figs. 4–5 (per-epoch curves), given the final classifier is an SVM?
* Q5 SVM kernel/C/gamma, scaling, fit set; Swin variant/weights/freezing; checkpoint rule.
* Q6 Latency protocol (batch size, warm-up), esp. CoAtNet-3 < Swin latency.
* Q7 Why the dataset is described as 3,500 TB when only 700 are public.
* Q8 Availability of original code/checkpoints (none linked in the paper).

## Published vs measured
| Quantity | Published (reference) | Measured here | Comparable? |
|---|---|---|---|
| Accuracy / AUC / etc. | Table 3 | — (not run) | n/a |
| Params / MACs (Swin, CoAtNet-3, MaxViT) | 50/8.7, 168/34.7, 31/5.6 | 48.8/8.77, 165.4/36.56, 30.4/5.60 | yes (assumed variants) |
| Params XACT-TB | 75 M | ≈ 50.0 M (reconstruction) | **no** (Q1) |
| Latency | Table 5 (RTX 3060) | CPU, noisy | no |
