# Assumptions and deviations

This repository is a **documented reconstruction**, not the authors' code. No original source was
available or verified. IDs are referenced by tests, configs and the traceability matrix.

## Provenance
The paper links only the Kaggle dataset (Data Availability). No code/checkpoint repository is cited.
A search for author-linked code has **not** been completed in this session (no network research
tool was used); treat original-code provenance as *unverified/absent*. Nothing here claims to be the original.

## Assumptions (paper silent) — all configurable
| ID | Setting | Value | Where |
|---|---|---|---|
| A1 | CNN channels/kernel/norm | (32,64,128,256), 3×3, pad 1, none | `models/cnn.py` |
| A2 | Swin variant | `swin_small_patch4_window7_224`, ImageNet-pretrained, fine-tuned | `models/backbones.py` |
| A3 | Baselines | `coatnet_3_224`, `maxvit_tiny_tf_224` (ImageNet init, fine-tuned) | same |
| A4 | Resize before CLAHE/BM3D, INTER_AREA | `PreprocessConfig` | `preprocessing/pipeline.py` |
| A5 | Scheduler monitor / checkpoint metric | val_loss (min) | `TrainConfig` |
| A6 | Clipping | global L2 norm, max 0.2 (paper: "threshold 0.2") | `TrainConfig` |
| A7 | Weight decay / AMP | 0 / off | `TrainConfig` |
| A8 | Fusion | concatenation (CNN 256 ‖ Swin 768) | `features/fusion_svm.py` |
| A9 | SVM | StandardScaler(train only) + SVC; grid {linear,rbf}×C∈{0.1,1,10}, gamma=scale, chosen by **val** AUC; fit on **un-augmented** train embeddings | same |
| A10 | Split | stratified, 70/15/15, seed 42, groups = exact-duplicate checksum | `data/splits.py` |
| A11 | Attenuation schedule | alpha ∈ {1,.75,.5,.25,0} in scaled space | `explainability/methods.py` |
| A12 | Swin attention | received-attention, mean over heads/blocks of stage 1 or 2, shift undone | same |
| A13 | Each branch trained independently with its own auxiliary linear head and BCE | per §4.3 "both branches optimised with BCE" |

Parameter-count matching (Swin-small ≈ 50 M, CoAtNet-3 ≈ 168 M, MaxViT-tiny ≈ 31 M vs Table 5) is
**supporting evidence only** for A2/A3; measured counts are reported by `xact-tb benchmark`.

## Deviations from the paper's literal text
| ID | Deviation | Reason |
|---|---|---|
| D1 | Table 1 (3,500 TB) cannot be built from the public archive (700 TB). Public-data runs use whatever the archive contains and are labelled `public_only`. | Data access (agreement-controlled NIAID images) |
| D2 | Splits are image-level (+ exact-duplicate grouping); never described as patient-independent. | Paper itself admits no patient IDs |
| D3 | BM3D: λ2D 2.7σ mapped to package `lambda_thr3d=2.7`; τmatch left at the package profile value. | `bm3d` 4.0.3 has no λ2D parameter |
| D4 | No epoch-wise "SVM accuracy" curve is produced. Learning curves are labelled as the auxiliary CNN/Swin classifiers. | SVM has no epochs |
| D5 | Latency/FLOPs are measured on this machine (CPU torch build; RTX 3050 6 GB present but torch is CPU-only here), not the paper's RTX 3060. | Hardware |
| D6 | Paper figures are not reproduced pixel-for-pixel; scripts regenerate figure *types* from real artifacts. | Integrity rule |

## Not done / pending (blockers stated plainly)
* No dataset was downloaded (Kaggle credentials absent) → **no trained model, no measured accuracy/AUC exists yet**.
* GPU training needs a CUDA torch build (`pip install torch --index-url https://download.pytorch.org/whl/cu126`); 6 GB VRAM is tight for CoAtNet-3 at batch 16 (use batch accumulation or AMP; any change must be recorded).
