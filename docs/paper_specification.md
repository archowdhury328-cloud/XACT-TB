# Paper specification (audited against the attached PDF)

Paper: Sikder et al., *XACT-TB an explainable hybrid CNN–swin transformer framework for tuberculosis
screening from chest X-ray images*, Discover Artificial Intelligence 6:1030 (2026),
doi:10.1007/s44163-026-01509-4. Licence of the article: CC BY-NC-ND 4.0 (no redistribution of adapted
figures/text; this repository contains none of the paper's figures).

Status legend: **S** = stated by paper, **A** = not stated (assumption in this repo), **C** = conflict/ambiguity in paper.

## Data (§3, Table 1)
| Item | Paper | Status |
|---|---|---|
| Source | Kaggle "Tuberculosis (TB) Chest X-ray Database" (Qatar Univ./Dhaka) | S |
| Classes | binary: TB vs Normal | S |
| Counts | 3,500 TB + 3,500 Normal = 7,000; split 4,900/1,050/1,050 (2,450/525/525 per class) | S |
| Public availability | "700 TB images are publicly accessible, 2,800 more via NIAID portal under data-sharing agreement" | **C**: the public Kaggle archive therefore cannot contain 3,500 TB images; Table 1 is not reproducible from public data alone |
| Patient separation | §3: "no common patients across splits … preventing patient level leakage" vs §3.1 & §5.4: "patient-level separation cannot be strictly enforced … image-level" | **C**: direct contradiction inside the paper; this repo treats it as **image-level** |
| Class balance | §3.5: "TB class was smaller than Normal in the original training split" vs Table 1: 2,450 each | **C** |
| Split seed / procedure | "fixed hold-out", image level | A (seed 42, stratified, exact-duplicate groups) |

## Preprocessing (§3.2–3.5)
| Item | Paper | Status |
|---|---|---|
| CLAHE | OpenCV, clip 2.0, tile 8×8 | S |
| BM3D | `bm3d` package; "λ2D = 2.7σ", σ = 0.05; τmatch "selected so small-distance patches grouped" (no value) | S/A. Installed `bm3d` 4.0.3 exposes `lambda_thr3d` (default None → profile value) and `tau_match`, **no λ2D**. 2.7 is the standard BM3D λ3D, so mapped to `lambda_thr3d=2.7` (deviation D3) |
| Order | normalise → CLAHE → BM3D | S |
| Resize position / interpolation | not stated | A (resize first, INTER_AREA; configurable) |
| Augmentation | train only, TB only, to 1:1; rot ±15°, translate ±10 %, scale 0.95–1.05, brightness/contrast ±10 %; no flips, no SMOTE | S |
| Class weights | Normal 1.0, TB 1.5 in BCE | S |
| Input | 224×224×3 (grayscale replicated) | S |

## Models (§4)
| Item | Paper | Status |
|---|---|---|
| Custom CNN | 4 × [Conv, Conv, SiLU, MaxPool] → GAP → SiLU → Dropout 0.4 → 256-D embedding (Fig. 2) | S (topology) |
| CNN channels / kernels / strides / norm | not stated | A: (32,64,128,256), k3, pad 1, no norm |
| Swin variant, pretrained source, frozen vs fine-tuned, feature dim | not stated | A: `swin_small_patch4_window7_224`, ImageNet init, fine-tuned, 768-D pooled |
| Fusion | features "fused" → SVM (Fig. 3 "Ensemble Classification Head SVM") | S; concatenation assumed (A) |
| SVM kernel/C/gamma, scaling, fit set | not stated | A: scaler(train), grid {linear, rbf}×C{0.1,1,10} chosen on val AUC, fit on un-augmented train |
| Training | Adam 1e-4, batch 16, 40 epochs, weighted BCE, grad clip τ = 0.2, ReduceLROnPlateau ×0.5 / patience 5, dropout 0.4 | S |
| Monitor, checkpoint rule, weight decay, clip norm type | not stated | A: val_loss, val_loss, 0, L2 |
| Baselines | Swin Transformer, CoAtNet-3, MaxViT (variants unspecified) | S/A: timm `swin_small…`, `coatnet_3_224`, `maxvit_tiny_tf_224` |
| Explainability | Grad-CAM on CNN pre-GAP map; Swin attention maps; post-hoc branch-feature attenuation (§4.4) | S; attenuation schedule not given (A) |

## Reported results (REFERENCE only; Tables 3–5)
See `src/xact_tb/reporting/paper_reference.py`. Arithmetic audit: `xact-tb audit-metrics`
(results in `docs/reproducibility_report.md`).

## Observations about the paper (not accusations; things a reproducer must know)
1. Table 3 values are not integer-consistent with 525+525 test images (audit, no matching confusion matrix for any row).
2. For a balanced test set, accuracy = (sens+spec)/2 exactly; reported rows differ by 0.09, 0.54, 0.625 and −0.03 pp.
3. Table 4: 95 % CIs for the baselines ([0.875, 1.0] etc.) are far wider than expected for n = 1,050; coarse step ROC curves in Figs. 6–7 also suggest a much smaller evaluated sample than Table 1.
4. Table 5: CoAtNet-3 (34.7 GFLOPs) reports lower latency (0.189 ms) than Swin (8.7 GFLOPs, 0.867 ms); batch size / timing method are not stated.
5. Figs. 4–5 show monotone per-epoch *accuracy/loss* curves for a model whose final classifier is an SVM (no epoch-wise SVM training exists); the network the curves belong to is not stated.
6. Fig. 3 labels the input image "CT Scan" for a chest-radiograph pipeline; §5.1 refers to "Figure A".
7. Table 6 compares across different datasets/protocols (external published numbers).
8. No code or checkpoints are linked; no author-linked repository was found at the time of writing (see `docs/assumptions_and_deviations.md`, section Provenance).
