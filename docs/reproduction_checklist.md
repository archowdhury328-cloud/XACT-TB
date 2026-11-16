# Reproduction checklist / traceability matrix

Status: ✅ implemented + unit/integration tested · 🟡 implemented, needs real data to verify · ⛔ blocked · ❓ unresolved in paper.
"Verified" below means *software behaviour tested on synthetic data*; **no research result has been measured**.

| Paper requirement | Ref | Config / module | Verification | Status |
|---|---|---|---|---|
| Binary labels, Normal=0 TB=1 | §3 | `utils.common` | `test_manifest_labels_and_counts` | ✅ |
| 4,900/1,050/1,050 (70/15/15), stratified | Table 1 | `data/splits.py` | `test_split_ratios_and_stratification` | ✅ (data ⛔ D1) |
| Hold-out, no leakage, split frozen | §3.1 | `split_hash`, `verify_split` | `test_duplicates_share_group…` | ✅ |
| Patient-level separation | §3 vs §3.1 | — | n/a | ❓ (D2) |
| CLAHE clip 2.0 / 8×8 | §3.3 | `PreprocessConfig` | `test_preprocess_range_shape_dtype_and_determinism` | ✅ |
| BM3D σ=0.05, λ 2.7 | §3.4 | `apply_bm3d` | `test_bm3d_runs_and_is_deterministic_small` | 🟡 (D3) |
| Order: CLAHE→BM3D | §3.4 | `preprocess_array` | same | ✅ |
| Train-only TB-only aug, 1:1, no flips | §3.5 | `augment.py`, `ManifestDataset` | `test_epoch_plan_*`, `test_augmentation_has_no_flip…` | ✅ |
| Weighted BCE 1.0/1.5 | Eq. 2 | `training/loss.py` | `test_weighted_bce_matches_hand_computation` | ✅ |
| Grad clip 0.2 | §4.3 | `TrainConfig`, after unscale | code review; `test_checkpoint_resume…` runs it | ✅ (A6) |
| Adam 1e-4, bs16, 40 ep, ROP 0.5/5 | §4.5, Table 2 | `TrainConfig` defaults | trainer tests | ✅ |
| CNN 224×224×3 → 256-D, dropout 0.4 | §4.1, Fig. 2 | `models/cnn.py` | `test_cnn_shapes_and_embedding_dim` | ✅ (A1) |
| Swin branch + embeddings | §4.2 | `models/backbones.py` | smoke model tests | 🟡 (A2, pretrained needs download) |
| Concatenate → SVM, frozen features | §4.2–4.3 | `features/fusion_svm.py` | `test_fusion_*`, `test_svm_scaler_fit_on_train_only…` | ✅ (A8/A9) |
| Grad-CAM (CNN pre-GAP) | §4.1 | `explainability.gradcam_cnn` | `test_gradcam_shape_range_finite` | ✅ (branch-specific) |
| Swin attention maps | §4.2 | `swin_attention_map` | `test_swin_attention_is_real_attention…` | ✅ |
| Branch attenuation analysis | §4.4 | `branch_attenuation` | `test_branch_attenuation_records_baseline_and_flips` | ✅ (A11) |
| Baselines Swin/CoAtNet-3/MaxViT | §4.6 | `train-baseline` | instantiation in `benchmark` | 🟡 (no data) |
| Metrics Acc/Sens/Spec/Prec/F1/AUC | Table 3 | `evaluation/metrics.py` | `test_metrics_known_confusion` | ✅ |
| AUC CI & p-values | Table 4 | `evaluation/stats.py` | bootstrap/DeLong/McNemar tests | ✅ |
| Params/FLOPs/size/latency | Table 5 | `reporting/costs.py`, `xact-tb benchmark` | measured, see `docs/measured_cost_cpu.md` | ✅ measured (CPU) |
| Learning curves / ROC / confusion | Figs. 4–8 | `reporting/figures.py` | `test_figures_from_artifacts` | 🟡 |
| Table 3 arithmetic consistency | Table 3 | `evaluation/audit.py` | `xact-tb audit-metrics` | ✅ findings in report |
| Literature rows (Table 6, related work) | §5.1, §2 | `docs/literature_comparisons.md` | classification only | ✅ |
| Inference bundle | — | `inference/bundle.py` | `test_end_to_end_and_bundle_roundtrip` | ✅ |
| Real-data run, final metrics | §5 | — | — | ⛔ no dataset access |
