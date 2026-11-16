# Measured computational cost (this repo's implemented models)

Command: `xact-tb benchmark --flops` (random init, 1-logit head, input 224×224×3).
Machine: Windows 11, CPU-only PyTorch 2.13.0+cpu, timm 1.0.28, Python 3.10.11, 15.7 GB RAM
(an RTX 3050 6 GB exists but the installed torch build has no CUDA). fp32.

| Model (timm id) | Params (total = trainable) | fp32 state_dict MB | MACs (G, fvcore, 1 MAC = 1 flop) | batch-1 CPU latency median / p95 ms |
|---|---|---|---|---|
| custom CNN (A1 widths) | 1,172,513 | 4.5 | 2.59 | 152 / 350 |
| Swin (`swin_small_patch4_window7_224`) | 48,838,027 | 186.4 | 8.77 | 2610 / 4420 |
| CoAtNet-3 (`coatnet_3_224`) | 165,430,801 | 631.4 | 36.56 | 2925 / 3912 |
| MaxViT (`maxvit_tiny_tf_224`) | 30,404,041 | 116.4 | 5.60 | 768 / 948 |

Caveats
* Latency was measured while other processes (tests) were running and is **noisy**; re-run on an idle machine (and on GPU) before using. 30 timed iterations, 5 warm-up.
* Counts exclude the 1000-class ImageNet head (replaced by 1 logit); they line up with the paper's Table 5 sizes for Swin (50 M / 8.7 G), CoAtNet-3 (168 M / 34.7 G) and MaxViT (31 M / 5.6 G), which is *supporting evidence only* for assumptions A2/A3.
* **Proposed model**: CNN (1.17 M) + Swin-small (48.8 M) ≈ **50.0 M** parameters under assumptions A1/A2, versus **75 M** in the paper's Table 5. The paper's CNN or Swin must therefore differ from this reconstruction (e.g. a much wider CNN or a larger Swin). Unresolved question Q1; not tuned to match.
* FLOPs for the SVM stage and preprocessing (CLAHE/BM3D) are not included; SVM storage is reported per run (`costs.svm_storage`).
