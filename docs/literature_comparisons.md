# Literature comparison audit

Classification of every comparison/related-work entry in the paper.

| Entry (paper ref) | Class | Reproducible here? |
|---|---|---|
| Swin Transformer (Table 3) | Directly evaluated baseline (variant unstated) | Yes: `train-baseline --baseline swin` |
| CoAtNet-3 (Table 3) | Directly evaluated baseline | Yes: `coatnet3` |
| MaxViT (Table 3) | Directly evaluated baseline (variant unstated) | Yes: `maxvit` |
| ViT + ConvStem + PEG + CLAHE [37], 99.28 % | External published result, "CXR TB dataset" (split/protocol not verified) | Not attempted; original paper/code not inspected |
| PaliGemma-CXR [38], 98.95 % | External; **multitask** VLM (5 tasks, needs segmentation/VQA/report annotations) | Not reproducible as a plain classifier; do not relabel one as PaliGemma-CXR |
| Inception-v3 + preprocessing [39], 98.78 % | External published result | Optional reimplementation possible; not done |
| Andayani PNN [24], Duong EfficientNet+ViT [23], TB-UNet hybrid [25], hybrid descriptors [26], CNN–ViT two-step [27], handcrafted SVM [28] | Related-work citations, not experiments of this paper | No |
| Kırdar MANN-BiLSTM [29] | Related-work citation on **bearing fault** detection, not TB | No (domain-unrelated) |
| Yanar et al. [30–32] | Related-work citations (other chest tasks) | No |

Table 6 compares models on different datasets/splits; numbers are literature-only and must not enter
the measured-results columns. A precise missing-dependency record for each external method (dataset
split, annotations, checkpoints) is **pending** a read of those papers.
