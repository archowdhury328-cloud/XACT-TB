# Artifact management policy

| Artifact | Where | In Git? |
|---|---|---|
| Source, configs, tests, docs, small tables (<1 MB) | repo | yes |
| Raw images, derived caches | `data/`, `cache/` (local) | **never** |
| Manifests with file hashes / any patient metadata | `manifests/private/` (local) | **never** |
| Per-image predictions | `runs/*/predictions/` (local) or private storage | no (may reveal test IDs) |
| Aggregate metrics JSON, figures w/o patient images, benchmark tables | `docs/` | yes |
| Checkpoints / bundles (`*.pt`, `*.joblib`) | release asset or external storage with SHA-256 | no in tree; release only if the data licence permits |
| Secrets | environment / credential manager | **never** (a token pasted in a chat is considered exposed and must be revoked) |

Code licence (MIT) does **not** extend to datasets, the paper's figures/text (CC BY-NC-ND 4.0), or
third-party pretrained weights (check timm/ImageNet terms before redistributing).
