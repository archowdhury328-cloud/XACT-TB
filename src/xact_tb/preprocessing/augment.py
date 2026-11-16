"""Training-only, TB-only augmentation and the epoch plan that realises a 1:1 effective balance.

Augmentation (paper §3.5): rotation ±15°, translation ±10 %, scale 0.95-1.05, brightness and
contrast ±10 %; NO horizontal flips; no SMOTE/ADASYN.
Balancing is done in exactly ONE place (``build_epoch_plan``): extra augmented TB views are added
until counts match; originals are always kept unaugmented. Do not also use a weighted sampler.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torchvision.transforms import v2

from xact_tb.utils.common import NORMAL, TB


@dataclass(frozen=True)
class AugmentConfig:
    rotation: float = 15.0
    translate: float = 0.10
    scale: tuple[float, float] = (0.95, 1.05)
    brightness: float = 0.10
    contrast: float = 0.10


def build_augmentation(cfg: AugmentConfig) -> v2.Compose:
    """Operates on float tensors in [0,1], shape (C,H,W). No flips by construction."""
    return v2.Compose(
        [
            v2.RandomAffine(degrees=cfg.rotation, translate=(cfg.translate, cfg.translate), scale=cfg.scale),
            v2.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast),
            v2.Lambda(lambda t: t.clamp(0, 1)),
        ]
    )


def build_epoch_plan(labels: np.ndarray, rng: np.random.Generator) -> list[tuple[int, bool]]:
    """Return [(dataset_index, augmented?)]. Minority-class (TB or Normal) is only augmented if TB.

    Augmentation is applied only to TB images (paper). If Normal is the minority, no augmented
    Normal samples are created and the imbalance is left to the loss weights; this is logged by
    the caller via ``plan_counts``.
    """
    labels = np.asarray(labels)
    plan = [(int(i), False) for i in range(len(labels))]
    n_tb, n_norm = int((labels == TB).sum()), int((labels == NORMAL).sum())
    deficit = n_norm - n_tb
    if deficit > 0 and n_tb > 0:
        tb_idx = np.flatnonzero(labels == TB)
        extra = rng.choice(tb_idx, size=deficit, replace=True)
        plan.extend((int(i), True) for i in extra)
    rng.shuffle(plan)
    return plan


def plan_counts(plan: list[tuple[int, bool]], labels: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels)
    c = {"Normal": 0, "TB": 0, "TB_augmented_views": 0}
    for i, aug in plan:
        if labels[i] == TB:
            c["TB"] += 1
            c["TB_augmented_views"] += int(aug)
        else:
            c["Normal"] += 1
    return c


def apply_to_tensor(aug: v2.Compose, x: torch.Tensor, generator_seed: int | None = None) -> torch.Tensor:
    if generator_seed is not None:
        torch.manual_seed(generator_seed)
    return aug(x)
