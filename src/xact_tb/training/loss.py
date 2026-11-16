"""Class-weighted binary cross-entropy (paper Eq. 2), numerically stable via logits.

L = -(1/N) * sum_i w_{y_i} [ y log s(z) + (1-y) log (1-s(z)) ],  w_Normal=1.0, w_TB=1.5.
Mean over N (NOT normalised by the sum of weights), exactly as written in the paper.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, w_normal: float = 1.0, w_tb: float = 1.5
) -> torch.Tensor:
    targets = targets.float()
    w = torch.where(targets > 0.5, torch.full_like(targets, w_tb), torch.full_like(targets, w_normal))
    return (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * w).mean()
