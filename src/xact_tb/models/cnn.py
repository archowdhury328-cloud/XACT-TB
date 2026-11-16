"""Custom CNN branch reconstructed from paper Fig. 2 + §4.1.

Diagram: [Conv,Conv,SiLU,MaxPool] x4 -> GAP -> SiLU -> Dropout(0.4) -> Embedding(256-D).
NOT specified by the paper (configurable assumptions): channel widths, kernel size, padding,
normalisation layers (none by default), pooling kernel. Defaults: (32,64,128,256), k=3, pad=1,
no norm, pool 2. With last width 256 the GAP output is already the 256-D embedding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CNNConfig:
    channels: tuple[int, ...] = (32, 64, 128, 256)
    kernel: int = 3
    norm: str = "none"  # none|batch
    dropout: float = 0.4
    embed_dim: int = 256


def _norm(kind: str, c: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(c)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm {kind}")


class CustomCNN(nn.Module):
    def __init__(self, cfg: CNNConfig = CNNConfig(), in_ch: int = 3):
        super().__init__()
        self.cfg = cfg
        blocks, c_in = [], in_ch
        for c in cfg.channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(c_in, c, cfg.kernel, padding=cfg.kernel // 2),
                    _norm(cfg.norm, c),
                    nn.Conv2d(c, c, cfg.kernel, padding=cfg.kernel // 2),
                    _norm(cfg.norm, c),
                    nn.SiLU(inplace=False),
                    nn.MaxPool2d(2),
                )
            )
            c_in = c
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.act = nn.SiLU(inplace=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.proj = nn.Identity() if c_in == cfg.embed_dim else nn.Linear(c_in, cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, 1)  # auxiliary supervised head (removable)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Final conv feature map consumed by GAP (Grad-CAM target layer)."""
        return self.blocks(x)

    def embed_from_map(self, fmap: torch.Tensor) -> torch.Tensor:
        return self.proj(self.drop(self.act(self.gap(fmap).flatten(1))))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed_from_map(self.feature_map(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logit (B,) of the auxiliary classifier. Use ``embed`` for features."""
        return self.head(self.embed(x)).squeeze(1)
