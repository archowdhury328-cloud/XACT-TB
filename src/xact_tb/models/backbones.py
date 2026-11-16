"""timm backbones: Swin branch + direct baselines (Swin, CoAtNet-3, MaxViT).

Model identifiers are timm names verified against timm 1.0.28 (`timm.list_models`).
The paper does NOT state the Swin/MaxViT variant or the pretrained source. Candidate evidence only
(not proof): timm `swin_small_patch4_window7_224` ~50 M params / 8.7 GFLOPs, `coatnet_3_224`
~168 M / 34.7 G, `maxvit_tiny_tf_224` ~31 M / 5.6 G match the paper's Table 5 sizes. These are
ASSUMPTIONS recorded in docs/assumptions_and_deviations.md and verified by measurement
(`xact-tb benchmark`), never silently inferred.
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BASELINES = {
    "swin": "swin_small_patch4_window7_224",
    "coatnet3": "coatnet_3_224",
    "maxvit": "maxvit_tiny_tf_224",
}
SMOKE = "smoke_swin"  # tiny random-init Swin for CI only; never reported as a result


@dataclass(frozen=True)
class BackboneConfig:
    name: str = "swin_small_patch4_window7_224"
    pretrained: bool = True  # ASSUMPTION: ImageNet init (paper silent)
    freeze: bool = False  # ASSUMPTION: fine-tune
    normalize: str = "imagenet"  # imagenet|none


class Normalize(nn.Module):
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def _create(name: str, pretrained: bool, **kw) -> nn.Module:
    if name == SMOKE:
        from timm.models.swin_transformer import SwinTransformer

        return SwinTransformer(
            img_size=224,
            patch_size=4,
            embed_dim=16,
            depths=(1, 1, 2, 1),
            num_heads=(1, 2, 4, 8),
            window_size=7,
            num_classes=0,
        )
    return timm.create_model(name, pretrained=pretrained, **kw)


class Backbone(nn.Module):
    """Normalise -> timm trunk (pooled features) -> linear head (1 logit)."""

    def __init__(self, cfg: BackboneConfig, num_features: int | None = None):
        super().__init__()
        self.cfg = cfg
        if cfg.name.startswith(("swin", "smoke")):
            timm.layers.set_fused_attn(False)  # needed so attention maps are materialised
        self.norm = Normalize() if cfg.normalize == "imagenet" else nn.Identity()
        self.trunk = _create(cfg.name, cfg.pretrained, num_classes=0) if cfg.name != SMOKE else _create(SMOKE, False)
        self.embed_dim = self.trunk.num_features
        self.head = nn.Linear(self.embed_dim, 1)
        if cfg.freeze:
            for p in self.trunk.parameters():
                p.requires_grad = False

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(self.norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x)).squeeze(1)


def build_baseline(key: str, pretrained: bool = True) -> Backbone:
    if key not in BASELINES and key != SMOKE:
        raise KeyError(f"unknown baseline {key}; choose from {list(BASELINES)} or {SMOKE}")
    name = BASELINES.get(key, SMOKE)
    return Backbone(BackboneConfig(name=name, pretrained=pretrained))
