"""Explainability with explicit targets.

* ``gradcam_cnn``: Grad-CAM on the CNN feature map (pre-GAP). Target = logit of the CNN's
  AUXILIARY classifier head. This explains the CNN branch, NOT the fused SVM decision.
* ``swin_attention_map``: genuine window-attention weights from timm Swin blocks (requires
  fused_attn disabled; Backbone does that). Handles windows and cyclic shift; no CLS token assumed.
* ``branch_attenuation``: post-hoc perturbation of frozen feature blocks through a declared
  strength schedule. Describes score sensitivity only; NOT causal importance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gradcam_cnn(cnn, x: torch.Tensor, out_size: int = 224) -> torch.Tensor:
    """(B,1,out,out) Grad-CAM in [0,1] per image for the auxiliary TB logit."""
    cnn.eval()
    fmap = cnn.feature_map(x).detach().requires_grad_(True)
    logit = cnn.head(cnn.embed_from_map(fmap)).sum()
    (grad,) = torch.autograd.grad(logit, fmap)
    w = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(out_size, out_size), mode="bilinear", align_corners=False)
    mx = cam.amax(dim=(2, 3), keepdim=True)
    return torch.where(mx > 0, cam / mx.clamp_min(1e-12), torch.zeros_like(cam)).detach()


def _unwindow(t: torch.Tensor, H: int, W: int, ws: tuple[int, int]) -> torch.Tensor:
    """(B*nW, wh*ww) -> (B,H,W) reversing window_partition."""
    wh, ww = ws
    nW = (H // wh) * (W // ww)
    B = t.shape[0] // nW
    t = t.view(B, H // wh, W // ww, wh, ww)
    return t.permute(0, 1, 3, 2, 4).reshape(B, H, W)


@torch.no_grad()
def swin_attention_map(backbone, x: torch.Tensor, stage: int = 1, out_size: int = 224) -> torch.Tensor:
    """Attention-received map (B,1,out,out) averaged over heads and blocks of one Swin stage.

    For each block: attention probabilities (B*nW, heads, N, N) -> mean over heads, mean over
    queries (= attention each token RECEIVES) -> un-window -> undo cyclic shift. Padding is not
    handled (224 with window 7 needs none); a ValueError is raised if dims do not tile.
    """
    trunk = backbone.trunk
    blocks = trunk.layers[stage].blocks
    store: list[tuple[torch.Tensor, object]] = []
    hooks = []
    for blk in blocks:
        if getattr(blk.attn, "fused_attn", False):
            raise RuntimeError("fused attention active; build the model after set_fused_attn(False)")
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o, blk=blk: store.append((o.detach(), blk))))
    try:
        backbone.eval()
        backbone.embed(x)
    finally:
        for h in hooks:
            h.remove()
    maps = []
    for attn, blk in store:
        H, W = blk.input_resolution
        ws = tuple(blk.window_size)
        if H % ws[0] or W % ws[1]:
            raise ValueError("feature map does not tile into windows (padding unsupported)")
        recv = attn.mean(dim=1).mean(dim=1)  # (B*nW, N)
        m = _unwindow(recv, H, W, ws)
        sh = tuple(blk.shift_size)
        if any(sh):
            m = torch.roll(m, shifts=(sh[0], sh[1]), dims=(1, 2))  # inverse of the forward roll(-s)
        maps.append(m)
    cam = torch.stack(maps).mean(0).unsqueeze(1)
    cam = F.interpolate(cam, size=(out_size, out_size), mode="bilinear", align_corners=False)
    lo, hi = cam.amin(dim=(2, 3), keepdim=True), cam.amax(dim=(2, 3), keepdim=True)
    return (cam - lo) / (hi - lo).clamp_min(1e-12)


def branch_attenuation(
    x_scaled: np.ndarray,
    slices: dict[str, slice],
    decision_fn,
    alphas=(1.0, 0.75, 0.5, 0.25, 0.0),
) -> list[dict]:
    """Multiply one branch's block by alpha (in SCALED space, where 0 == training mean), keep the
    other branch intact, record decision scores and class flips. Baseline (alpha=1) is stored
    separately and perturbed predictions are never forced to match it."""
    base = decision_fn(x_scaled)
    rows = []
    for name, sl in slices.items():
        for a in alphas:
            xp = x_scaled.copy()
            xp[:, sl] = xp[:, sl] * a
            s = decision_fn(xp)
            rows.append(
                {
                    "attenuated_branch": name,
                    "alpha": a,
                    "mean_abs_score_change": float(np.mean(np.abs(s - base))),
                    "n_class_flips": int(((s > 0) != (base > 0)).sum()),
                    "n": int(len(s)),
                }
            )
    return rows
