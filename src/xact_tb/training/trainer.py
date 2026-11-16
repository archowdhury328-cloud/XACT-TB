"""Supervised training of one network (CNN branch, Swin branch, or a direct baseline).

Choices the paper does not specify, all exposed in ``TrainConfig``:
  * scheduler monitor (default val_loss, mode min)   * checkpoint-selection metric (val_loss)
  * gradient clipping norm type (default L2 total norm, max_norm=0.2) -- "threshold 0.2" is ambiguous
  * weight decay (0)   * AMP (off)
Learning curves written here belong to THIS network's auxiliary classifier, not to the final SVM.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from xact_tb.evaluation.metrics import compute_metrics
from xact_tb.training.loss import weighted_bce_with_logits
from xact_tb.utils.common import seed_everything, worker_init_fn


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.0
    clip_norm: float | None = 0.2
    clip_norm_type: float = 2.0
    w_normal: float = 1.0
    w_tb: float = 1.5
    sched_factor: float = 0.5
    sched_patience: int = 5
    monitor: str = "val_loss"  # val_loss|val_auc
    select_metric: str = "val_loss"  # checkpoint selection (never test data)
    amp: bool = False
    num_workers: int = 0
    seed: int = 42

    def validate(self) -> None:
        if self.monitor not in ("val_loss", "val_auc") or self.select_metric not in ("val_loss", "val_auc"):
            raise ValueError("monitor/select_metric must be val_loss or val_auc")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")


def _better(metric: str, new: float, best: float | None) -> bool:
    if best is None or np.isnan(best):
        return True
    return new < best if metric == "val_loss" else new > best


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ids, ys, zs = [], [], []
    for b in loader:
        zs.append(model(b["image"].to(device)).float().cpu())
        ys.append(b["label"])
        ids.extend(b["image_id"])
    return ids, torch.cat(ys).numpy().astype(int), torch.cat(zs).numpy()


def evaluate_loss(model, loader, device, cfg: TrainConfig) -> dict:
    ids, y, z = predict_logits(model, loader, device)
    loss = float(weighted_bce_with_logits(torch.tensor(z), torch.tensor(y), cfg.w_normal, cfg.w_tb))
    m = compute_metrics(y, z)
    return {
        "val_loss": loss,
        "val_auc": m["roc_auc"],
        "val_acc": m["accuracy"],
        "val_sens": m["sensitivity"],
        "val_spec": m["specificity"],
    }


def train_network(
    model: nn.Module,
    train_ds,
    val_loader: DataLoader,
    cfg: TrainConfig,
    out_dir: str | Path,
    device: torch.device | str = "cpu",
    resume: bool = True,
    tag: str = "net",
) -> list[dict]:
    """Train and return per-epoch history. Writes last.pt, best.pt, history.jsonl in out_dir."""
    cfg.validate()
    device = torch.device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    mode = "min" if cfg.monitor == "val_loss" else "max"
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode=mode, factor=cfg.sched_factor, patience=cfg.sched_patience
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    history: list[dict] = []
    start, best = 0, None
    last = out / "last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)  # own, trusted checkpoint
        if ck["tag"] != tag:
            raise ValueError(f"checkpoint tag {ck['tag']} != {tag}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        history, start, best = ck["history"], ck["epoch"] + 1, ck["best"]
    for epoch in range(start, cfg.epochs):
        t0 = time.time()
        counts = train_ds.set_epoch(epoch)
        g = torch.Generator().manual_seed(cfg.seed + epoch)
        loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=g,
            num_workers=cfg.num_workers,
            worker_init_fn=worker_init_fn,
        )
        model.train()
        run, nb = 0.0, 0
        for b in loader:
            x, y = b["image"].to(device), b["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
                z = model(x)
            loss = weighted_bce_with_logits(z.float(), y, cfg.w_normal, cfg.w_tb)
            scaler.scale(loss).backward()
            if cfg.clip_norm:
                scaler.unscale_(opt)  # clip AFTER unscaling
                nn.utils.clip_grad_norm_(params, cfg.clip_norm, cfg.clip_norm_type)
            scaler.step(opt)
            scaler.update()
            run += float(loss.detach())
            nb += 1
        val = evaluate_loss(model, val_loader, device, cfg)
        sched.step(val[cfg.monitor])
        is_best = _better(cfg.select_metric, val[cfg.select_metric], best)
        if is_best:
            best = val[cfg.select_metric]
        rec = {
            "epoch": epoch,
            "train_loss": run / max(nb, 1),
            **val,
            "lr": opt.param_groups[0]["lr"],
            "effective_counts": counts,
            "seconds": round(time.time() - t0, 2),
            "selected_as_best": bool(is_best),
            "tag": tag,
        }
        history.append(rec)
        state = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "history": history,
            "epoch": epoch,
            "best": best,
            "tag": tag,
            "cfg": asdict(cfg),
        }
        torch.save(state, last)
        if is_best:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "metric": cfg.select_metric, "value": best, "tag": tag},
                out / "best.pt",
            )
        with open(out / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=float) + "\n")
    return history


def load_best(model: nn.Module, out_dir: str | Path, device="cpu") -> dict:
    ck = torch.load(Path(out_dir) / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    return {k: ck[k] for k in ("epoch", "metric", "value", "tag")}
