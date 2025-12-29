from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    neutral_class_id: int,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute CE loss only on non-neutral samples (Up/Down)."""
    mask = target != neutral_class_id
    if mask.sum() == 0:
        return torch.zeros((), device=logits.device)
    logits_sel = logits[mask]
    target_sel = target[mask]
    return nn.functional.cross_entropy(logits_sel, target_sel, weight=class_weights)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.int64))
    ds = TensorDataset(X_t, y_t)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    class_weights: Optional[torch.Tensor],
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.lr_min
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb)
            loss = masked_cross_entropy(logits, yb, cfg.neutral_class_id, class_weights)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            n_train += xb.size(0)
        train_loss = train_loss / max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = masked_cross_entropy(logits, yb, cfg.neutral_class_id, class_weights)
                val_loss += loss.item() * xb.size(0)
                n_val += xb.size(0)
        val_loss = val_loss / max(n_val, 1)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if cfg.verbose and (epoch == 1 or epoch % 10 == 0):
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {cur_lr:.2e}")

        if val_loss + 1e-12 < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                if cfg.verbose:
                    print(f"Early stop at epoch {epoch}, best_val={best_val:.6f}")
                break

    model.load_state_dict(best_state)
    return model, history


def predict_logits(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model = model.to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size].astype(np.float32)).to(device)
            logits = model(xb)
            outputs.append(logits.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


