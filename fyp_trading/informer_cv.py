from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from .config import PipelineConfig, TrainConfig
from .informer_models import InformerConfig, InformerMultiClassifier
from .labeling import CLASS_ID_DOWN, CLASS_ID_NEUTRAL, CLASS_ID_UP, NUM_CLASSES
from .sequences import fit_scaler_3d, transform_3d
from .train import make_loader, predict_logits, softmax_np, train_model


def fixed_window_cv_informer(
    X_seq_all: np.ndarray,
    y_all: np.ndarray,
    seq_index: pd.DatetimeIndex,
    cfg_pipe: PipelineConfig,
    cfg_train: TrainConfig,
    cfg_model: InformerConfig,
    device: torch.device,
    save_last_fold_artifacts: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict[str, Any]]]:
    """Fixed-window walk-forward CV using an Informer-style encoder classifier."""
    metrics: List[Dict[str, Any]] = []
    preds_list: List[pd.DataFrame] = []

    n = len(X_seq_all)
    start_index = cfg_pipe.train_window + cfg_pipe.val_size
    fold_id = 0
    step = cfg_pipe.resolved_step_size()

    last_artifacts: Optional[Dict[str, Any]] = None

    for test_start in range(start_index, n - cfg_pipe.test_size + 1, step):
        train_end = test_start
        val_start = train_end - cfg_pipe.val_size
        train_start = val_start - cfg_pipe.train_window
        test_end = test_start + cfg_pipe.test_size
        if train_start < 0:
            continue

        X_train = X_seq_all[train_start:val_start]
        y_train = y_all[train_start:val_start]
        X_val = X_seq_all[val_start:train_end]
        y_val = y_all[val_start:train_end]
        X_test = X_seq_all[test_start:test_end]
        y_test = y_all[test_start:test_end]
        test_dates = seq_index[test_start:test_end]

        mask_ud_train = y_train != cfg_train.neutral_class_id
        mask_ud_val = y_val != cfg_train.neutral_class_id
        if mask_ud_train.sum() < 20 or mask_ud_val.sum() < 5:
            fold_id += 1
            continue

        mean, std = fit_scaler_3d(X_train)
        X_train_sc = transform_3d(X_train, mean, std)
        X_val_sc = transform_3d(X_val, mean, std)
        X_test_sc = transform_3d(X_test, mean, std)

        if cfg_train.loss_mode == "full_ce":
            counts_all = np.bincount(y_train, minlength=NUM_CLASSES)
            # balanced weights; avoid division by zero
            w = counts_all.sum() / np.maximum(counts_all, 1.0)
            w = w / w.mean()
            class_weights = torch.tensor(w, device=device, dtype=torch.float32)
        else:
            counts = np.bincount(y_train[mask_ud_train], minlength=NUM_CLASSES)
            w_down = counts.sum() / max(1.0, 2 * counts[CLASS_ID_DOWN])
            w_up = counts.sum() / max(1.0, 2 * counts[CLASS_ID_UP])
            class_weights = torch.tensor([w_down, 0.0, w_up], device=device, dtype=torch.float32)

        model = InformerMultiClassifier(
            input_size=X_train_sc.shape[-1],
            seq_len=X_train_sc.shape[1],
            cfg=cfg_model,
        )

        train_loader = make_loader(X_train_sc, y_train, cfg_train.batch_size, shuffle=True)
        val_loader = make_loader(X_val_sc, y_val, cfg_train.batch_size, shuffle=False)
        model, _ = train_model(model, train_loader, val_loader, cfg_train, device=device, class_weights=class_weights)

        logits_test = predict_logits(model, X_test_sc, device=device, batch_size=256)
        proba_test = softmax_np(logits_test)

        maxp = proba_test.max(axis=1)
        pred_raw = proba_test.argmax(axis=1)
        pred_class = pred_raw.copy()
        pred_class[maxp < cfg_pipe.proba_threshold] = CLASS_ID_NEUTRAL

        y_true = y_test
        mask_pred_up = pred_class == CLASS_ID_UP
        mask_pred_down = pred_class == CLASS_ID_DOWN
        mask_pred_ud = mask_pred_up | mask_pred_down

        up_acc = accuracy_score(y_true[mask_pred_up], pred_class[mask_pred_up]) if mask_pred_up.any() else np.nan
        down_acc = accuracy_score(y_true[mask_pred_down], pred_class[mask_pred_down]) if mask_pred_down.any() else np.nan
        dir_acc = accuracy_score(y_true[mask_pred_ud], pred_class[mask_pred_ud]) if mask_pred_ud.any() else np.nan
        coverage = float(mask_pred_ud.mean())

        metrics.append(
            {
                "fold": fold_id,
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "up_acc": up_acc,
                "down_acc": down_acc,
                "dir_acc_on_signals": dir_acc,
                "coverage": coverage,
                "num_pred_up": int(mask_pred_up.sum()),
                "num_pred_down": int(mask_pred_down.sum()),
                "num_pred_ud": int(mask_pred_ud.sum()),
                "test_size": int(len(X_test_sc)),
            }
        )

        fold_pred = pd.DataFrame(
            {
                "fold": fold_id,
                "date": test_dates,
                "actual_class": y_true,
                "pred_class": pred_class,
                "proba_down": proba_test[:, CLASS_ID_DOWN],
                "proba_neutral": proba_test[:, CLASS_ID_NEUTRAL],
                "proba_up": proba_test[:, CLASS_ID_UP],
            }
        )
        preds_list.append(fold_pred)

        if save_last_fold_artifacts:
            last_artifacts = {
                "fold": fold_id,
                "test_start": str(test_dates[0]),
                "test_end": str(test_dates[-1]),
                "scaler_mean": mean,
                "scaler_std": std,
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "cfg_pipe": asdict(cfg_pipe),
                "cfg_train": asdict(cfg_train),
                "cfg_model": asdict(cfg_model),
            }

        fold_id += 1
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(metrics)
    preds_df = pd.concat(preds_list, ignore_index=True) if preds_list else pd.DataFrame()
    return metrics_df, preds_df, last_artifacts


