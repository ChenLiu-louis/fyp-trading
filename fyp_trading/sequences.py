from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def make_sequences(
    feat_df: pd.DataFrame,
    feature_cols: List[str],
    lookback: int,
    label_cols: List[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], pd.DatetimeIndex]:
    """Convert 2D features into 3D sequences with end-aligned labels."""
    X_2d = feat_df[feature_cols].astype(np.float32).values
    dates = feat_df.index
    n = len(feat_df)
    if n < lookback:
        raise ValueError(f"Insufficient samples: N={n} < lookback={lookback}")

    X_list = []
    labels_dict: Dict[str, list] = {col: [] for col in label_cols}
    idx_list = []

    for end in range(lookback - 1, n):
        start = end - lookback + 1
        X_list.append(X_2d[start : end + 1])
        idx_list.append(dates[end])
        for col in label_cols:
            labels_dict[col].append(feat_df.iloc[end][col])

    X_seq = np.stack(X_list).astype(np.float32)
    labels_np = {col: np.asarray(vals) for col, vals in labels_dict.items()}
    seq_index = pd.DatetimeIndex(idx_list)
    return X_seq, labels_np, seq_index


def fit_scaler_3d(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std for 3D sequences (fit on train only)."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-12, 1e-12, std)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_3d(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean[None, None, :]) / std[None, None, :]


