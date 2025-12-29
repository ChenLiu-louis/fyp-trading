from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LabelingConfig

CLASS_ID_DOWN = 0
CLASS_ID_NEUTRAL = 1
CLASS_ID_UP = 2
NUM_CLASSES = 3


def apply_3class_labeling(feat_df: pd.DataFrame, cfg: LabelingConfig) -> pd.Series:
    """Label by comparing `next_return` to threshold.

    Requires columns: `next_return`, `logret_1d`.
    """
    if cfg.use_dynamic_threshold:
        roll_vol = feat_df["logret_1d"].rolling(cfg.vol_window).std().shift(1)
        thresholds = (cfg.k_dynamic * roll_vol).clip(lower=cfg.min_vol)
    else:
        thresholds = pd.Series(cfg.k_dynamic, index=feat_df.index, dtype=float)

    labels = pd.Series(np.nan, index=feat_df.index, dtype=float)
    labels[feat_df["next_return"] >= thresholds] = CLASS_ID_UP
    labels[feat_df["next_return"] <= -thresholds] = CLASS_ID_DOWN
    labels[(feat_df["next_return"] > -thresholds) & (feat_df["next_return"] < thresholds)] = CLASS_ID_NEUTRAL
    return labels


