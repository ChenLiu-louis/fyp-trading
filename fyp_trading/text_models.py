from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FinBertConfig:
    """
    FinBERT fine-tuning config (news text -> 3-class direction label).

    This module is optional: it requires `transformers` and a model download.
    Recommended starting points:
    - ProsusAI/finbert
    - yiyanghkust/finbert-tone
    """

    model_name: str = "ProsusAI/finbert"
    max_length: int = 256
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 16
    weight_decay: float = 0.01
    seed: int = 42


def _require_transformers() -> Any:
    try:
        import transformers  # type: ignore

        return transformers
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "This feature requires `transformers`. Install it first, e.g.\n"
            "  pip install transformers datasets accelerate sentencepiece\n"
        ) from e


def finbert_tokenizer(cfg: FinBertConfig) -> Any:
    transformers = _require_transformers()
    return transformers.AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)


def finbert_model(cfg: FinBertConfig, num_labels: int = 3) -> Any:
    transformers = _require_transformers()
    return transformers.AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=num_labels)


def predict_proba_finbert(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    batch_size: int = 16,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Return probabilities of shape (N,3). Assumes model outputs logits for 3 labels.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=getattr(tokenizer, "model_max_length", 256),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            out.append(p)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3), dtype=float)


