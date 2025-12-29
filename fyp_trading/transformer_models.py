from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn

from .labeling import NUM_CLASSES


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        t = x.size(1)
        x = x + self.pe[:, :t, :]
        return self.dropout(x)


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.2
    use_cls_token: bool = True


class TransformerMultiClassifier(nn.Module):
    """Transformer encoder classifier for fixed-length feature sequences.

    Input: X of shape (B, T, F)
    Output: logits of shape (B, NUM_CLASSES)
    """

    def __init__(self, input_size: int, seq_len: int, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.seq_len = seq_len

        self.in_proj = nn.Linear(input_size, cfg.d_model)

        if cfg.use_cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            pe_len = seq_len + 1
        else:
            self.cls = None
            pe_len = seq_len

        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=max(64, pe_len + 8), dropout=cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)

        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, NUM_CLASSES),
        )

        self._init_params()

    def _init_params(self) -> None:
        if self.cls is not None:
            nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        z = self.in_proj(x)  # (B, T, D)
        if self.cls is not None:
            cls = self.cls.expand(z.size(0), -1, -1)  # (B,1,D)
            z = torch.cat([cls, z], dim=1)  # (B, T+1, D)
        z = self.pos_enc(z)
        z = self.encoder(z)  # (B, T(+1), D)
        if self.cls is not None:
            pooled = z[:, 0, :]
        else:
            pooled = z[:, -1, :]
        pooled = self.norm(pooled)
        return self.head(pooled)


