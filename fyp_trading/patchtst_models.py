from __future__ import annotations

"""
PatchTST-style classifier (ICLR 2023-inspired) for multivariate time series.

Key ideas we implement (classification adaptation):
- Patchify the time axis into short local windows ("patches"), similar to ViT.
- Channel independence: model each variate independently with shared weights,
  then aggregate channel representations for classification.

Input:  x [B, L, C]
Output: logits [B, 3] for {Down, Neutral, Up}
"""

from dataclasses import dataclass
from typing import Literal, Optional

import math
import torch
import torch.nn as nn

from .labeling import NUM_CLASSES


@dataclass(frozen=True)
class PatchTSTConfig:
    # patching
    patch_len: int = 6
    stride: int = 6
    padding_mode: Literal["replicate", "zero"] = "replicate"

    # transformer
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.15

    # pooling & aggregation
    token_pool: Literal["mean", "last"] = "mean"
    channel_agg: Literal["mean", "concat"] = "concat"

    # regularization
    head_dropout: float = 0.2


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _patchify_1d(
    x: torch.Tensor,
    patch_len: int,
    stride: int,
    padding_mode: str,
) -> torch.Tensor:
    """
    x: [B, L]
    returns patches: [B, N, patch_len]
    """
    if x.dim() != 2:
        raise ValueError(f"patchify expects [B,L], got {tuple(x.shape)}")
    B, L = x.shape
    if patch_len <= 0 or stride <= 0:
        raise ValueError("patch_len and stride must be positive.")

    if L <= patch_len:
        n_patches = 1
    else:
        n_patches = 1 + _ceil_div(L - patch_len, stride)

    L_needed = patch_len + (n_patches - 1) * stride
    pad = int(max(0, L_needed - L))
    if pad > 0:
        if padding_mode == "replicate":
            last = x[:, -1:].expand(B, pad)
            x = torch.cat([x, last], dim=1)
        elif padding_mode == "zero":
            x = torch.cat([x, x.new_zeros(B, pad)], dim=1)
        else:
            raise ValueError(f"Unknown padding_mode={padding_mode!r}")

    patches = x.unfold(dimension=1, size=patch_len, step=stride)  # [B, N, patch_len]
    return patches


class PatchTSTMultiClassifier(nn.Module):
    def __init__(self, seq_len: int, num_channels: int, cfg: PatchTSTConfig):
        super().__init__()
        self.cfg = cfg
        self.seq_len = int(seq_len)
        self.num_channels = int(num_channels)

        # determine token length after patching
        if self.seq_len <= cfg.patch_len:
            n_tokens = 1
        else:
            n_tokens = 1 + _ceil_div(self.seq_len - cfg.patch_len, cfg.stride)
        self.n_tokens = int(n_tokens)

        # shared patch embedding across channels
        self.patch_proj = nn.Linear(cfg.patch_len, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_tokens, cfg.d_model))
        self.in_drop = nn.Dropout(cfg.dropout)

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

        if cfg.channel_agg == "concat":
            head_in = cfg.d_model * self.num_channels
        else:
            head_in = cfg.d_model

        self.head = nn.Sequential(
            nn.Dropout(cfg.head_dropout),
            nn.Linear(head_in, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.d_model, NUM_CLASSES),
        )

        self._init_params()

    def _init_params(self) -> None:
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        # patch_proj and head get default init; fine for our use

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        """
        if x.dim() != 3:
            raise ValueError(f"PatchTST expects [B,L,C], got {tuple(x.shape)}")
        B, L, C = x.shape
        if C != self.num_channels:
            raise ValueError(f"num_channels mismatch: got C={C}, expected {self.num_channels}")

        # channel independence with shared weights:
        # reshape to (B*C, L)
        xc = x.permute(0, 2, 1).contiguous().view(B * C, L)
        patches = _patchify_1d(
            xc,
            patch_len=self.cfg.patch_len,
            stride=self.cfg.stride,
            padding_mode=self.cfg.padding_mode,
        )  # [B*C, N, patch_len]

        z = self.patch_proj(patches)  # [B*C, N, d_model]
        z = z + self.pos_emb[:, : z.size(1), :]
        z = self.in_drop(z)
        z = self.encoder(z)

        if self.cfg.token_pool == "last":
            pooled = z[:, -1, :]
        else:
            pooled = z.mean(dim=1)
        pooled = self.norm(pooled)  # [B*C, d_model]

        # back to [B, C, d_model]
        pooled = pooled.view(B, C, -1)

        if self.cfg.channel_agg == "concat":
            rep = pooled.reshape(B, -1)
        else:
            rep = pooled.mean(dim=1)

        return self.head(rep)


