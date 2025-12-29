from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn

from .labeling import NUM_CLASSES
from .transformer_models import PositionalEncoding


@dataclass(frozen=True)
class InformerConfig:
    """
    Informer-style encoder classifier config (adapted for classification).

    Key ideas from Informer (Zhou et al., 2021):
    - ProbSparse self-attention: compute full attention only for top-u queries.
    - Distilling: downsample sequence length between encoder layers to reduce redundancy.
    """

    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.2
    use_cls_token: bool = True

    # ProbSparse parameters
    factor: int = 5  # controls sample_k / top_u via factor * ln(L)

    # Distilling (downsample between encoder layers)
    distil: bool = True


def _prob_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    factor: int,
    attn_dropout_p: float,
    training: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    ProbSparse attention (approximate, Informer-style) for one head.

    Shapes:
      q, k, v: (B, H, L, D)
    Returns:
      context: (B, H, L, D)
      attn (optional): None (we skip returning full attention to save memory)
    """
    b, h, l_q, d = q.shape
    _, _, l_k, _ = k.shape

    # Sample keys for sparsity measurement
    sample_k = min(l_k, max(1, factor * int(math.ceil(math.log(max(l_k, 2))))))  # factor * ln(Lk)
    top_u = min(l_q, max(1, factor * int(math.ceil(math.log(max(l_q, 2))))))  # factor * ln(Lq)

    # Randomly sample a subset of keys to estimate sparsity of each query
    # idx: (sample_k,)
    idx = torch.randint(low=0, high=l_k, size=(sample_k,), device=q.device)
    k_sample = k[:, :, idx, :]  # (B,H,sample_k,D)

    # (B,H,Lq,sample_k)
    scores_sample = torch.matmul(q, k_sample.transpose(-2, -1)) / math.sqrt(d)
    # sparsity measurement: max - mean over sampled keys
    m = scores_sample.max(dim=-1).values - scores_sample.mean(dim=-1)  # (B,H,Lq)

    # pick top_u queries per (B,H)
    top_idx = torch.topk(m, k=top_u, dim=-1, largest=True, sorted=False).indices  # (B,H,top_u)

    # Default context for all queries: mean(V) (Informer uses different init for causal settings;
    # here we do non-causal encoder attention, so mean is a reasonable fallback.)
    context = v.mean(dim=-2, keepdim=True).expand(b, h, l_q, d).contiguous()  # (B,H,Lq,D)

    # Gather top queries: (B,H,top_u,D)
    q_top = torch.gather(q, dim=2, index=top_idx.unsqueeze(-1).expand(-1, -1, -1, d))
    # Full attention only for selected queries: (B,H,top_u,Lk)
    scores_top = torch.matmul(q_top, k.transpose(-2, -1)) / math.sqrt(d)
    attn = torch.softmax(scores_top, dim=-1)
    if attn_dropout_p > 0:
        attn = torch.dropout(attn, p=attn_dropout_p, train=training)
    # (B,H,top_u,D)
    out_top = torch.matmul(attn, v)

    # Scatter back into context
    context = context.scatter(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, d), out_top)
    return context, None


class ProbSparseMultiheadAttention(nn.Module):
    """Multi-head ProbSparse self-attention (Informer-style)."""

    def __init__(self, d_model: int, nhead: int, factor: int, dropout: float):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model = d_model
        self.nhead = nhead
        self.factor = factor
        self.dropout = dropout
        self.d_head = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,D)
        b, l, d = x.shape
        q = self.q_proj(x).view(b, l, self.nhead, self.d_head).transpose(1, 2)  # (B,H,L,Dh)
        k = self.k_proj(x).view(b, l, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, l, self.nhead, self.d_head).transpose(1, 2)

        context, _ = _prob_sparse_attn(
            q=q,
            k=k,
            v=v,
            factor=self.factor,
            attn_dropout_p=self.dropout,
            training=self.training,
        )  # (B,H,L,Dh)
        out = context.transpose(1, 2).contiguous().view(b, l, d)  # (B,L,D)
        return self.out_proj(out)


class InformerEncoderLayer(nn.Module):
    def __init__(self, cfg: InformerConfig):
        super().__init__()
        self.attn = ProbSparseMultiheadAttention(cfg.d_model, cfg.nhead, cfg.factor, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_feedforward),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_feedforward, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm
        y = self.attn(self.norm1(x))
        x = x + self.drop(y)
        y2 = self.ff(self.norm2(x))
        x = x + self.drop(y2)
        return x


class _DistillConv(nn.Module):
    """Informer 'distilling' block: Conv1d + activation + maxpool to reduce length."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,D) -> (B,D,L)
        z = x.transpose(1, 2)
        z = self.conv(z)
        z = self.act(z)
        z = self.pool(z)
        z = self.drop(z)
        return z.transpose(1, 2)


class InformerMultiClassifier(nn.Module):
    """
    Informer-style encoder classifier for fixed-length feature sequences.

    Input: (B, T, F)
    Output: logits (B, NUM_CLASSES)
    """

    def __init__(self, input_size: int, seq_len: int, cfg: InformerConfig):
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

        self.layers = nn.ModuleList([InformerEncoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.distills = nn.ModuleList(
            [_DistillConv(cfg.d_model, cfg.dropout) for _ in range(max(cfg.num_layers - 1, 0))]
        )

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
        z = self.in_proj(x)  # (B,T,D)
        if self.cls is not None:
            cls = self.cls.expand(z.size(0), -1, -1)
            z = torch.cat([cls, z], dim=1)
        z = self.pos_enc(z)

        for i, layer in enumerate(self.layers):
            z = layer(z)
            if self.cfg.distil and i < len(self.distills):
                # Keep CLS token (if present) while distilling the remaining sequence
                if self.cls is not None:
                    cls_tok = z[:, :1, :]
                    rest = z[:, 1:, :]
                    rest = self.distills[i](rest)
                    z = torch.cat([cls_tok, rest], dim=1)
                else:
                    z = self.distills[i](z)

        if self.cls is not None:
            pooled = z[:, 0, :]
        else:
            pooled = z[:, -1, :]
        pooled = self.norm(pooled)
        return self.head(pooled)


