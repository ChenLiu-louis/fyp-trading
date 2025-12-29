from __future__ import annotations

import torch
import torch.nn as nn

from .labeling import NUM_CLASSES


class LSTMMultiClassifier(nn.Module):
    """3-class LSTM classifier (Down/Neutral/Up) ported from `LSTM_2.ipynb`."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ReLU()
        self.out = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        h = self.dropout(h)
        h = self.act(self.fc1(h))
        return self.out(h)


