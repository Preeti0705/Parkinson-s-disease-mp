"""
inference_models.py
--------------------
Lightweight inference-only module exposing CNN1D and LSTMModel.

Architecture matches advanced_model.py v2:
  - CNN1D      : Residual 1-D CNN + AdaptiveMaxPool1d (dimension-agnostic)
  - LSTMModel  : Bidirectional stacked LSTM + self-attention pooling

Importing this file does NOT pull in Boruta, ADASYN, XGBoost or any
other training-only dependency — safe to import inside Streamlit.
"""

import torch
import torch.nn as nn


# ── Residual block ──────────────────────────────────────────────────────────────
class _ResBlock(nn.Module):
    """Two conv layers with a skip connection (He et al., 2016)."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.body(x))


# ── Residual CNN1D ──────────────────────────────────────────────────────────────
class CNN1D(nn.Module):
    """
    Residual 1-D Convolutional Network.

    AdaptiveMaxPool1d(1) collapses the sequence dimension to 1 regardless of
    input length, making the model dimension-agnostic.  The same weights
    therefore work for:
      • Training on 41/50+ PCA components
      • Inference on 13 real-time audio features
    """

    def __init__(self, input_dim: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.res1 = _ResBlock(64)
        self.down = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.res2 = _ResBlock(128)
        self.pool = nn.AdaptiveMaxPool1d(1)          # dimension-agnostic
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, seq_len)
        x = self.stem(x)
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        x = self.pool(x).squeeze(-1)        # → (batch, 128)
        return self.head(x).squeeze(-1)     # → (batch,)


# ── Bidirectional LSTM with Self-Attention ──────────────────────────────────────
class LSTMModel(nn.Module):
    """
    Bidirectional stacked LSTM with self-attention pooling.

    input_size=1 treats each feature as a single time-step value so the model
    works with any sequence length (any number of features).
    """

    def __init__(self, input_dim: int = 1):
        super().__init__()
        # Layer 1: input_size=1  → hidden=64 (each direction) → output=128
        self.lstm1 = nn.LSTM(
            input_size=1, hidden_size=64,
            batch_first=True, bidirectional=True,
        )
        self.drop1 = nn.Dropout(0.3)

        # Layer 2: input_size=128 → hidden=64 (each direction) → output=128
        self.lstm2 = nn.LSTM(
            input_size=128, hidden_size=64,
            batch_first=True, bidirectional=True,
        )
        self.drop2 = nn.Dropout(0.3)

        # Self-attention scoring over time steps
        self.attn = nn.Linear(128, 1)

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        x, _ = self.lstm1(x)          # → (batch, seq, 128)
        x     = self.drop1(x)
        x, _ = self.lstm2(x)          # → (batch, seq, 128)
        x     = self.drop2(x)

        # Soft attention: weighted sum over sequence positions
        w = torch.softmax(self.attn(x), dim=1)   # (batch, seq, 1)
        x = (w * x).sum(dim=1)                   # (batch, 128)

        return self.head(x).squeeze(-1)           # (batch,)
