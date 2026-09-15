"""Continuous channel-calibration network for reliability-aware PAT."""
from __future__ import annotations

import torch
from torch import nn


class CalibrationNet(nn.Module):
    """Predict fault type, log-gain, delay, and a time-local spike mask."""

    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, channels, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.type_head = nn.Sequential(nn.Linear(channels, 16), nn.ReLU(inplace=True), nn.Linear(16, 4))
        self.gain_head = nn.Linear(channels, 1)
        self.delay_head = nn.Linear(channels, 1)
        self.spike_head = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, traces: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(traces)
        pooled = self.pool(encoded).squeeze(-1)
        return {
            "type_logits": self.type_head(pooled),
            "log_gain": self.gain_head(pooled).squeeze(-1),
            "delay": self.delay_head(pooled).squeeze(-1),
            "spike_logits": self.spike_head(encoded).squeeze(1),
        }
