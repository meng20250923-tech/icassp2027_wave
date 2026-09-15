"""QualityNet and its shared waveform-preparation utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .core import WaveOperator


class QualityNet(nn.Module):
    """Small 1-D CNN that maps observed/predicted/residual traces to q_m."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Linear(channels, 8), nn.ReLU(inplace=True), nn.Linear(8, 1))

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(traces).squeeze(-1)).squeeze(-1)


def robust_prediction(operator: WaveOperator, data: np.ndarray) -> np.ndarray:
    """PDE prediction from an energy-normalized adjoint initialization."""
    energy = np.linalg.norm(data, axis=1)
    normalized = data * np.median(energy) / (energy[:, None] + 1e-9)
    image = np.maximum(operator.adjoint(normalized), 0.0)
    image /= image.max() + 1e-9
    prediction = operator.forward(image)
    return prediction * np.sum(data * prediction) / (np.sum(prediction * prediction) + 1e-9)


def channel_inputs(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Standardized [observation, prediction, residual] traces."""
    scale = observed.std(axis=1, keepdims=True) + 1e-9
    return np.stack([observed / scale, predicted / scale, (observed - predicted) / scale], axis=1).astype(np.float32)


def auroc(good_probability: np.ndarray, good_label: np.ndarray) -> float:
    """Dependency-free AUROC for good-channel probabilities."""
    good = good_probability[good_label == 1]
    bad = good_probability[good_label == 0]
    return float(np.mean(good[:, None] > bad[None, :]) + 0.5 * np.mean(good[:, None] == bad[None, :]))
