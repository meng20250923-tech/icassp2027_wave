"""Non-learned robust data-fidelity baselines for PAT reconstruction."""
from __future__ import annotations

import numpy as np

from .core import ExperimentConfig, WaveOperator, tv_gradient


def reconstruct_huber(
    operator: WaveOperator,
    data: np.ndarray,
    config: ExperimentConfig,
    delta_multiplier: float = 8.0,
) -> np.ndarray:
    """Reconstruct with Huber-clipped waveform residuals and TV regularization.

    The threshold is set from the median absolute residual on every iteration,
    so the baseline has no learned channel labels or fault-specific parameters.
    """
    scale = np.max(np.abs(operator.adjoint(data))) + 1e-9
    image = np.maximum(operator.adjoint(data) / scale, 0.0)
    for _ in range(config.iterations):
        residual = operator.forward(image) - data
        delta = delta_multiplier * (np.median(np.abs(residual)) + 1e-9)
        huber_gradient = np.clip(residual, -delta, delta)
        gradient = operator.adjoint(huber_gradient) + config.tv_weight * tv_gradient(image)
        image = np.maximum(image - config.step_size * gradient / (np.max(np.abs(gradient)) + 1e-9), 0.0)
    return image / (image.max() + 1e-9)
