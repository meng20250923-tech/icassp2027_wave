"""Time-local reliability handling for impulsive channel corruption."""
from __future__ import annotations
import numpy as np
from .core import ExperimentConfig, WaveOperator, tv_gradient

def spike_time_weights(observed: np.ndarray, predicted: np.ndarray, low: float = .05) -> np.ndarray:
    residual = np.abs(observed - predicted)
    med = np.median(residual, axis=1, keepdims=True)
    mad = np.median(np.abs(residual - med), axis=1, keepdims=True) + 1e-9
    mask = residual > med + 6 * 1.4826 * mad
    mask[:, 1:] |= mask[:, :-1]
    mask[:, :-1] |= mask[:, 1:]
    return np.where(mask, low, 1.0)

def reconstruct_temporal(operator: WaveOperator, data: np.ndarray, weights: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[:, None]
    if weights.shape not in ((operator.cfg.sensors, 1), data.shape):
        raise ValueError("invalid temporal weights")
    scale = np.max(np.abs(operator.adjoint(data))) + 1e-9
    image = np.maximum(operator.adjoint(data) / scale, 0.0)
    for _ in range(config.iterations):
        grad = operator.adjoint(weights * (operator.forward(image) - data))
        grad += config.tv_weight * tv_gradient(image)
        image = np.maximum(
            image - config.step_size * grad / (np.max(np.abs(grad)) + 1e-9),
            0.0,
        )
    return image / (image.max() + 1e-9)
