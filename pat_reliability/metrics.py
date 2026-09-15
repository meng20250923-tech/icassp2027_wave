"""Image-quality metrics used by the final PAT evaluation protocol."""
from __future__ import annotations

import numpy as np


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = np.mean((reference - estimate) ** 2)
    return float(10 * np.log10(1 / max(mse, 1e-12)))


def nrmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Root-mean-square error normalized by reference energy."""
    rmse = np.sqrt(np.mean((reference - estimate) ** 2))
    return float(rmse / (np.sqrt(np.mean(reference**2)) + 1e-12))


def ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Global SSIM for the [0, 1]-normalized PAT reconstructions."""
    mean_reference = float(reference.mean())
    mean_estimate = float(estimate.mean())
    variance_reference = float(reference.var())
    variance_estimate = float(estimate.var())
    covariance = float(np.mean((reference - mean_reference) * (estimate - mean_estimate)))
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mean_reference * mean_estimate + c1) * (2 * covariance + c2)
    denominator = (mean_reference**2 + mean_estimate**2 + c1) * (variance_reference + variance_estimate + c2)
    return float(numerator / denominator)
