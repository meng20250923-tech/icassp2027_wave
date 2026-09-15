"""Compact numerical core for the sparse-view PAT feasibility experiment.

The solver uses the initial-pressure wave equation on a square grid.  The
adjoint is the exact reverse mode of the same discrete time-stepping scheme,
which keeps data consistency and residual scoring physically coherent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:
    image_size: int = 96
    time_steps: int = 240
    dx: float = 1.0
    dt: float = 0.25
    sound_speed: float = 1.0
    sensors: int = 32
    sensor_radius: float = 38.0
    pml_width: int = 12
    iterations: int = 35
    step_size: float = 0.18
    tv_weight: float = 0.002
    noise_std: float = 0.002
    seed: int = 2026


def vessel_phantom(n: int) -> np.ndarray:
    """Create a deterministic vessel-like initial-pressure image in [0, 1]."""
    yy, xx = np.mgrid[:n, :n].astype(np.float64)
    image = np.zeros((n, n), dtype=np.float64)

    def tube(x0: float, y0: float, x1: float, y1: float, width: float, value: float) -> None:
        vx, vy = x1 - x0, y1 - y0
        projection = np.clip(((xx - x0) * vx + (yy - y0) * vy) / (vx * vx + vy * vy), 0, 1)
        distance = np.hypot(xx - (x0 + projection * vx), yy - (y0 + projection * vy))
        image[:] += value * np.exp(-(distance / width) ** 2)

    tube(26, 68, 69, 29, 1.6, 1.0)
    tube(46, 50, 75, 59, 1.35, 0.78)
    tube(42, 54, 28, 33, 1.25, 0.65)
    image += 0.8 * np.exp(-((xx - 33) ** 2 + (yy - 68) ** 2) / (2 * 3.2**2))
    image += 0.55 * np.exp(-((xx - 69) ** 2 + (yy - 29) ** 2) / (2 * 2.6**2))
    return image / image.max()


class WaveOperator:
    """2-D finite-difference wave operator with point sensors on a circle."""

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        n = config.image_size
        if config.sound_speed * config.dt / config.dx >= 1 / np.sqrt(2):
            raise ValueError("Unstable FDTD configuration: CFL condition is violated.")
        angles = np.linspace(0, 2 * np.pi, config.sensors, endpoint=False)
        center = (n - 1) / 2
        rows = np.rint(center + config.sensor_radius * np.sin(angles)).astype(int)
        cols = np.rint(center + config.sensor_radius * np.cos(angles)).astype(int)
        if np.any(rows < 1) or np.any(rows >= n - 1) or np.any(cols < 1) or np.any(cols >= n - 1):
            raise ValueError("Sensors must lie inside the computational grid.")
        self.rows, self.cols = rows, cols
        self.damping = self._damping_mask(n, config.pml_width)
        self.c2dt2dx2 = (config.sound_speed * config.dt / config.dx) ** 2

    @staticmethod
    def _damping_mask(n: int, width: int) -> np.ndarray:
        grid = np.indices((n, n))
        edge_distance = np.minimum.reduce([grid[0], grid[1], n - 1 - grid[0], n - 1 - grid[1]])
        fraction = np.clip((width - edge_distance) / max(width, 1), 0.0, 1.0)
        return np.exp(-0.18 * fraction**2)

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        padded = np.pad(field, 1)
        return (
            padded[2:, 1:-1] + padded[:-2, 1:-1] + padded[1:-1, 2:] + padded[1:-1, :-2] - 4 * field
        )

    def _base_step(self, field: np.ndarray) -> np.ndarray:
        return 2 * field + self.c2dt2dx2 * self._laplacian(field)

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Return detector pressures with shape [sensors, time_steps]."""
        if image.shape != (self.cfg.image_size, self.cfg.image_size):
            raise ValueError("Image has an unexpected shape.")
        previous = image.astype(np.float64, copy=True)  # zero initial velocity
        current = image.astype(np.float64, copy=True)
        data = np.empty((self.cfg.sensors, self.cfg.time_steps), dtype=np.float64)
        for t in range(self.cfg.time_steps):
            data[:, t] = current[self.rows, self.cols]
            following = self.damping * (self._base_step(current) - previous)
            previous, current = current, following
        return data

    def adjoint(self, data: np.ndarray) -> np.ndarray:
        """Exact discrete adjoint of :meth:`forward`."""
        if data.shape != (self.cfg.sensors, self.cfg.time_steps):
            raise ValueError("Data has an unexpected shape.")
        adj_current = np.zeros((self.cfg.image_size, self.cfg.image_size), dtype=np.float64)
        adj_previous = np.zeros_like(adj_current)
        for t in range(self.cfg.time_steps - 1, -1, -1):
            injected = np.zeros_like(adj_current)
            np.add.at(injected, (self.rows, self.cols), data[:, t])
            damped = self.damping * adj_current
            new_current = injected + self._base_step(damped) + adj_previous
            new_previous = -damped
            adj_current, adj_previous = new_current, new_previous
        return adj_current + adj_previous

    def check_adjoint(self, rng: np.random.Generator) -> float:
        x = rng.normal(size=(self.cfg.image_size, self.cfg.image_size))
        y = rng.normal(size=(self.cfg.sensors, self.cfg.time_steps))
        left = float(np.vdot(self.forward(x), y))
        right = float(np.vdot(x, self.adjoint(y)))
        return abs(left - right) / max(abs(left), abs(right), 1e-12)


def inject_corruption(
    clean: np.ndarray,
    bad_channels: Iterable[int],
    kind: str,
    rng: np.random.Generator,
    noise_std: float,
    severity: float = 1.0,
) -> np.ndarray:
    """Add ordinary noise and one controlled anomaly family to selected channels."""
    corrupted = clean + rng.normal(scale=noise_std * max(clean.std(), 1e-12), size=clean.shape)
    bad = np.asarray(list(bad_channels), dtype=int)
    if kind == "gain":
        corrupted[bad] *= 1 + 0.4 * severity
    elif kind == "delay":
        delay = max(1, int(round(4 * severity)))
        corrupted[bad, delay:] = corrupted[bad, :-delay]
        corrupted[bad, :delay] = 0.0
    elif kind == "spike":
        width = 10
        pulse = 8 * severity * clean.std() * np.hanning(width)
        for channel in bad:
            start = int(rng.integers(clean.shape[1] // 3, 2 * clean.shape[1] // 3))
            corrupted[channel, start : start + width] += pulse
    else:
        raise ValueError("kind must be one of: gain, delay, spike")
    return corrupted


def normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = first - first.mean()
    second = second - second.mean()
    return float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second) + 1e-12))


def robust_zscore(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return np.clip((values - median) / (1.4826 * mad + 1e-9), -4, 8)


def estimate_reliability(observed: np.ndarray, predicted: np.ndarray, max_delay: int = 6) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Physics-residual rule that outputs a soft reliability for every channel."""
    amplitude = np.linalg.norm(predicted - observed, axis=1) / (np.linalg.norm(observed, axis=1) + 1e-9)
    relative_energy = np.abs(np.log((np.linalg.norm(observed, axis=1) + 1e-9) / (np.median(np.linalg.norm(observed, axis=1)) + 1e-9)))
    delay = np.empty(observed.shape[0])
    neighbor = np.empty(observed.shape[0])
    normalized = (observed - observed.mean(axis=1, keepdims=True)) / (observed.std(axis=1, keepdims=True) + 1e-9)
    for m in range(observed.shape[0]):
        correlations = []
        for shift in range(-max_delay, max_delay + 1):
            shifted = np.roll(observed[m], shift)
            correlations.append(normalized_correlation(predicted[m], shifted))
        delay[m] = abs(int(np.argmax(correlations)) - max_delay)
        neighbor_reference = 0.5 * (normalized[(m - 1) % observed.shape[0]] + normalized[(m + 1) % observed.shape[0]])
        neighbor[m] = 1 - normalized_correlation(normalized[m], neighbor_reference)
    score = 0.45 * robust_zscore(relative_energy) + 0.35 * robust_zscore(delay) + 0.15 * robust_zscore(amplitude) + 0.05 * robust_zscore(neighbor)
    threshold = np.quantile(score, 0.875)  # expected fault rate: 4 / 32 channels
    reliability = 1 / (1 + np.exp(6.0 * (score - threshold)))
    return np.clip(reliability, 0.05, 1.0), {"amplitude": amplitude, "delay": delay, "neighbor": neighbor, "score": score}


def tv_gradient(image: np.ndarray) -> np.ndarray:
    """Smoothed isotropic-TV gradient."""
    gx = np.diff(image, axis=1, append=image[:, -1:])
    gy = np.diff(image, axis=0, append=image[-1:, :])
    magnitude = np.sqrt(gx * gx + gy * gy + 1e-6)
    nx, ny = gx / magnitude, gy / magnitude
    return np.diff(nx, axis=1, prepend=nx[:, :1]) + np.diff(ny, axis=0, prepend=ny[:1, :])


def reconstruct(operator: WaveOperator, data: np.ndarray, weights: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    """Projected gradient descent for weighted PDE data consistency plus TV."""
    weights = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    scale = np.max(np.abs(operator.adjoint(data))) + 1e-9
    image = np.maximum(operator.adjoint(data) / scale, 0.0)
    for _ in range(config.iterations):
        residual = weights * (operator.forward(image) - data)
        gradient = operator.adjoint(residual) + config.tv_weight * tv_gradient(image)
        image = np.maximum(image - config.step_size * gradient / (np.max(np.abs(gradient)) + 1e-9), 0.0)
    return image / (image.max() + 1e-9)


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = np.mean((reference - estimate) ** 2)
    return float(10 * np.log10(1 / max(mse, 1e-12)))
