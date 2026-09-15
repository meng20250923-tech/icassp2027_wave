"""Measured-geometry operators, corruptions, and robust PAT baselines."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ..calibration import shift_trace
from ..quality_net import channel_inputs
from ..core import tv_gradient

# Backward-compatible public name used by earlier experiment scripts.
shift = shift_trace

@dataclass
class MeasuredGeometryOperator:
    geometry_m: np.ndarray
    time_start_s: float
    time_stop_s: float
    image_size: int = 128
    fov_m: float = 0.024975
    sound_speed_m_per_s: float = 1449.0
    samples: int = 256
    def __post_init__(self) -> None:
        coordinates = np.asarray(self.geometry_m, dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("geometry_m must have shape [sensors, 2].")
        self.geometry_m = coordinates
        axis = np.linspace(-self.fov_m / 2, self.fov_m / 2, self.image_size, dtype=np.float32)
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        pixels = np.stack((xx.ravel(), yy.ravel()), axis=1)
        distance = np.linalg.norm(coordinates[:, None, :] - pixels[None, :, :], axis=-1)
        sample_position = (distance / self.sound_speed_m_per_s - self.time_start_s) / ((self.time_stop_s - self.time_start_s) / max(self.samples - 1, 1))
        self.low = np.floor(sample_position).astype(np.int32)
        self.fraction = (sample_position - self.low).astype(np.float32)
        self.valid = (self.low >= 0) & (self.low < self.samples - 1)
        self.low = np.clip(self.low, 0, self.samples - 2)
        self.spreading = (1.0 / np.maximum(distance, 1e-3)).astype(np.float32)
    @property
    def sensors(self) -> int:
        return int(self.geometry_m.shape[0])
    def adjoint(self, data: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        values = np.asarray(data, dtype=np.float32)
        if values.shape != (self.sensors, self.samples):
            raise ValueError("data has an unexpected shape.")
        if weights is not None:
            values = values * np.asarray(weights, dtype=np.float32)
        image = np.zeros(self.image_size * self.image_size, dtype=np.float32)
        for sensor in range(self.sensors):
            low = self.low[sensor]
            fraction = self.fraction[sensor]
            sampled = (1 - fraction) * values[sensor, low] + fraction * values[sensor, low + 1]
            image += self.valid[sensor] * self.spreading[sensor] * sampled
        return image.reshape(self.image_size, self.image_size)
    def forward(self, image: np.ndarray) -> np.ndarray:
        pixels = np.asarray(image, dtype=np.float32).reshape(-1)
        if pixels.size != self.image_size * self.image_size:
            raise ValueError("image has an unexpected shape.")
        data = np.zeros((self.sensors, self.samples), dtype=np.float32)
        for sensor in range(self.sensors):
            valid = self.valid[sensor]
            low = self.low[sensor, valid]
            fraction = self.fraction[sensor, valid]
            amplitudes = pixels[valid] * self.spreading[sensor, valid]
            data[sensor] += np.bincount(low, weights=amplitudes * (1 - fraction), minlength=self.samples)
            data[sensor] += np.bincount(low + 1, weights=amplitudes * fraction, minlength=self.samples)
        return data
    def prediction(self, observed: np.ndarray) -> np.ndarray:
        energy = np.linalg.norm(observed, axis=1)
        normalized = observed * np.median(energy) / (energy[:, None] + 1e-8)
        image = np.maximum(self.adjoint(normalized), 0.0)
        image /= max(float(np.max(image)), 1e-8)
        predicted = self.forward(image)
        scale = float(np.sum(observed * predicted) / (np.sum(predicted * predicted) + 1e-8))
        return predicted * scale
    def reconstruct(self, data: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        image = np.maximum(self.adjoint(data, weights), 0.0)
        return image / max(float(np.max(image)), 1e-8)

    def reconstruct_pde_tv(
        self,
        data: np.ndarray,
        weights: np.ndarray | None = None,
        iterations: int = 35,
        step_size: float = 0.18,
        tv_weight: float = 0.002,
    ) -> np.ndarray:
        """Solve weighted data consistency with nonnegative TV regularization."""
        measured = np.asarray(data, dtype=np.float32)
        temporal_weights = np.ones_like(measured) if weights is None else np.asarray(weights, dtype=np.float32)
        if temporal_weights.shape != measured.shape:
            raise ValueError("weights must have the same shape as data.")

        initial = np.maximum(self.adjoint(measured), 0.0)
        initial /= max(float(np.max(initial)), 1e-9)
        nominal = self.forward(initial)
        data_scale = float(np.sum(measured * nominal) / (np.sum(nominal * nominal) + 1e-9))
        if abs(data_scale) < 1e-9:
            return initial
        values = measured / data_scale
        image = initial
        for _ in range(iterations):
            residual = self.forward(image) - values
            gradient = self.adjoint(temporal_weights * residual)
            gradient += tv_weight * tv_gradient(image)
            image = np.maximum(
                image - step_size * gradient / (np.max(np.abs(gradient)) + 1e-9),
                0.0,
            )
        return image / max(float(np.max(image)), 1e-9)

def reconstruct_huber(
    operator: MeasuredGeometryOperator,
    data: np.ndarray,
    iterations: int = 35,
    step_size: float = 0.18,
    tv_weight: float = 0.002,
    delta_multiplier: float = 8.0,
) -> np.ndarray:
    """Type-agnostic Huber data fidelity for measured-geometry PAT."""
    scale = float(np.max(np.abs(operator.adjoint(data))) + 1e-9)
    image = np.maximum(operator.adjoint(data) / scale, 0.0)
    for _ in range(iterations):
        residual = operator.forward(image) - data
        delta = delta_multiplier * (np.median(np.abs(residual)) + 1e-9)
        gradient = operator.adjoint(np.clip(residual, -delta, delta))
        gradient += tv_weight * tv_gradient(image)
        image = np.maximum(
            image - step_size * gradient / (np.max(np.abs(gradient)) + 1e-9),
            0.0,
        )
    return image / max(float(image.max()), 1e-9)

def mixed_corruption(clean: np.ndarray, rng: np.random.Generator, fault_count: int = 4, severity: float = 3.0, layout: str = "random") -> tuple[np.ndarray, dict[str, np.ndarray]]:
    observed = np.asarray(clean, dtype=np.float32).copy()
    channels, samples = observed.shape
    bad = rng.choice(channels, size=fault_count, replace=False) if layout == "random" else (int(rng.integers(channels)) + np.arange(fault_count)) % channels
    kinds = rng.integers(1, 4, size=fault_count)
    truth = {"type": np.zeros(channels, dtype=np.int64), "log_gain": np.zeros(channels, dtype=np.float32), "delay": np.zeros(channels, dtype=np.float32), "spike_mask": np.zeros_like(observed, dtype=np.float32)}
    truth["type"][bad] = kinds
    acquisition_scale = float(np.std(clean))
    for channel, kind in zip(bad, kinds):
        if kind == 1:
            extent = np.log1p(0.4 * severity)
            log_gain = float(rng.uniform(-extent, extent))
            observed[channel] *= np.exp(log_gain)
            truth["log_gain"][channel] = log_gain
        elif kind == 2:
            maximum = max(1, int(round(4 * severity)))
            amount = int(rng.choice(np.r_[np.arange(-maximum, 0), np.arange(1, maximum + 1)]))
            observed[channel] = shift(observed[channel], amount)
            truth["delay"][channel] = amount
        else:
            width = int(rng.integers(6, 15))
            start = int(rng.integers(samples // 3, 2 * samples // 3 - width))
            amplitude = 8 * severity * acquisition_scale * rng.uniform(0.6, 1.4) * rng.choice((-1.0, 1.0))
            observed[channel, start : start + width] += amplitude * np.hanning(width)
            truth["spike_mask"][channel, start : start + width] = 1.0
    return observed, truth

def rule_weights(observed: np.ndarray, predicted: np.ndarray | None = None, low: float = 0.05) -> np.ndarray:
    del predicted
    padded = np.pad(observed, ((0, 0), (4, 4)), mode="edge")
    local = np.median(np.lib.stride_tricks.sliding_window_view(padded, 9, axis=1), axis=-1)
    deviation = np.abs(observed - local)
    median = np.median(deviation, axis=1, keepdims=True)
    mad = np.median(np.abs(deviation - median), axis=1, keepdims=True) + 1e-8
    mask = deviation > median + 8 * 1.4826 * mad
    mask[:, 1:] |= mask[:, :-1]
    mask[:, :-1] |= mask[:, 1:]
    return np.where(mask, low, 1.0).astype(np.float32)

def inputs(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return channel_inputs(observed, predicted)
