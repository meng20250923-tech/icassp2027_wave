"""Simulator returning continuous supervision targets for mixed channel faults."""
from __future__ import annotations

import numpy as np

from .core import WaveOperator
from .phantom_families import sample_phantom


def _shift(trace: np.ndarray, samples: int) -> np.ndarray:
    result = np.zeros_like(trace)
    if samples >= 0:
        result[samples:] = trace[: trace.size - samples]
    else:
        result[:samples] = trace[-samples:]
    return result


def calibration_case(
    operator: WaveOperator,
    rng: np.random.Generator,
    fault_count: int = 4,
    severity: float = 3.0,
    layout: str = "random",
    phantom_family: str = "id",
    fault_type: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Generate a nominal-PDE mixed-fault acquisition with exact parameters."""
    image = sample_phantom(operator.cfg.image_size, rng, family=phantom_family)
    clean = operator.forward(image)
    observed = clean + rng.normal(scale=operator.cfg.noise_std * max(clean.std(), 1e-12), size=clean.shape)
    if layout == "random":
        bad = rng.choice(operator.cfg.sensors, size=fault_count, replace=False)
    else:
        bad = (int(rng.integers(operator.cfg.sensors)) + np.arange(fault_count)) % operator.cfg.sensors
    if fault_type is not None and fault_type not in (1, 2, 3):
        raise ValueError("fault_type must be None, 1 (gain), 2 (delay), or 3 (spike).")
    types = np.full(fault_count, fault_type, dtype=np.int64) if fault_type is not None else rng.integers(1, 4, size=fault_count)
    targets = {
        "type": np.zeros(operator.cfg.sensors, dtype=np.int64),
        "log_gain": np.zeros(operator.cfg.sensors, dtype=np.float32),
        "delay": np.zeros(operator.cfg.sensors, dtype=np.float32),
        "spike_mask": np.zeros_like(observed, dtype=np.float32),
    }
    targets["type"][bad] = types
    for channel, kind in zip(bad, types):
        if kind == 1:
            extent = np.log1p(0.4 * severity)
            log_gain = float(rng.uniform(-extent, extent))
            observed[channel] *= np.exp(log_gain)
            targets["log_gain"][channel] = log_gain
        elif kind == 2:
            maximum = max(1, int(round(4 * severity)))
            shift = int(rng.choice(np.r_[np.arange(-maximum, 0), np.arange(1, maximum + 1)]))
            observed[channel] = _shift(observed[channel], shift)
            targets["delay"][channel] = shift
        else:
            width = int(rng.integers(6, 15))
            start = int(rng.integers(clean.shape[1] // 3, 2 * clean.shape[1] // 3 - width))
            amplitude = 8 * severity * clean.std() * rng.uniform(0.6, 1.4) * rng.choice((-1.0, 1.0))
            observed[channel, start : start + width] += amplitude * np.hanning(width)
            targets["spike_mask"][channel, start : start + width] = 1.0
    return image, observed, targets
