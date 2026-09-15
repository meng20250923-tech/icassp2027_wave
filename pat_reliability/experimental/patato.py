"""PATATO HDF5 loading and deterministic preprocessing utilities.

The PATATO preclinical data contain raw, hardware-digitised time series from a
non-circular 256-element, partial-view array.  These helpers extract a fixed
32-channel sparse subset without normalising individual traces (which would
erase real inter-channel gain differences), remove the early trigger region,
and retain enough metadata to reproduce every split.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PatatoPreparation:
    """All preprocessing choices stored beside an exported dataset."""

    source: str
    detector_indices: tuple[int, ...]
    sample_start: int = 400
    sample_stop: int = 1900
    output_samples: int = 256
    lower_hz: float = 5.0e4
    upper_hz: float = 7.0e6
    train_frames: tuple[int, ...] = tuple(range(8))
    validation_frames: tuple[int, ...] = (8, 9)
    test_frames: tuple[int, ...] = (10, 11)

    def metadata(self) -> dict[str, object]:
        data = asdict(self)
        for key in ("detector_indices", "train_frames", "validation_frames", "test_frames"):
            data[key] = list(data[key])
        return data


def uniform_detector_subset(detectors: int, count: int = 32) -> np.ndarray:
    """Return reproducible uniformly distributed native detector indices."""
    if not 0 < count <= detectors:
        raise ValueError("count must be in [1, detectors].")
    return np.rint(np.linspace(0, detectors - 1, count)).astype(np.int64)


def bandpass_traces(traces: np.ndarray, sample_rate_hz: float, lower_hz: float, upper_hz: float) -> np.ndarray:
    """Apply the documented simple FFT bandpass to traces of shape [C, T]."""
    frequency = np.fft.rfftfreq(traces.shape[-1], d=1.0 / sample_rate_hz)
    spectrum = np.fft.rfft(traces, axis=-1)
    spectrum[..., (frequency < lower_hz) | (frequency > upper_hz)] = 0.0
    return np.fft.irfft(spectrum, n=traces.shape[-1], axis=-1).astype(np.float32)


def resample_traces(traces: np.ndarray, samples: int) -> np.ndarray:
    """Linearly resample time traces while preserving channel amplitudes."""
    source = np.linspace(0.0, 1.0, traces.shape[-1], dtype=np.float32)
    target = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    return np.stack([np.interp(target, source, row) for row in traces], axis=0).astype(np.float32)


def preprocess_acquisition(
    raw: np.ndarray,
    *,
    sample_rate_hz: float,
    sample_start: int,
    sample_stop: int,
    output_samples: int,
    lower_hz: float,
    upper_hz: float,
) -> tuple[np.ndarray, float]:
    """Preprocess one [channels, raw-time] real acquisition.

    A per-channel initial median removes ADC DC offsets.  Filtering occurs on
    the full trace before cropping to avoid an artificial edge at the crop.
    One *acquisition-wide* robust scale is used after cropping, retaining the
    relative gains between channels needed by the calibration task.
    """
    if raw.ndim != 2:
        raise ValueError("raw acquisition must have shape [channels, time].")
    if not 0 <= sample_start < sample_stop <= raw.shape[-1]:
        raise ValueError("invalid crop interval.")
    traces = raw.astype(np.float32, copy=True)
    baseline_count = min(128, traces.shape[-1])
    traces -= np.median(traces[:, :baseline_count], axis=1, keepdims=True)
    traces = bandpass_traces(traces, sample_rate_hz, lower_hz, upper_hz)
    traces = traces[:, sample_start:sample_stop]
    scale = float(np.percentile(np.abs(traces), 95))
    traces /= max(scale, 1e-6)
    return resample_traces(traces, output_samples), scale


def resize_reference(image: np.ndarray, size: int = 128) -> np.ndarray:
    """Resize a 2-D reference BP image without an image-processing dependency."""
    if image.ndim != 2:
        raise ValueError("reference image must be 2-D.")
    source = np.linspace(0.0, 1.0, image.shape[1], dtype=np.float32)
    target = np.linspace(0.0, 1.0, size, dtype=np.float32)
    horizontal = np.stack([np.interp(target, source, row) for row in image], axis=0)
    vertical = np.stack([np.interp(target, source, horizontal[:, column]) for column in range(size)], axis=1)
    vertical = vertical.astype(np.float32)
    return vertical / max(float(np.max(np.abs(vertical))), 1e-6)


def frame_split(frames: np.ndarray, preparation: PatatoPreparation) -> np.ndarray:
    """Return split labels (0 train, 1 val, 2 test) for acquisition frames."""
    result = np.full(frames.shape, -1, dtype=np.int8)
    for label, members in enumerate((preparation.train_frames, preparation.validation_frames, preparation.test_frames)):
        result[np.isin(frames, members)] = label
    if np.any(result < 0):
        raise ValueError("encountered a frame outside the stored split.")
    return result


def operator_from_package(package: np.lib.npyio.NpzFile):
    """Construct the measured-data operator stored in an exported NPZ package."""
    from .routing import MeasuredGeometryOperator

    metadata = json.loads(str(package["metadata"]))
    return MeasuredGeometryOperator(
        package["geometry_m"],
        metadata["sample_start"] / metadata["sample_rate_hz"],
        metadata["sample_stop"] / metadata["sample_rate_hz"],
        samples=package["signals"].shape[-1],
        sound_speed_m_per_s=metadata["sound_speed_m_per_s"],
        fov_m=0.024975,
    )
