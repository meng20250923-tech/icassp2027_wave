"""Shared waveform correction utilities for ContinuousRoute.

These functions contain the deterministic post-processing applied to network
outputs. Keeping them here ensures that batch evaluation and qualitative
visualization use exactly the same gain, delay, and spike corrections.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


NetworkOutput = Mapping[str, torch.Tensor]


def shift_trace(trace: np.ndarray, samples: int) -> np.ndarray:
    """Shift a one-dimensional trace with zero padding and no wraparound."""
    shifted = np.zeros_like(trace)
    if samples >= 0:
        shifted[samples:] = trace[: trace.size - samples]
    else:
        shifted[:samples] = trace[-samples:]
    return shifted


def apply_predicted_correction(
    observed: np.ndarray,
    output: NetworkOutput,
    labels: np.ndarray,
    confident: np.ndarray,
    *,
    predicted: np.ndarray | None = None,
    correct_gain: bool = True,
    correct_delay: bool = True,
    whole_channel_spike: bool = False,
    residual_gate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply type-specific network predictions to waveform data.

    Gain and delay actions can optionally be accepted only when they reduce
    the residual to ``predicted``. Spike predictions are converted into
    sample-level reliability weights, or a whole-channel weight for the
    corresponding ablation.
    """
    if residual_gate and predicted is None:
        raise ValueError("predicted is required when residual_gate=True.")

    corrected = observed.copy()
    weights = np.ones_like(observed)
    for channel in range(observed.shape[0]):
        if not confident[channel]:
            continue

        if labels[channel] == 1 and correct_gain:
            log_gain = np.clip(output["log_gain"][channel].item(), -1, 1)
            candidate = observed[channel] / np.exp(log_gain)
            if not residual_gate or np.linalg.norm(candidate - predicted[channel]) < np.linalg.norm(
                observed[channel] - predicted[channel]
            ):
                corrected[channel] = candidate
        elif labels[channel] == 2 and correct_delay:
            delay = int(np.rint(np.clip(output["delay"][channel].item(), -12, 12)))
            candidate = shift_trace(observed[channel], -delay)
            if not residual_gate or np.linalg.norm(candidate - predicted[channel]) < np.linalg.norm(
                observed[channel] - predicted[channel]
            ):
                corrected[channel] = candidate
        elif labels[channel] == 3:
            if whole_channel_spike:
                weights[channel] = 0.05
            else:
                spike_probability = torch.sigmoid(output["spike_logits"][channel]).numpy()
                weights[channel] = 1.0 - 0.95 * spike_probability
    return corrected, weights


def apply_oracle_correction(
    observed: np.ndarray,
    truth: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exact simulated fault parameters to construct OracleFull."""
    corrected = observed.copy()
    weights = np.ones_like(observed)
    for channel, fault_type in enumerate(truth["type"]):
        if fault_type == 1:
            corrected[channel] = observed[channel] / np.exp(truth["log_gain"][channel])
        elif fault_type == 2:
            corrected[channel] = shift_trace(observed[channel], -int(truth["delay"][channel]))
        elif fault_type == 3:
            weights[channel] = np.where(truth["spike_mask"][channel] > 0, 0.05, 1.0)
    return corrected, weights
