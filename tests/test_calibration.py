"""Unit tests for deterministic ContinuousRoute waveform corrections."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from pat_reliability.calibration import (
    apply_oracle_correction,
    apply_predicted_correction,
    shift_trace,
)


class CalibrationUtilitiesTest(unittest.TestCase):
    def test_shift_trace_uses_zero_padding(self) -> None:
        trace = np.arange(5, dtype=np.float32)
        np.testing.assert_array_equal(shift_trace(trace, 2), [0, 0, 0, 1, 2])
        np.testing.assert_array_equal(shift_trace(trace, -2), [2, 3, 4, 0, 0])

    def test_predicted_gain_correction(self) -> None:
        observed = np.array([[2.0, 4.0]], dtype=np.float32)
        output = {
            "type_logits": torch.zeros((1, 4)),
            "log_gain": torch.tensor([np.log(2.0)], dtype=torch.float32),
            "delay": torch.zeros(1),
            "spike_logits": torch.zeros((1, 2)),
        }
        corrected, weights = apply_predicted_correction(
            observed,
            output,
            labels=np.array([1]),
            confident=np.array([True]),
        )
        np.testing.assert_allclose(corrected, [[1.0, 2.0]], rtol=1e-6)
        np.testing.assert_array_equal(weights, np.ones_like(observed))

    def test_oracle_spike_is_time_local(self) -> None:
        observed = np.ones((1, 4), dtype=np.float32)
        truth = {
            "type": np.array([3]),
            "log_gain": np.zeros(1, dtype=np.float32),
            "delay": np.zeros(1, dtype=np.float32),
            "spike_mask": np.array([[0, 1, 1, 0]], dtype=np.float32),
        }
        corrected, weights = apply_oracle_correction(observed, truth)
        np.testing.assert_array_equal(corrected, observed)
        np.testing.assert_allclose(weights, [[1.0, 0.05, 0.05, 1.0]])


if __name__ == "__main__":
    unittest.main()
