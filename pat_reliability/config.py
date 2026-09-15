"""Shared experiment-configuration loading."""
from __future__ import annotations
import json
from dataclasses import replace
from pathlib import Path
from .core import ExperimentConfig


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    values = json.loads(Path(path).read_text())
    n = int(values["image_size"])
    return replace(ExperimentConfig(), image_size=n, time_steps=int(values["time_steps"]), sensors=int(values["sensors"]),
                   sensor_radius=n * float(values["sensor_radius_fraction"]), iterations=int(values["iterations"]),
                   step_size=float(values["step_size"]), tv_weight=float(values["tv_weight"]), noise_std=float(values["noise_std"]),
                   seed=int(values["seed"]))
