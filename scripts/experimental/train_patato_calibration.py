"""Fine-tune CalibrationNet on labelled corruptions of real PATATO traces."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pat_reliability.calibration_net import CalibrationNet
from pat_reliability.experimental.patato import operator_from_package
from pat_reliability.experimental.routing import MeasuredGeometryOperator, inputs, mixed_corruption


def build(signals: np.ndarray, op: MeasuredGeometryOperator, rng: np.random.Generator, cases: int, fault_count: int) -> tuple[np.ndarray, ...]:
    records = [[], [], [], [], []]
    for _ in range(cases):
        observed, truth = mixed_corruption(signals[int(rng.integers(len(signals)))], rng, fault_count=fault_count)
        values = (inputs(observed, op.prediction(observed)), truth["type"], truth["log_gain"], truth["delay"], truth["spike_mask"])
        for destination, value in zip(records, values):
            destination.append(value)
    return tuple(np.concatenate(values) for values in records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--fault-count", type=int, default=4)
    parser.add_argument("--train-cases", type=int, default=2000)
    parser.add_argument("--validation-cases", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    package = np.load(args.data)
    op = operator_from_package(package)
    train_signals = package["signals"][package["split"] == 0]
    validation_signals = package["signals"][package["split"] == 1]
    train = build(train_signals, op, rng, args.train_cases, args.fault_count)
    valid = build(validation_signals, op, rng, args.validation_cases, args.fault_count)
    loader = DataLoader(TensorDataset(*(torch.from_numpy(value) for value in train)), batch_size=64, shuffle=True)
    network = CalibrationNet()
    counts = np.bincount(train[1], minlength=4)
    cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (4 * np.maximum(counts, 1)), dtype=torch.float32))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0))
    regression = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    for _ in range(args.epochs):
        for trace, target_type, target_gain, target_delay, target_mask in loader:
            output = network(trace)
            loss = cross_entropy(output["type_logits"], target_type) + bce(output["spike_logits"], target_mask)
            gain, delay = target_type == 1, target_type == 2
            if gain.any(): loss = loss + regression(output["log_gain"][gain], target_gain[gain])
            if delay.any(): loss = loss + .05 * regression(output["delay"][delay], target_delay[delay])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        labels = network(torch.from_numpy(valid[0]))["type_logits"].argmax(1).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": network.state_dict(), "seed": args.seed, "data": str(args.data), "protocol": "patato-real-waveform-controlled-corruption"}, args.output)
    print(f"seed={args.seed}; train cases={args.train_cases}; validation type accuracy={(labels == valid[1]).mean():.4f}; bad-type accuracy={(labels[valid[1] > 0] == valid[1][valid[1] > 0]).mean():.4f}; parameters={sum(p.numel() for p in network.parameters())}")


if __name__ == "__main__":
    main()
