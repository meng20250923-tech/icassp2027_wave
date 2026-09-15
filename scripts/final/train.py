"""Train continuous gain, delay, and time-mask predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pat_reliability.calibration_data import calibration_case
from pat_reliability.calibration_net import CalibrationNet
from pat_reliability.config import load_config
from pat_reliability.core import WaveOperator
from pat_reliability.quality_net import channel_inputs, robust_prediction


def build_dataset(
    operator: WaveOperator,
    cases: int,
    rng: np.random.Generator,
    fault_count: int,
) -> tuple[np.ndarray, ...]:
    """Generate flattened channel examples and calibration targets."""
    records: list[list[np.ndarray]] = [[], [], [], [], []]
    for _ in range(cases):
        _, observed, targets = calibration_case(operator, rng, fault_count=fault_count)
        values = (
            channel_inputs(observed, robust_prediction(operator, observed)),
            targets["type"], targets["log_gain"], targets["delay"], targets["spike_mask"],
        )
        for destination, value in zip(records, values):
            destination.append(value)
    return tuple(np.concatenate(values) for values in records)


def main() -> None:
    """Parse arguments, train CalibrationNet, and save one checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--fault-count", type=int, default=4, help="Absolute number of corrupted channels (legacy default: 4).")
    parser.add_argument("--fault-fraction", type=float, help="Override --fault-count with a fixed fraction of the active sensor array.")
    parser.add_argument("--train-cases", type=int, default=600)
    parser.add_argument("--validation-cases", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    operator = WaveOperator(load_config(args.config))
    if args.fault_fraction is not None:
        if not 0 < args.fault_fraction < 1:
            parser.error("--fault-fraction must be strictly between 0 and 1.")
        args.fault_count = max(1, int(round(args.fault_fraction * operator.cfg.sensors)))

    rng = np.random.default_rng(args.seed)
    train = build_dataset(operator, args.train_cases, rng, args.fault_count)
    valid = build_dataset(operator, args.validation_cases, rng, args.fault_count)
    loader = DataLoader(TensorDataset(*(torch.from_numpy(value) for value in train)), batch_size=64, shuffle=True)

    network = CalibrationNet()
    counts = np.bincount(train[1], minlength=4)
    class_weights = torch.tensor(counts.sum() / (4 * np.maximum(counts, 1)), dtype=torch.float32)
    type_loss = nn.CrossEntropyLoss(weight=class_weights)
    spike_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0))
    regression_loss = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    for _ in range(args.epochs):
        for traces, target_type, target_gain, target_delay, target_mask in loader:
            output = network(traces)
            loss = type_loss(output["type_logits"], target_type) + spike_loss(output["spike_logits"], target_mask)
            gain, delay = target_type == 1, target_type == 2
            if gain.any():
                loss = loss + regression_loss(output["log_gain"][gain], target_gain[gain])
            if delay.any():
                loss = loss + 0.05 * regression_loss(output["delay"][delay], target_delay[delay])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        predictions = network(torch.from_numpy(valid[0]))["type_logits"].argmax(1).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": network.state_dict(), "seed": args.seed, "config": operator.cfg.__dict__}, args.output)

    bad = valid[1] > 0
    fraction = args.fault_count / operator.cfg.sensors
    print(
        f"seed: {args.seed}; fault_count: {args.fault_count}; fault_fraction: {fraction:.4f}; "
        f"validation type accuracy: {(predictions == valid[1]).mean():.4f}; "
        f"bad-type accuracy: {(predictions[bad] == valid[1][bad]).mean():.4f}; "
        f"parameters: {sum(value.numel() for value in network.parameters())}"
    )


if __name__ == "__main__":
    main()
