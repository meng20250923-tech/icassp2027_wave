"""Create paper-ready mixed-fault ContinuousRoute figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pat_reliability.calibration import apply_oracle_correction, apply_predicted_correction
from pat_reliability.calibration_data import calibration_case
from pat_reliability.calibration_net import CalibrationNet
from pat_reliability.config import load_config
from pat_reliability.core import WaveOperator, psnr
from pat_reliability.quality_net import channel_inputs, robust_prediction
from pat_reliability.temporal_routing import reconstruct_temporal, spike_time_weights


def main() -> None:
    """Generate reconstruction and channel-diagnosis figures for one case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int, default=85051)
    parser.add_argument("--severity", type=float, default=3)
    parser.add_argument("--layout", choices=("random", "contiguous"), default="random")
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    network = CalibrationNet()
    network.load_state_dict(state["state_dict"])
    network.eval()
    operator = WaveOperator(load_config(args.config))
    image, observed, truth = calibration_case(
        operator,
        np.random.default_rng(args.seed),
        severity=args.severity,
        layout=args.layout,
    )
    predicted = robust_prediction(operator, observed)
    with torch.no_grad():
        output = network(torch.from_numpy(channel_inputs(observed, predicted)))
    probabilities = torch.softmax(output["type_logits"], dim=1).numpy()
    labels = probabilities.argmax(axis=1)
    confident = probabilities.max(axis=1) >= args.confidence
    corrected, weights = apply_predicted_correction(observed, output, labels, confident)
    oracle_data, oracle_weights = apply_oracle_correction(observed, truth)
    hard_weights = np.ones_like(observed)
    hard_weights[(labels != 0) & confident] = 0.05

    reconstructions = [
        image,
        reconstruct_temporal(operator, observed, np.ones_like(observed), operator.cfg),
        reconstruct_temporal(operator, observed, spike_time_weights(observed, predicted), operator.cfg),
        reconstruct_temporal(operator, observed, hard_weights, operator.cfg),
        reconstruct_temporal(operator, corrected, weights, operator.cfg),
        reconstruct_temporal(operator, oracle_data, oracle_weights, operator.cfg),
    ]
    names = ["Ground truth", "Equal", "RuleTemporal", "HardReject", "ContinuousRoute", "OracleFull reference"]
    args.output.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 6, figsize=(18, 4.25))
    for axis, name, reconstruction in zip(axes, names, reconstructions):
        axis.imshow(reconstruction, cmap="magma", vmin=0, vmax=1)
        title = name if name == "Ground truth" else f"{name}\n{psnr(image, reconstruction):.2f} dB"
        axis.set_title(title, pad=12)
        axis.axis("off")
    figure.subplots_adjust(left=0.015, right=0.995, bottom=0.03, top=0.80, wspace=0.05)
    figure.savefig(args.output / "mixed_reconstruction.png", dpi=240)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    axes[0].imshow(observed - predicted, aspect="auto", cmap="RdBu_r")
    axes[0].set(title="Wave residual", xlabel="time", ylabel="channel")
    axes[1].imshow(torch.sigmoid(output["spike_logits"]).numpy(), aspect="auto", cmap="magma")
    axes[1].set(title="Predicted spike probability", xlabel="time", ylabel="channel")
    axes[2].imshow(probabilities.T, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axes[2].set(
        title="Type probability", xlabel="channel", yticks=range(4),
        yticklabels=("good", "gain", "delay", "spike"),
    )
    figure.tight_layout()
    figure.savefig(args.output / "mixed_diagnosis.png", dpi=240)
    plt.close(figure)


if __name__ == "__main__":
    main()
