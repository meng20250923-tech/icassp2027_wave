"""Render controlled-corruption qualitative cases from held-out PATATO data."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pat_reliability.calibration_net import CalibrationNet
from pat_reliability.experimental.patato import operator_from_package
from pat_reliability.experimental.routing import inputs, mixed_corruption, rule_weights
from scripts.experimental.evaluate_patato_calibration import correction, oracle_correction


def render_case(package, operator, network, args, case: int, output: Path):
    """Render one deterministic test corruption and return its reconstruction row."""
    signals = package["signals"][package["split"] == 2]
    if not 0 <= case < len(signals):
        raise ValueError(f"case must be in [0, {len(signals) - 1}].")

    clean = signals[case]
    rng = np.random.default_rng(args.seed)
    # Advance the seeded generator so an index matches the batch evaluator.
    for index in range(case + 1):
        source = clean if index == case else signals[index]
        observed, truth = mixed_corruption(source, rng, args.fault_count, args.severity, args.layout)

    predicted = operator.prediction(observed)
    with torch.no_grad():
        output_net = network(torch.from_numpy(inputs(observed, predicted)))
    probabilities = torch.softmax(output_net["type_logits"], dim=1).numpy()
    labels = probabilities.argmax(1)
    confident = probabilities.max(1) >= args.confidence
    corrected, weights = correction(observed, output_net, labels, confident)
    oracle_data, oracle_weights = oracle_correction(observed, truth)
    hard = np.ones_like(observed)
    hard[(labels != 0) & confident] = 0.05

    images = (
        operator.reconstruct_pde_tv(clean),
        operator.reconstruct_pde_tv(observed),
        operator.reconstruct_pde_tv(observed, rule_weights(observed, predicted)),
        operator.reconstruct_pde_tv(observed, hard),
        operator.reconstruct_pde_tv(corrected, weights),
        operator.reconstruct_pde_tv(oracle_data, oracle_weights),
    )
    names = (
        "Clean real reference", "Equal", "RuleTemporal", "HardReject",
        "ContinuousRoute", "OracleFull",
    )

    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(images), figsize=(18, 3.4), constrained_layout=True)
    for axis, image, name in zip(axes, images, names):
        axis.imshow(image, cmap="magma", vmin=0, vmax=1)
        axis.set_title(name)
        axis.axis("off")
    figure.savefig(output / "reconstruction.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    traces = (observed, predicted, corrected, weights)
    trace_names = (
        "Corrupted real traces", "Geometry prediction",
        "Continuous corrected", "Temporal reliability",
    )
    for axis, values, name in zip(axes, traces, trace_names):
        axis.plot(values.T, alpha=0.32, linewidth=0.6)
        axis.set_ylabel(name)
    axes[-1].set_xlabel("Resampled time sample")
    figure.savefig(output / "waveforms.png", dpi=220)
    plt.close(figure)

    np.savez_compressed(
        output / "case.npz",
        truth_type=truth["type"],
        predicted_type=labels,
        confident=confident,
        truth_mask=truth["spike_mask"],
        predicted_mask=torch.sigmoid(output_net["spike_logits"]).numpy(),
    )
    print(f"wrote {output / 'reconstruction.png'} and waveforms.png")
    print(f"case={case}; truth types={truth['type'].tolist()}")
    print(f"predicted types={labels.tolist()}; confident={confident.astype(int).tolist()}")
    return images, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=12061)
    parser.add_argument("--case", type=int, action="append", help="Held-out case index. Repeat this option to create a montage.")
    parser.add_argument("--fault-count", type=int, default=4)
    parser.add_argument("--severity", type=float, default=3)
    parser.add_argument("--layout", choices=("random", "contiguous"), default="random")
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    package = np.load(args.data)
    operator = operator_from_package(package)
    state = torch.load(args.checkpoint, map_location="cpu")
    network = CalibrationNet()
    network.load_state_dict(state["state_dict"])
    network.eval()

    cases = args.case or [0]
    if len(cases) == 1:
        render_case(package, operator, network, args, cases[0], args.output)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    names = None
    for case in cases:
        images, names = render_case(
            package, operator, network, args, case,
            args.output / f"case_{case:03d}",
        )
        rows.append(images)

    figure, axes = plt.subplots(
        len(rows), len(names), figsize=(18, 3.1 * len(rows)),
        constrained_layout=True, squeeze=False,
    )
    for row_index, images in enumerate(rows):
        for column, (axis, image, name) in enumerate(zip(axes[row_index], images, names)):
            axis.imshow(image, cmap="magma", vmin=0, vmax=1)
            if row_index == 0:
                axis.set_title(name)
            if column == 0:
                axis.set_ylabel(f"Case {cases[row_index]}")
            axis.axis("off")
    path = args.output / "reconstruction_montage.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
