"""Evaluate controlled channel calibration on held-out real PATATO waveforms."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pat_reliability.calibration_net import CalibrationNet
from pat_reliability.calibration import apply_oracle_correction, apply_predicted_correction
from pat_reliability.experimental.patato import operator_from_package
from pat_reliability.experimental.routing import (
    MeasuredGeometryOperator,
    inputs,
    mixed_corruption,
    reconstruct_huber,
    rule_weights,
)
from pat_reliability.metrics import nrmse, ssim


def score(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, float, float]:
    reference = reference / max(float(reference.max()), 1e-8)
    estimate = estimate / max(float(estimate.max()), 1e-8)
    return (float(10 * np.log10(1 / max(float(np.mean((reference - estimate) ** 2)), 1e-12))), ssim(reference, estimate), nrmse(reference, estimate))


def correction(observed, output, labels, confident):
    """Apply the published measured-data route without residual gating."""
    return apply_predicted_correction(observed, output, labels, confident)


def oracle_correction(observed, truth):
    """Apply exact injected fault parameters for the OracleFull reference."""
    return apply_oracle_correction(observed, truth)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=12061)
    parser.add_argument("--fault-count", type=int, default=4)
    parser.add_argument("--severity", type=float, default=3)
    parser.add_argument("--layout", choices=("random", "contiguous"), default="random")
    parser.add_argument("--confidence", type=float, default=.90)
    parser.add_argument("--cases", type=int, default=130)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument(
        "--report-by-frame", action="store_true",
        help="Additionally report reconstruction metrics separately for each recorded PATATO frame.",
    )
    args = parser.parse_args()
    package = np.load(args.data)
    op = operator_from_package(package)
    split_label = {"train": 0, "validation": 1, "test": 2}[args.split]
    split_mask = package["split"] == split_label
    signals = package["signals"][split_mask]
    frames = package["frames"][split_mask]
    if args.cases > len(signals): raise ValueError("cases exceeds selected split acquisitions.")
    state = torch.load(args.checkpoint, map_location="cpu")
    network = CalibrationNet()
    network.load_state_dict(state["state_dict"])
    network.eval()
    rows = []
    rng = np.random.default_rng(args.seed)
    correct_bad = total_bad = 0
    gain_errors, delay_errors = [], []
    tp = fp = fn = actions = 0
    for clean in signals[:args.cases]:
        reference = op.reconstruct_pde_tv(clean)
        observed, truth = mixed_corruption(clean, rng, args.fault_count, args.severity, args.layout)
        predicted = op.prediction(observed)
        with torch.no_grad(): output = network(torch.from_numpy(inputs(observed, predicted)))
        probability = torch.softmax(output["type_logits"], dim=1).numpy()
        labels = probability.argmax(1)
        confident = probability.max(1) >= args.confidence
        actions += int(((labels != 0) & confident).sum())
        corrected, weights = correction(observed, output, labels, confident)
        oracle_data, oracle_weights = oracle_correction(observed, truth)
        hard = np.ones_like(observed)
        hard[(labels != 0) & confident] = 0.05
        reconstructions = (
            op.reconstruct_pde_tv(observed), reconstruct_huber(op, observed),
            op.reconstruct_pde_tv(observed, rule_weights(observed, predicted)),
            op.reconstruct_pde_tv(observed, hard), op.reconstruct_pde_tv(corrected, weights),
            op.reconstruct_pde_tv(oracle_data, oracle_weights),
        )
        rows.append([score(reference, item) for item in reconstructions])
        bad = truth["type"] > 0
        correct_bad += int((labels[bad] == truth["type"][bad]).sum())
        total_bad += int(bad.sum())
        gain = truth["type"] == 1
        delay = truth["type"] == 2
        spikes = truth["type"] == 3
        gain_errors.extend(np.abs(output["log_gain"].numpy()[gain] - truth["log_gain"][gain]))
        delay_errors.extend(np.abs(output["delay"].numpy()[delay] - truth["delay"][delay]))
        proposed = torch.sigmoid(output["spike_logits"]).numpy() >= 0.5
        target = truth["spike_mask"] >= 0.5
        tp += np.logical_and(proposed[spikes], target[spikes]).sum()
        fp += np.logical_and(proposed[spikes], ~target[spikes]).sum()
        fn += np.logical_and(~proposed[spikes], target[spikes]).sum()
    values = np.asarray(rows)
    names = ("Equal", "Huber", "RuleTemporal", "HardReject", "ContinuousRoute", "OracleFull")
    for metric_index, metric in enumerate(("PSNR", "SSIM", "NRMSE")):
        print("real-measurement " + metric + " " + ", ".join(f"{name}={values[:, column, metric_index].mean():.4f}" for column, name in enumerate(names)))
    if args.report_by_frame:
        for frame in np.unique(frames[:args.cases]):
            frame_mask = frames[:args.cases] == frame
            print(f"frame={int(frame)}, cases={int(frame_mask.sum())}")
            for metric_index, metric in enumerate(("PSNR", "SSIM", "NRMSE")):
                print("frame " + metric + " " + ", ".join(
                    f"{name}={values[frame_mask, column, metric_index].mean():.4f}"
                    for column, name in enumerate(names)
                ))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    print(f"controlled-corruption bad_type_accuracy={correct_bad / max(total_bad, 1):.4f}; gain_log_MAE={np.mean(gain_errors):.4f}; delay_MAE_samples={np.mean(delay_errors):.4f}; spike_mask_F1={f1:.4f}")
    print(f"channel_action_rate={actions / (args.cases * op.sensors):.4f}")
    print("reference=uncorrupted sparse 32-channel reconstruction from the same held-out real acquisition; input=real raw waveforms with synthetic labelled channel faults")


if __name__ == "__main__":
    main()
