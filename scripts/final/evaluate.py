"""Evaluate the final continuous channel-calibration method.

Protocols:
  mixed       Main mixed-fault reconstruction comparison (default).
  hard        Adds the full-channel HardReject baseline.
  clean       Checks that routing does not harm clean acquisitions.
  parameters  Reports held-out gain, delay, and spike-mask accuracy.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from multiprocessing import Pool
from time import perf_counter

import numpy as np
import torch

from pat_reliability.calibration_data import calibration_case
from pat_reliability.calibration_net import CalibrationNet
from pat_reliability.calibration import apply_oracle_correction, apply_predicted_correction
from pat_reliability.config import load_config
from pat_reliability.core import WaveOperator, psnr
from pat_reliability.metrics import nrmse, ssim
from pat_reliability.phantom_families import sample_phantom
from pat_reliability.quality_net import channel_inputs, robust_prediction
from pat_reliability.temporal_routing import reconstruct_temporal, spike_time_weights
from pat_reliability.robust_reconstruction import reconstruct_huber

_OP = _NET = _ARGS = None
_METRIC_NAMES = ("PSNR", "SSIM", "NRMSE")


def _init_worker(config, checkpoint, args):
    global _OP, _NET, _ARGS
    _OP, _ARGS = WaveOperator(config), args
    state = torch.load(checkpoint, map_location="cpu")
    _NET = CalibrationNet()
    _NET.load_state_dict(state["state_dict"])
    _NET.eval()


def _scores(reference, estimate):
    return (psnr(reference, estimate), ssim(reference, estimate), nrmse(reference, estimate))


def _predict_actions(observed, predicted):
    with torch.no_grad():
        output = _NET(torch.from_numpy(channel_inputs(observed, predicted)))
    probabilities = torch.softmax(output["type_logits"], dim=1).numpy()
    return output, probabilities.argmax(axis=1), probabilities.max(axis=1) >= _ARGS.confidence


def _continuous_correction(observed, predicted, output, labels, confident):
    no_gain = _ARGS.ablation in ("no_gain", "no_gain_no_residual_gate")
    no_delay = _ARGS.ablation in ("no_delay", "no_delay_no_residual_gate")
    whole_spike = _ARGS.ablation in ("whole_channel_spike", "whole_channel_spike_no_residual_gate")
    residual_gate = _ARGS.ablation not in (
        "no_residual_gate", "no_gain_no_residual_gate",
        "no_delay_no_residual_gate", "whole_channel_spike_no_residual_gate",
    )
    return apply_predicted_correction(
        observed,
        output,
        labels,
        confident,
        predicted=predicted,
        correct_gain=not no_gain,
        correct_delay=not no_delay,
        whole_channel_spike=whole_spike,
        residual_gate=residual_gate,
    )


def _oracle_correction(observed, truth):
    return apply_oracle_correction(observed, truth)

def _mixed_case(index, include_hard, include_huber=False):
    rng = np.random.default_rng(_ARGS.seed + index)
    image, observed, truth = calibration_case(_OP, rng, fault_count=_ARGS.fault_count, severity=_ARGS.severity, layout=_ARGS.layout, phantom_family=_ARGS.phantom_family, fault_type=_ARGS.fault_type)
    predicted = robust_prediction(_OP, observed)
    output, labels, confident = _predict_actions(observed, predicted)
    corrected, weights = _continuous_correction(observed, predicted, output, labels, confident)
    oracle_data, oracle_weights = _oracle_correction(observed, truth)
    reconstructions = [
        reconstruct_temporal(_OP, observed, np.ones_like(observed), _OP.cfg),
        reconstruct_temporal(_OP, observed, spike_time_weights(observed, predicted), _OP.cfg),
    ]
    if include_huber:
        reconstructions.insert(1, reconstruct_huber(_OP, observed, _OP.cfg))
    if include_hard:
        hard_weights = np.ones_like(observed)
        hard_weights[(labels != 0) & confident] = 0.05
        reconstructions.append(reconstruct_temporal(_OP, observed, hard_weights, _OP.cfg))
    reconstructions.extend((
        reconstruct_temporal(_OP, corrected, weights, _OP.cfg),
        reconstruct_temporal(_OP, oracle_data, oracle_weights, _OP.cfg),
    ))
    return np.asarray([_scores(image, reconstruction) for reconstruction in reconstructions])


def _runtime_case(index):
    """Time the route as four interpretable stages for one held-out case."""
    rng = np.random.default_rng(_ARGS.seed + index)
    image, observed, _ = calibration_case(_OP, rng, fault_count=_ARGS.fault_count, severity=_ARGS.severity, layout=_ARGS.layout, phantom_family=_ARGS.phantom_family, fault_type=_ARGS.fault_type)
    started = perf_counter(); predicted = robust_prediction(_OP, observed); bootstrap_seconds = perf_counter() - started
    started = perf_counter(); output, labels, confident = _predict_actions(observed, predicted); corrected, weights = _continuous_correction(observed, predicted, output, labels, confident); routing_seconds = perf_counter() - started
    started = perf_counter(); equal = reconstruct_temporal(_OP, observed, np.ones_like(observed), _OP.cfg); equal_seconds = perf_counter() - started
    started = perf_counter(); routed = reconstruct_temporal(_OP, corrected, weights, _OP.cfg); routed_seconds = perf_counter() - started
    return np.asarray((bootstrap_seconds, routing_seconds, equal_seconds, routed_seconds, *_scores(image, equal), *_scores(image, routed)))


def _clean_case(index):
    rng = np.random.default_rng(_ARGS.seed + index)
    image = sample_phantom(_OP.cfg.image_size, rng, family=_ARGS.phantom_family)
    clean = _OP.forward(image)
    observed = clean + rng.normal(scale=_OP.cfg.noise_std * max(clean.std(), 1e-12), size=clean.shape)
    predicted = robust_prediction(_OP, observed)
    output, labels, confident = _predict_actions(observed, predicted)
    corrected, weights = _continuous_correction(observed, predicted, output, labels, confident)
    equal = reconstruct_temporal(_OP, observed, np.ones_like(observed), _OP.cfg)
    routed = reconstruct_temporal(_OP, corrected, weights, _OP.cfg)
    return np.asarray((*_scores(image, equal), *_scores(image, routed), np.mean((labels != 0) & confident)))


def _run_parallel(worker, args, *worker_args):
    with Pool(args.workers, _init_worker, (load_config(args.config), args.checkpoint, args)) as pool:
        return np.asarray(pool.starmap(worker, ((index, *worker_args) for index in range(args.cases))))


def _print_metrics(prefix, samples, names):
    for metric_index, metric_name in enumerate(_METRIC_NAMES):
        values = samples[:, :, metric_index]
        print(f"{prefix} {metric_name} " + ", ".join(
            f"{name}={values[:, column].mean():.4f}" for column, name in enumerate(names)
        ))

def _evaluate_reconstruction(args, include_hard, include_huber=False):
    started = perf_counter()
    scores = _run_parallel(_mixed_case, args, include_hard, include_huber)
    elapsed = perf_counter() - started
    if include_huber:
        names = ("Equal", "Huber", "RuleTemporal", "ContinuousRoute", "OracleFull")
    elif include_hard:
        names = ("Equal", "RuleTemporal", "HardReject", "ContinuousRoute", "OracleFull")
    else:
        names = ("Equal", "RuleTemporal", "ContinuousRoute", "OracleFull")
    _print_metrics("mixed", scores, names)
    route_index = names.index("ContinuousRoute")
    delta = scores[:, route_index, 0] - scores[:, 0, 0]
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap = rng.choice(delta, size=(args.bootstrap_samples, delta.size), replace=True).mean(axis=1)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    print(f"paired PSNR delta ContinuousRoute-Equal mean={delta.mean():+.4f}, 95CI=[{lower:+.4f}, {upper:+.4f}], win_rate={(delta > 0).mean():.4f}, bootstrap_samples={args.bootstrap_samples}")
    if args.save_scores is not None:
        args.save_scores.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_scores, scores=scores, method_names=np.asarray(names),
            metric_names=np.asarray(_METRIC_NAMES), test_seed=args.seed,
            fault_count=args.fault_count, severity=args.severity,
            layout=args.layout, fault_type=args.fault_type,
            ablation=args.ablation,
        )
        print(f"saved per_case_scores={args.save_scores}")
    print(f"runtime wall_seconds={elapsed:.2f}, seconds_per_case={elapsed / args.cases:.4f}, workers={args.workers}")

def _evaluate_clean(args):
    started = perf_counter()
    scores = _run_parallel(_clean_case, args)
    elapsed = perf_counter() - started
    for metric_index, metric_name in enumerate(_METRIC_NAMES):
        equal, routed = scores[:, metric_index], scores[:, metric_index + 3]
        print(f"clean {metric_name} Equal={equal.mean():.4f}, ContinuousRoute={routed.mean():.4f}, Delta={(routed - equal).mean():+.4f}")
    print(f"clean channel_action_rate={scores[:, 6].mean():.4f}")
    print(f"runtime wall_seconds={elapsed:.2f}, seconds_per_case={elapsed / args.cases:.4f}, workers={args.workers}")


def _evaluate_runtime(args):
    started = perf_counter(); samples = _run_parallel(_runtime_case, args); elapsed = perf_counter() - started
    names = ("bootstrap_prediction", "network_and_correction", "equal_reconstruction", "routed_reconstruction")
    for index, name in enumerate(names): print(f"runtime {name}_seconds_per_case={samples[:, index].mean():.4f}")
    equal_psnr, routed_psnr = samples[:, 4], samples[:, 7]
    print(f"runtime PSNR Equal={equal_psnr.mean():.4f}, ContinuousRoute={routed_psnr.mean():.4f}, Delta={(routed_psnr - equal_psnr).mean():+.4f}")
    print(f"runtime network_parameters={sum(parameter.numel() for parameter in CalibrationNet().parameters())}")
    print(f"runtime wall_seconds={elapsed:.2f}, wall_seconds_per_case={elapsed / args.cases:.4f}, workers={args.workers}")


def _evaluate_parameters(args):
    state = torch.load(args.checkpoint, map_location="cpu")
    network = CalibrationNet()
    network.load_state_dict(state["state_dict"])
    network.eval()
    operator = WaveOperator(load_config(args.config))
    rng = np.random.default_rng(args.seed)
    gains, delays = [], []
    true_positive = false_positive = false_negative = 0
    correct_types = total_bad = 0
    for _ in range(args.cases):
        _, observed, truth = calibration_case(operator, rng, fault_count=args.fault_count, severity=args.severity, layout=args.layout, phantom_family=args.phantom_family, fault_type=args.fault_type)
        predicted = robust_prediction(operator, observed)
        with torch.no_grad():
            output = network(torch.from_numpy(channel_inputs(observed, predicted)))
        labels = output["type_logits"].argmax(dim=1).numpy()
        bad = truth["type"] > 0
        correct_types += (labels[bad] == truth["type"][bad]).sum()
        total_bad += bad.sum()
        gain_mask, delay_mask, spike_mask = truth["type"] == 1, truth["type"] == 2, truth["type"] == 3
        if gain_mask.any():
            gains.extend(np.abs(output["log_gain"].numpy()[gain_mask] - truth["log_gain"][gain_mask]))
        if delay_mask.any():
            delays.extend(np.abs(output["delay"].numpy()[delay_mask] - truth["delay"][delay_mask]))
        predicted_mask = torch.sigmoid(output["spike_logits"]).numpy() >= 0.5
        target_mask = truth["spike_mask"] >= 0.5
        true_positive += np.logical_and(predicted_mask[spike_mask], target_mask[spike_mask]).sum()
        false_positive += np.logical_and(predicted_mask[spike_mask], ~target_mask[spike_mask]).sum()
        false_negative += np.logical_and(~predicted_mask[spike_mask], target_mask[spike_mask]).sum()
    f1 = 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)
    print(f"bad_type_accuracy={correct_types / total_bad:.4f}")
    print(f"gain_log_MAE={np.mean(gains):.4f}")
    print(f"spike_mask_F1={f1:.4f}")
    print(f"delay_MAE_samples={np.mean(delays):.4f}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("mixed", "robust", "hard", "clean", "parameters", "runtime"), default="mixed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int, default=81051, help="Backward-compatible test-set seed.")
    parser.add_argument("--test-seed", type=int, help="Explicit shared test-set seed; overrides --seed.")
    parser.add_argument("--fault-count", type=int, default=4, help="Absolute number of corrupted channels (legacy default: 4).")
    parser.add_argument("--fault-fraction", type=float, help="Override --fault-count with a fixed fraction of active sensors.")
    parser.add_argument("--severity", type=float, default=3)
    parser.add_argument("--phantom-family", choices=("id", "ood"), default="id")
    parser.add_argument("--layout", choices=("random", "contiguous"), default="random")
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--save-scores", type=Path, help="Optional .npz destination for per-case scores.")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument("--fault-type", choices=("mixed", "gain", "delay", "spike"), default="mixed", help="Inject one fault type per corrupted channel, or use the default mixed-fault protocol.")
    parser.add_argument("--ablation", choices=("full", "no_gain", "no_delay", "whole_channel_spike", "no_residual_gate", "no_gain_no_residual_gate", "no_delay_no_residual_gate", "whole_channel_spike_no_residual_gate"), default="no_residual_gate", help="Ablation choice. The default is the final route without the harmful residual-acceptance check.")
    args = parser.parse_args()
    if args.test_seed is not None:
        args.seed = args.test_seed
    config = load_config(args.config)
    args.fault_type = {"mixed": None, "gain": 1, "delay": 2, "spike": 3}[args.fault_type]
    if args.fault_fraction is not None:
        if not 0 < args.fault_fraction < 1:
            parser.error("--fault-fraction must be strictly between 0 and 1.")
        args.fault_count = max(1, int(round(args.fault_fraction * config.sensors)))
    print(f"protocol={args.protocol}, test_seed={args.seed}, cases={args.cases}, fault_count={args.fault_count}, fault_fraction={args.fault_count / config.sensors:.4f}, phantom_family={args.phantom_family}, fault_type={args.fault_type or "mixed"}, ablation={args.ablation}, severity={args.severity}, layout={args.layout}")
    if args.protocol == "mixed":
        _evaluate_reconstruction(args, include_hard=False)
    elif args.protocol == "robust":
        _evaluate_reconstruction(args, include_hard=False, include_huber=True)
    elif args.protocol == "hard":
        _evaluate_reconstruction(args, include_hard=True)
    elif args.protocol == "clean":
        _evaluate_clean(args)
    elif args.protocol == "runtime":
        _evaluate_runtime(args)
    else:
        _evaluate_parameters(args)


if __name__ == "__main__":
    main()
