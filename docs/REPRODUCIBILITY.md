# Reproducibility notes

## Code used by the paper

Simulation:

- `scripts/final/train.py`
- `scripts/final/evaluate.py`
- `scripts/final/qualitative.py`
- top-level modules in `pat_reliability/`

Measured PATATO experiments:

- `scripts/experimental/prepare_patato.py`
- `scripts/experimental/train_patato_calibration.py`
- `scripts/experimental/evaluate_patato_calibration.py`
- `scripts/experimental/qualitative_patato.py`
- `pat_reliability/experimental/patato.py`
- `pat_reliability/experimental/routing.py`

`scripts/legacy/` and `pat_reliability/legacy/` contain earlier methods not used for the manuscript's ContinuousRoute results.

## Model seeds

- Simulation CalibrationNet: 51, 52, 53.
- Preclinical PATATO CalibrationNet: 61, 62, 63.
- In vivo PATATO CalibrationNet: 71, 72, 73.

The small final checkpoints are allowlisted by `.gitignore`. Smoke-test and earlier diagnostic checkpoints are excluded.

## Simulation protocol

- Image size: 128 x 128.
- Circular sensors: 32.
- Temporal samples: 320.
- Fault types: gain, delay, spike.
- Main fault count: 4.
- Main severity: 3.
- Layouts: random and contiguous.
- Reconstruction: 35 projected-gradient iterations, step size 0.18, TV weight 0.002.

## Measured protocol

The repository does not redistribute raw or processed PATATO measurements. Prepare them locally with `scripts.experimental.prepare_patato`.

Preclinical split: train frames 0-7, validation frames 8-9, test frames 10-11.

In vivo split: train frames 0-51, validation frames 52-63, test frames 64-75.

For both acquisitions, 32 detectors are uniformly selected from the native 256-detector array, and each waveform is filtered, cropped, and resampled to 256 temporal samples.

## Reference definition

For measured-waveform experiments, the reference is the uncorrupted sparse 32-channel reconstruction from the same held-out real acquisition. Controlled labelled faults are injected into real waveforms. Thus, measured PSNR quantifies recovery from injected channel mismatch, not agreement with a fully sampled anatomical ground truth.

## Local-only material

The following are deliberately excluded from Git: raw/processed data, old and intermediate results, archive bundles, virtual environments, caches, previews, and LaTeX build products.
