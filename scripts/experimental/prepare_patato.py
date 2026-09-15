"""Export a compact, reproducible sparse-view PATATO real-measurement package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_frame_spec(value: str) -> tuple[int, ...]:
    # Parse comma-separated frame indices and inclusive ranges, for example 0-7,10.
    frames: list[int] = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        if '-' in item:
            start, stop = (int(part.strip()) for part in item.split('-', 1))
            if stop < start:
                raise argparse.ArgumentTypeError(f'invalid descending frame range: {item}')
            frames.extend(range(start, stop + 1))
        else:
            frames.append(int(item))
    if not frames:
        raise argparse.ArgumentTypeError('frame specification must not be empty')
    return tuple(sorted(set(frames)))


from pat_reliability.experimental.patato import (
    PatatoPreparation,
    frame_split,
    preprocess_acquisition,
    resize_reference,
    uniform_detector_subset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detectors", type=int, default=32)
    parser.add_argument("--output-samples", type=int, default=256)
    parser.add_argument("--sample-start", type=int, default=400)
    parser.add_argument("--sample-stop", type=int, default=1900)
    parser.add_argument("--reference-size", type=int, default=128)
    parser.add_argument("--max-cases", type=int, default=None, help="Optional smoke-test cap.")
    parser.add_argument('--train-frames', type=parse_frame_spec, default=parse_frame_spec('0-7'))
    parser.add_argument('--validation-frames', type=parse_frame_spec, default=parse_frame_spec('8-9'))
    parser.add_argument('--test-frames', type=parse_frame_spec, default=parse_frame_spec('10-11'))
    args = parser.parse_args()

    with h5py.File(args.input, "r") as source:
        raw = source["raw_data"]
        frames, wavelengths, detector_count, _ = raw.shape
        subset = uniform_detector_subset(detector_count, args.detectors)
        preparation = PatatoPreparation(
            source=str(args.input), detector_indices=tuple(int(v) for v in subset),
            sample_start=args.sample_start, sample_stop=args.sample_stop, output_samples=args.output_samples,
            train_frames=args.train_frames, validation_frames=args.validation_frames, test_frames=args.test_frames,
        )
        records = [(frame, wavelength) for frame in range(frames) for wavelength in range(wavelengths)]
        if args.max_cases is not None:
            records = records[: args.max_cases]
        signal_rows, reference_rows, frame_rows, wavelength_rows, scale_rows = [], [], [], [], []
        reference = source["recons/Reference Backprojection/0"]
        sample_rate = float(raw.attrs.get("fs", source.attrs["fs"]))
        for frame, wavelength in records:
            clean, scale = preprocess_acquisition(
                np.asarray(raw[frame, wavelength, subset]), sample_rate_hz=sample_rate,
                sample_start=preparation.sample_start, sample_stop=preparation.sample_stop,
                output_samples=preparation.output_samples, lower_hz=preparation.lower_hz,
                upper_hz=preparation.upper_hz,
            )
            signal_rows.append(clean)
            reference_rows.append(resize_reference(np.asarray(reference[frame, wavelength, 0]), args.reference_size))
            frame_rows.append(frame)
            wavelength_rows.append(wavelength)
            scale_rows.append(scale)

        frame_array = np.asarray(frame_rows, dtype=np.int16)
        metadata = preparation.metadata() | {
            "sample_rate_hz": sample_rate,
            "sound_speed_m_per_s": float(raw.attrs.get("speedofsound", source.attrs["speedofsound"])),
            "reference_dataset": "recons/Reference Backprojection/0",
            "reference_size": args.reference_size,
            "wavelength_nm": np.asarray(source["wavelengths"])[np.asarray(wavelength_rows)].tolist(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            signals=np.asarray(signal_rows, dtype=np.float32),
            reference=np.asarray(reference_rows, dtype=np.float32),
            geometry_m=np.asarray(source["GEOMETRY"])[subset, :2].astype(np.float32),
            detector_indices=subset,
            frames=frame_array,
            wavelengths=np.asarray(wavelength_rows, dtype=np.int16),
            split=frame_split(frame_array, preparation),
            acquisition_scales=np.asarray(scale_rows, dtype=np.float32),
            metadata=json.dumps(metadata),
        )
    print(f"wrote {args.output}")
    print(f"cases={len(records)}, signals={np.asarray(signal_rows).shape}, reference={np.asarray(reference_rows).shape}")
    print(f"detectors={subset.tolist()}, frame split: train={list(preparation.train_frames)}, val={list(preparation.validation_frames)}, test={list(preparation.test_frames)}")


if __name__ == "__main__":
    main()
