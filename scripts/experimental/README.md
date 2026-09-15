# External-measurement supplement

`prepare_patato.py` exports a compact, reproducible package from the PATATO
preclinical raw HDF5 data.  It is intentionally separate from `scripts/final/`:
the main results use a circular full-wave simulator, whereas PATATO uses a
partial-view non-circular experimental array with its own acquisition response.

The exported package uses a fixed 32-of-256 detector subset; removes the
per-channel initial DC baseline; applies a 50 kHz--7 MHz FFT bandpass; discards
the early 0--10 us trigger-dominated interval; and rescales each acquisition
once globally, retaining inter-channel amplitude differences.  Frames 0--7,
8--9, and 10--11 are the train, validation, and held-out test splits.

This supplement must be reported as **controlled synthetic channel corruptions
of real measured waveforms**, not as naturally occurring sensor-fault labels.

qualitative_patato.py accepts repeated --case arguments. A single case
preserves the original output layout; repeated cases additionally create a
single reconstruction montage and one waveform panel per case. This keeps
paper-ready qualitative material with the PATATO adapter rather than adding a
separate plotting script.
