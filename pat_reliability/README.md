# Package layout

The maintained continuous-calibration pipeline uses the modules at this
package level. In particular, `calibration_data.py` defines the mixed
gain/delay/spike protocol and `calibration_net.py` jointly predicts fault type,
continuous gain, continuous delay, and a time-local spike mask.

Historical experimental implementations live in `legacy/`. Moving source
files does not alter existing artifacts under `results/`, `checkpoints/`, or
`archives/`.
