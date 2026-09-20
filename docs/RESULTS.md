# Paper results

These values are transcribed from the tables used by `paper/draft/main.tex`. Learned methods are reported for three independently trained models.

## Main reconstruction PSNR

Four severity-3 mixed faulty channels. `R` and `C` denote random and contiguous layouts.

| Method | Seed | Sim R | Sim C | Preclinical R | Preclinical C | In vivo R | In vivo C |
|---|---:|---:|---:|---:|---:|---:|---:|
| Equal | fixed | 26.553 | 26.415 | 27.405 | 27.613 | 25.584 | 25.603 |
| RuleTemporal | fixed | 26.773 | 26.627 | 24.603 | 24.629 | 22.550 | 22.525 |
| Huber | fixed | 26.500 | 26.359 | 22.219 | 22.048 | 20.526 | 20.492 |
| OracleFull | fixed | 27.475 | 27.334 | 60.861 | 53.730 | 58.585 | 50.606 |
| HardReject | 1 | 26.511 | 26.192 | 32.049 | 31.278 | 28.614 | 28.385 |
| HardReject | 2 | 26.521 | 26.199 | 31.988 | 31.227 | 28.523 | 28.507 |
| HardReject | 3 | 26.502 | 26.175 | 31.887 | 31.204 | 28.864 | 28.589 |
| **ContinuousRoute** | **1** | **27.367** | **27.196** | **37.025** | **37.128** | **30.057** | **30.134** |
| **ContinuousRoute** | **2** | **27.313** | **27.140** | **36.681** | **36.623** | **30.319** | **30.559** |
| **ContinuousRoute** | **3** | **27.300** | **27.127** | **34.772** | **35.207** | **30.507** | **30.744** |

## Calibration and clean-input behavior

| Measure | Simulation seed 51 | Simulation seed 52 | Simulation seed 53 |
|---|---:|---:|---:|
| Fault-type accuracy | 0.800 | 0.793 | 0.815 |
| Log-gain MAE | 0.149 | 0.151 | 0.159 |
| Delay MAE (samples) | 1.979 | 2.193 | 2.004 |
| Spike-mask F1 | 0.985 | 0.990 | 0.991 |
| Clean PSNR change (dB) | -0.018 | -0.025 | -0.066 |
| Clean action rate (%) | 0.97 | 0.66 | 1.19 |

| In vivo clean measure | Model 71 | Model 72 | Model 73 |
|---|---:|---:|---:|
| PSNR (dB) | 77.40 | 92.71 | 80.68 |
| SSIM | 0.9834 | 0.9921 | 0.9841 |
| NRMSE | 0.0997 | 0.0559 | 0.0955 |
| Action rate (%) | 2.66 | 1.59 | 1.98 |

## Measured-waveform robustness

Random severity-3 mixed faults. Preclinical frame rows use four faulty channels.

| Condition | Equal | CR-1 | CR-2 | CR-3 |
|---|---:|---:|---:|---:|
| Preclinical, 2 faulty | 31.962 | 35.142 | 35.104 | 39.721 |
| Preclinical, 4 faulty | 27.405 | 37.025 | 36.681 | 34.772 |
| Preclinical, 8 faulty | 24.186 | 28.742 | 28.824 | 31.558 |
| Preclinical, frame 10 | 27.072 | 37.432 | 36.285 | 34.494 |
| Preclinical, frame 11 | 27.800 | 36.544 | 37.149 | 35.102 |
| In vivo, 2 faulty | 29.897 | 33.937 | 35.288 | 33.264 |
| In vivo, 4 faulty | 25.584 | 30.057 | 30.319 | 30.507 |
| In vivo, 8 faulty | 22.594 | 27.589 | 27.662 | 28.022 |

## Interpretation

- ContinuousRoute exceeds Equal for both layouts across simulation, preclinical, and in vivo experiments.
- Generic residual weighting and Huber fitting do not distinguish amplitude, delay, and transient faults and are weaker on the measured experiments.
- Full-channel rejection removes useful samples; localized correction and weighting retain more information.
- OracleFull uses injected fault parameters and is an upper bound, not a deployable method.

Values are transcribed from the manuscript tables.
