# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Reproduction of the paper *"Experimental Evaluation of an Analog Gain Optimization Algorithm in Optical Camera Communications"* (Matus et al., 2020). The goal is to automatically optimize camera analog gain in OCC systems so ROI grayscale values approach 255 (saturation) without exceeding it.

The codebase also implements Ma (2024) adaptive damping algorithm and provides a pluggable multi-algorithm framework for real-hardware experiments.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Commands

```bash
# Quick sanity test (simulation + algorithm layer)
occ-gain-opt --test

# Run built-in examples
occ-gain-opt --examples
occ-gain-opt --example-id 1

# Single-frame multi-algorithm advisor
occ-gain-opt advisor --image real/new.jpeg --iso 35 --exposure 27.9

# Three-algorithm closed-loop experiment (manual camera mode)
occ-gain-opt experiment --rtsp-url none --max-rounds 3

# Full reproducibility workflow (run in order)
python scripts/demo_demodulation.py        # OOK demodulation & sync head detection
python scripts/validate_algorithms.py      # All algorithms on real dataset
python scripts/ber_comparison_plot.py      # BER comparison visualization
```

## Architecture

The package lives in `src/occ_gain_opt/`. Key modules:

### Stable core (do not modify)
| Module | Role |
|--------|------|
| `config.py` | `CameraConfig`, `LEDConfig`, `ROIStrategy`, `OptimizationConfig`, `DemodulationConfig`, **`CameraParams`** (ISO + exposure) |
| `demodulation.py` | `OOKDemodulator` — full pipeline: LED column detection → row-mean → binarize → sync-head detection → bit sampling → packet ROI extraction |
| `performance_evaluation.py` | `PerformanceEvaluator` — MSE, PSNR, SSIM, SNR metrics |
| `experiment_loader.py` | `ExperimentLoader` — reads the `ISO-Texp/` real image dataset |

### Simulation layer (backward compat)
| Module | Role |
|--------|------|
| `data_acquisition.py` | `DataAcquisition` — simulates image capture + ROI selection |
| `gain_optimization.py` | `GainOptimizer` / `AdaptiveGainOptimizer` (Matus algorithms, simulation-coupled) |
| `simulation.py` | `ExperimentSimulation` — scenario runs using the simulation layer |
| `visualization.py` | `ResultVisualizer` — Matplotlib plots |
| `realtime.py` | `compute_next_gain()`, `RealtimeGainController` — single-step real-hardware interface |
| `examples.py` | Six standalone usage examples |
| `cli.py` | `main()` entry point + `advisor`/`experiment` subcommands |

### Algorithm layer ★ (new)
| Module | Role |
|--------|------|
| `algorithms/base.py` | `AlgorithmBase` ABC — `compute_next_params(CameraParams, brightness, ber)` |
| `algorithms/__init__.py` | `REGISTRY`, `@register`, `get(name)`, `list_algorithms()` |
| `algorithms/matus_single.py` | `MatusSingleAlgorithm` (name="matus_single") |
| `algorithms/matus_adaptive.py` | `MatusAdaptiveAlgorithm` (name="matus_adaptive") |
| `algorithms/ma_damping.py` | `MaDampingAlgorithm` (name="ma_damping") — full 5-state machine |

### Data source layer ★ (new)
| Module | Role |
|--------|------|
| `data_sources/base.py` | `DataSource` ABC — `get_frame()`, `current_params`, `set_params()` |
| `data_sources/roi.py` | ROI utilities: `create_center_roi_mask`, `create_auto_roi_mask`, `create_sync_based_roi_mask`, `compute_roi_stats` |
| `data_sources/simulated.py` | `SimulatedDataSource` |
| `data_sources/dataset.py` | `DatasetDataSource` (wraps ExperimentLoader, selects nearest-ISO image) |
| `data_sources/camera.py` | `CameraDataSource` (RTSP, ThreadedCamera) |

### Hardware layer ★ (new)
| Module | Role |
|--------|------|
| `hardware/camera_controller.py` | `CameraController` — manual prompt + Hikvision ISAPI |

### Experiments layer ★ (new)
| Module | Role |
|--------|------|
| `experiments/advisor.py` | `run_advisor()` — single-frame three-algorithm advisor |
| `experiments/closed_loop.py` | `ClosedLoopExperiment` — three-algorithm round-robin experiment |
| `experiments/batch_demod.py` | `batch_demodulate()` — batch OOK demodulation |

## Algorithm API (new, preferred for real hardware)

```python
from occ_gain_opt.algorithms import get as algo_get, list_algorithms
from occ_gain_opt.config import CameraParams

# List registered algorithms
print(list_algorithms())  # ['matus_single', 'matus_adaptive', 'ma_damping']

# Single-shot
algo = algo_get("matus_single")()
next_params = algo.compute_next_params(
    CameraParams(iso=35, exposure_us=27.9),
    roi_brightness=110.0
)
print(next_params)  # CameraParams(ISO=77.1, ...)

# Adaptive (stateful across calls)
algo = algo_get("ma_damping")(target_brightness=125.0)
for brightness in measured_brightnesses:
    params = algo.compute_next_params(current_params, brightness)
    current_params = params
```

## CameraParams

```python
from occ_gain_opt.config import CameraParams

p = CameraParams(iso=35, exposure_us=27.9)
p.gain_db       # -9.12 dB (20·log10(iso/100))
p.gain_linear   # 0.35
p.exposure_s    # 2.79e-5 s

# From dB
p2 = CameraParams.from_gain_db(-9.12, 27.9)
```

## Core Algorithms

**Single-shot** (Paper eq. 7):
```
G_opt(dB) = G_curr(dB) + 20·log10(Y_target / Y_curr)
```

**Iterative** (learning rate α, recommended 0.3–0.7):
```
G_{k+1} = G_k + α × 20·log10(Y_target / Y_k)
```

**Ma 5-state adaptive damping** (exposure + ISO, state I–V):
- State I:   initialization normalization
- State II:  proportional fast convergence
- State III: local linear fitting
- State IV:  gain unidirectional convergence
- State V:   gain clamping convergence

Target grayscale (Matus): `Y_target = 255 × 0.95 = 242.25`. Convergence typically in 1–4 iterations.
Target brightness (Ma): 125 (midpoint 0–255).

## ROI Strategies

The preferred strategy for validation is `SYNC_BASED`, which uses OOK sync-head detection to locate the exact data-packet stripe region. If sync detection fails, scripts automatically fall back to `AUTO_BRIGHTNESS`.

Protocol: OOK modulation, sync head = 8 consecutive 1-bits, data = 32-bit sequence (p32), bit period ≈ 12.5 rows/bit.

## ISO ↔ dB Conversion

```
gain_dB = 20·log10(ISO/100)
ISO 30  = -10.46 dB  (camera minimum)
ISO 100 =   0.00 dB  (base)
ISO 35  =  -9.12 dB  (typical lab starting point)
```

## Real Dataset

`ISO-Texp/` contains 234 real experimental images across three conditions (bubble, tap water, turbidity), each with ISO and exposure-time sub-experiments. This directory is `.gitignore`d.

## Reference Files

- `real/自适应阻尼算法（马）/test - 副本.py` — Ma algorithm original implementation (reference, do not delete)
- `real/data/` — captured frames
- `real/*.jpeg` — test images

## Output Directories

Results are written to `results/` (also gitignored):
- `results/demodulation/` — sync-head visualization panels
- `results/algorithm_validation/` — algorithm validation reports
