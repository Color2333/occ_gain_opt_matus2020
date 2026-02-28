# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Reproduction of the paper *"Experimental Evaluation of an Analog Gain Optimization Algorithm in Optical Camera Communications"* (Matus et al., 2020). The goal is to automatically optimize camera analog gain in OCC systems so ROI grayscale values approach 255 (saturation) without exceeding it.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Commands

```bash
# Run basic simulation
occ-gain-opt
python -m occ_gain_opt.cli

# Run built-in examples
occ-gain-opt --examples
occ-gain-opt --example-id 1

# Quick sanity test
occ-gain-opt --test

# Full reproducibility workflow (run in order)
python scripts/demo_demodulation.py            # OOK demodulation & sync head detection
python scripts/validate_algorithm_on_real_data.py   # Single-shot optimization on real data
python scripts/validate_iterative_algorithm.py      # Iterative vs single-shot comparison
```

## Architecture

The package lives in `src/occ_gain_opt/`. Key modules:

| Module | Role |
|--------|------|
| `config.py` | Dataclasses: `CameraConfig`, `LEDConfig`, `ROIStrategy` enum, `OptimizationConfig`, `DemodulationConfig` |
| `data_acquisition.py` | `DataAcquisition` class — simulates image capture; implements three ROI strategies: `CENTER`, `AUTO_BRIGHTNESS`, `SYNC_BASED` |
| `gain_optimization.py` | `GainOptimizer` (single-shot) and `AdaptiveGainOptimizer` (iterative, learning rate α) |
| `demodulation.py` | `OOKDemodulator` — full pipeline: LED column detection → row-mean → binarize → sync-head detection → bit sampling → packet ROI extraction |
| `performance_evaluation.py` | `PerformanceEvaluator` — MSE, PSNR, SSIM, SNR metrics |
| `simulation.py` | `ExperimentSimulation` — wraps the other modules for scenario runs |
| `visualization.py` | `ResultVisualizer` — Matplotlib plots (Chinese font: Hiragino Sans GB on macOS) |
| `experiment_loader.py` | `ExperimentLoader` — reads the `ISO-Texp/` real image dataset |
| `examples.py` | Six standalone usage examples |
| `cli.py` | `main()` entry point wired by `pyproject.toml` |
| `tools/` | Ad-hoc analysis and diagram scripts (not part of the core pipeline) |

## Core Algorithm

**Single-shot** (Paper eq. 7):
```
G_opt(dB) = G_curr(dB) + 20·log10(Y_target / Y_curr)
```

**Iterative** (learning rate α, recommended 0.3–0.7):
```
G_{k+1} = G_k + α × 20·log10(Y_target / Y_k)
```

Target grayscale: `Y_target = 255 × 0.95 = 242.25`. Convergence typically in 1–4 iterations.

## ROI Strategies

The preferred strategy for validation is `SYNC_BASED`, which uses OOK sync-head detection to locate the exact data-packet stripe region. If sync detection fails, scripts automatically fall back to `AUTO_BRIGHTNESS`.

Protocol: OOK modulation, sync head = 8 consecutive 1-bits, data = 32-bit sequence (p32), bit period ≈ 12.5 rows/bit.

## Real Dataset

`ISO-Texp/` contains 234 real experimental images across three conditions (bubble, tap water, turbidity), each with ISO and exposure-time sub-experiments. This directory is `.gitignore`d.

## Output Directories

Results are written to `results/` (also gitignored):
- `results/demodulation/` — sync-head visualization panels
- `results/algorithm_validation/` — single-shot accuracy reports
- `results/iterative_validation/` — iterative vs. single-shot comparison plots and reports
