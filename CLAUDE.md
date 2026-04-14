# CLAUDE.md

Project: OCC (Optical Camera Communications) analog gain optimization.
Reproduction of Matus et al. 2020 + Ma 2024 adaptive damping.

## Architecture

```
src/occ_gain_opt/
├── config.py              CameraParams, DemodulationConfig, all shared types
├── demodulation.py        OOKDemodulator — THE ONLY demodulation entry point
├── experiment_loader.py   ExperimentLoader — loads ISO-Texp/ images
├── algorithms/            Pluggable gain optimization algorithms
│   ├── base.py            AlgorithmBase ABC
│   ├── single_shot.py     Matus Eq.7 single-shot
│   ├── adaptive_iter.py   Matus iterative with learning rate
│   ├── adaptive_damping.py Ma 5-state adaptive damping
│   └── ber_explore.py     BER-guided exploration
├── data_sources/          Data source abstraction + ROI strategies
├── experiments/           High-level experiment wrappers
├── hardware/              Hikvision ISAPI camera controller
├── cli.py                 occ-gain-opt CLI entry point
├── realtime.py            Single-step real-hardware interface
├── simulation.py          ExperimentSimulation
└── visualization.py       ResultVisualizer
```

## Demodulation — Single Entry Point

```python
from occ_gain_opt.demodulation import OOKDemodulator

demod = OOKDemodulator()
result = demod.demodulate(bgr_image)
result.packets[0]           # np.ndarray, 32-bit data
result.bit_period           # ~12 rows/bit
result.sync_positions_row   # sync header row positions
result.roi_mask             # data packet region mask
result.confidence           # 1.0 = two syncs found
```

NEVER duplicate demodulation logic inline. All scripts must use OOKDemodulator.

## Packet Structure

- Header: `[0, 1, 1, 1, 1, 1, 1, 0]` (8 bits, 6 consecutive bright rows)
- Data: 32-bit PRBS (Mseq)
- Repeat: 3× per image → 120 bits total
- Gap between sync headers = 34 bits (trailing 0 + 32 data + leading 0)

## Algorithm API

```python
from occ_gain_opt.algorithms import get as algo_get
from occ_gain_opt.config import CameraParams

algo = algo_get("single_shot")()
next_params = algo.compute_next_params(
    CameraParams(iso=35, exposure_us=27.9),
    roi_brightness=110.0
)
```

Registered algorithms: `single_shot`, `adaptive_iter`, `adaptive_damping`, `ber_explore`

## Key Conversions

```
gain_dB = 20·log10(ISO/100)
ISO 35  = -9.12 dB (typical starting point)
ISO 100 =  0.00 dB (base)
```

Target grayscale: 242.25 (255 × 0.95)

## Ground Truth

- `data/Mseq_32_original.csv` — 32-bit PRBS payload truth
- `data/Mseq_32_with_header.csv` — (header + payload) × 3

## Dataset

`ISO-Texp/` — 234 real experimental images across:
- `bubble/ISO/` (72), `bubble/Texp/` (42)
- `tap water/ISO/` (42), `tap water/Texp/` (24)
- `turbidity/ISO/` (18), `turbidity/Texp/` (36)

Filename format: `{exposure}_{iso}_p32_{condition}_{index}.jpg`
Example: `52600_640_p32_bubble_1_4_1.jpg` → exposure=1/52600s, ISO=640, PRBS-32

## Scripts

| Script | Purpose |
|--------|---------|
| `demo_demodulation.py` | OOK demodulation visual demo |
| `batch_demodulate.py` | Batch demod + BER + CSV + plots |
| `ber_analysis.py` | BER vs gain curves |
| `ber_comparison_plot.py` | BER comparison visualization |
| `validate_algorithms.py` | Three-algorithm validation |

## Coding Rules

- Demodulation: always use `OOKDemodulator`, never inline `_find_sync`/`_recover_data`/`polyfit_threshold`
- Camera params: always use `CameraParams` dataclass, never raw ISO/dB tuples
- ROI: prefer `sync_based` strategy, fallback to `auto_brightness`
- Imports: `sys.path.insert(0, "src")` in scripts, relative imports in package

## External References

- `real/自适应阻尼算法（马）/` — Ma algorithm reference implementation (do not delete)
- `example/` — MATLAB reference code (Rx_1.m, receive.m)
- `camera_isapi.py` — Hikvision ISAPI camera controller
- `realtime_experiment_app.py` — Streamlit experiment UI
