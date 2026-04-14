#!/usr/bin/env python3
"""
Batch Demodulation and BER Analysis Script

Processes ALL p32 images in ISO-Texp dataset using the correct demodulation algorithm.
Computes BER against ground truth, extracts ROI statistics, and generates visualizations.

Usage:
    python scripts/batch_demodulate.py
"""

import sys
import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import re

import numpy as np
import cv2
import matplotlib.pyplot as plt

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from occ_gain_opt.experiment_loader import ExperimentLoader, ExperimentImage
from occ_gain_opt.demodulation import OOKDemodulator


def load_ground_truth(csv_path: str) -> np.ndarray:
    """Load 32-bit PRBS ground truth sequence."""
    import pandas as pd

    df = pd.read_csv(csv_path, skiprows=6, header=None, names=["xpos", "value"])
    bits = df.iloc[:, 1].to_numpy().astype(np.uint8)
    assert len(bits) == 32, f"Expected 32 bits, got {len(bits)}"
    return bits


# OOKDemodulator instance for demodulate_image wrapper
_demod = OOKDemodulator()


def demodulate_image(
    img_bgr: np.ndarray, truth_32: Optional[np.ndarray] = None
) -> Dict:
    """Thin wrapper around OOKDemodulator to match the expected return format."""
    result = _demod.demodulate(img_bgr)
    status = "ok" if result.packets else "no_packets"
    bits = result.packets[0] if result.packets else np.array([], dtype=np.uint8)
    ber = None
    if truth_32 is not None and len(bits) == 32:
        ber = float(np.sum(bits != truth_32)) / 32.0
    return {
        "status": status,
        "ber": ber,
        "bits": bits,
        "equ": result.bit_period,
        "method": "direct",
        "sync1": (result.sync_positions_row[0], 0, 0)
        if result.sync_positions_row
        else None,
        "sync2": (result.sync_positions_row[1], 0, 0)
        if len(result.sync_positions_row) > 1
        else None,
        "bright_mean": result.stats.get("bright_mean", 0),
        "dark_mean": result.stats.get("dark_mean", 0),
    }


@dataclass
class ImageResult:
    filepath: str
    filename: str
    experiment: str
    image_type: str
    condition: str
    exposure_denom: int
    iso: float
    index: int
    equivalent_gain_db: float

    status: str
    ber: float
    method: str
    equ: float
    sync1_start: int
    sync1_end: int
    sync1_len: int
    sync2_start: int
    sync2_end: int
    sync2_len: int

    mean_gray: float
    bright_mean: float
    dark_mean: float
    eye_opening: float
    snr: float

    is_perfect: bool


def parse_filename_detailed(filename: str) -> Optional[Dict]:
    name = filename.replace(".jpg", "")
    parts = name.split("_")

    if len(parts) < 5:
        return None

    try:
        exposure_denom = int(parts[0])
        iso = float(parts[1])
        seq_type = parts[2]

        index = int(parts[-1])

        condition = "_".join(parts[3:-1])

        return {
            "exposure_denom": exposure_denom,
            "iso": iso,
            "seq_type": seq_type,
            "condition": condition,
            "index": index,
        }
    except (ValueError, IndexError):
        return None


def process_image(filepath: str, truth_32: np.ndarray) -> Optional[ImageResult]:
    filename = os.path.basename(filepath)
    info = parse_filename_detailed(filename)

    if info is None:
        print(f"  Warning: Could not parse filename: {filename}")
        return None

    if info["seq_type"] != "p32":
        return None

    img = cv2.imread(filepath)
    if img is None:
        print(f"  Warning: Could not load image: {filename}")
        return None

    exp_img = ExperimentImage(
        filepath=filepath,
        exposure_time=1.0 / info["exposure_denom"],
        iso=info["iso"],
        sequence_length=32,
        condition=info["condition"],
        index=info["index"],
        image_type="ISO",
    )

    path_parts = filepath.split(os.sep)
    image_type = "ISO"
    experiment = "unknown"
    for i, part in enumerate(path_parts):
        if part in ["ISO", "Texp"]:
            image_type = part
            if i > 0:
                experiment = path_parts[i - 1]

    exp_img.image_type = image_type
    equivalent_gain = exp_img.calculate_equivalent_gain()

    result = demodulate_image(img, truth_32)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mean_gray = float(np.mean(gray))

    # Eye opening and SNR
    bright_mean = result["bright_mean"]
    dark_mean = result["dark_mean"]
    eye_opening = bright_mean - dark_mean

    row_mean = np.mean(gray, axis=1)
    x = np.arange(1, gray.shape[0] + 1, dtype=np.float64)
    coeffs = np.polyfit(x, row_mean, 3)
    yfit = np.polyval(coeffs, x)
    yy = row_mean - yfit
    rr = (yy > 0).astype(np.uint8)

    dark_pixels = gray[rr == 0]
    std_dark = float(np.std(dark_pixels)) if len(dark_pixels) > 0 else 1.0
    snr = eye_opening / std_dark if std_dark > 0 else 0.0

    sync1 = result.get("sync1") or (0, 0, 0)
    sync2 = result.get("sync2") or (0, 0, 0)

    return ImageResult(
        filepath=filepath,
        filename=filename,
        experiment=experiment,
        image_type=image_type,
        condition=info["condition"],
        exposure_denom=info["exposure_denom"],
        iso=info["iso"],
        index=info["index"],
        equivalent_gain_db=equivalent_gain,
        status=result["status"],
        ber=result["ber"] if result["ber"] is not None else 0.5,
        method=result["method"],
        equ=result["equ"],
        sync1_start=sync1[0],
        sync1_end=sync1[1],
        sync1_len=sync1[2],
        sync2_start=sync2[0],
        sync2_end=sync2[1],
        sync2_len=sync2[2],
        mean_gray=mean_gray,
        bright_mean=bright_mean,
        dark_mean=dark_mean,
        eye_opening=eye_opening,
        snr=snr,
        is_perfect=(result["ber"] is not None and result["ber"] == 0),
    )


def process_all_images(dataset_dir: str, truth_csv: str) -> List[ImageResult]:
    truth_32 = load_ground_truth(truth_csv)
    print(f"Loaded ground truth: {truth_32}")
    print(
        f"  Header pattern: {[int(b) for b in truth_32[:8]]} (should be [0,1,1,1,1,1,1,0])"
    )

    results = []

    total_found = 0
    for exp_type in ["bubble", "tap water", "turbidity"]:
        for img_type in ["ISO", "Texp"]:
            img_dir = Path(dataset_dir) / exp_type / img_type
            if not img_dir.exists():
                continue

            for filepath in img_dir.glob("*.jpg"):
                total_found += 1
                info = parse_filename_detailed(filepath.name)
                if info and info["seq_type"] == "p32":
                    result = process_image(str(filepath), truth_32)
                    if result:
                        results.append(result)
                        print(
                            f"  Processed: {filepath.name} -> BER={result.ber:.4f} ({result.method})"
                        )

    print(f"\nProcessed {len(results)} p32 images (found {total_found} total files)")
    return results


def generate_console_summary(results: List[ImageResult]):
    print("\n" + "=" * 70)
    print("=== Batch Demodulation Results ===")
    print("=" * 70)

    groups = {}
    for r in results:
        key = f"{r.experiment}/{r.image_type}"
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    total_perfect = 0
    total_images = 0
    total_ber_sum = 0

    print(f"\n{'Group':<20} {'n':>5}  {'avg_BER':>10}  {'perfect':>15}  {'equ':>8}")
    print("-" * 70)

    for key in sorted(groups.keys()):
        group_results = groups[key]
        n = len(group_results)
        perfect = sum(1 for r in group_results if r.is_perfect)
        avg_ber = sum(r.ber for r in group_results) / n if n > 0 else 0
        avg_equ = sum(r.equ for r in group_results) / n if n > 0 else 0

        total_perfect += perfect
        total_images += n
        total_ber_sum += sum(r.ber for r in group_results)

        perfect_str = f"{perfect}/{n}"
        print(f"{key:<20} {n:>5}  {avg_ber:>10.4f}  {perfect_str:>15}  {avg_equ:>8.1f}")

    print("-" * 70)
    overall_ber = total_ber_sum / total_images if total_images > 0 else 0
    print(
        f"{'Overall':<20} {total_images:>5}  {overall_ber:>10.4f}  {str(total_perfect) + '/' + str(total_images):>15}"
    )
    print("=" * 70)


def save_csv(results: List[ImageResult], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "filename",
        "experiment",
        "image_type",
        "condition",
        "exposure_denom",
        "iso",
        "index",
        "equivalent_gain_db",
        "status",
        "ber",
        "method",
        "equ",
        "sync1_start",
        "sync1_end",
        "sync1_len",
        "sync2_start",
        "sync2_end",
        "sync2_len",
        "mean_gray",
        "bright_mean",
        "dark_mean",
        "eye_opening",
        "snr",
        "is_perfect",
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = asdict(r)
            writer.writerow(row)

    print(f"CSV saved to: {output_path}")


def generate_report(results: List[ImageResult], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    groups = {}
    for r in results:
        key = f"{r.experiment}/{r.image_type}"
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("BATCH DEMODULATION AND BER ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total Images Processed: {len(results)}\n")
        f.write(f"Ground Truth: 32-bit PRBS sequence\n")
        f.write(f"Expected Header: [0, 1, 1, 1, 1, 1, 1, 0] (6 consecutive 1s)\n\n")

        f.write("-" * 70 + "\n")
        f.write("SUMMARY BY GROUP\n")
        f.write("-" * 70 + "\n\n")

        for key in sorted(groups.keys()):
            group_results = groups[key]
            n = len(group_results)
            perfect = sum(1 for r in group_results if r.is_perfect)
            avg_ber = sum(r.ber for r in group_results) / n if n > 0 else 0
            avg_equ = sum(r.equ for r in group_results) / n if n > 0 else 0
            avg_eye = sum(r.eye_opening for r in group_results) / n if n > 0 else 0
            avg_snr = sum(r.snr for r in group_results) / n if n > 0 else 0

            f.write(f"\n{key}:\n")
            f.write(f"  Images: {n}\n")
            f.write(f"  Perfect (BER=0): {perfect}/{n} ({100 * perfect / n:.1f}%)\n")
            f.write(f"  Average BER: {avg_ber:.4f}\n")
            f.write(f"  Average Bit Period (equ): {avg_equ:.1f} rows\n")
            f.write(f"  Average Eye Opening: {avg_eye:.1f}\n")
            f.write(f"  Average SNR: {avg_snr:.2f}\n")

            methods = {}
            for r in group_results:
                if r.method not in methods:
                    methods[r.method] = 0
                methods[r.method] += 1
            f.write(
                f"  Demodulation Methods: {', '.join(f'{k}={v}' for k, v in methods.items())}\n"
            )

        f.write("\n" + "-" * 70 + "\n")
        f.write("DETAILED RESULTS PER IMAGE\n")
        f.write("-" * 70 + "\n\n")

        for r in results:
            f.write(f"File: {r.filename}\n")
            f.write(f"  Experiment: {r.experiment}/{r.image_type}\n")
            f.write(f"  ISO: {r.iso}, Exposure: 1/{r.exposure_denom} s\n")
            f.write(f"  Gain: {r.equivalent_gain_db:.2f} dB\n")
            f.write(f"  Status: {r.status}, Method: {r.method}\n")
            f.write(f"  BER: {r.ber:.4f}, Perfect: {'Yes' if r.is_perfect else 'No'}\n")
            f.write(f"  Bit Period: {r.equ:.1f} rows\n")
            f.write(f"  Mean Gray: {r.mean_gray:.1f}\n")
            f.write(f"  Bright/Dark: {r.bright_mean:.1f} / {r.dark_mean:.1f}\n")
            f.write(f"  Eye Opening: {r.eye_opening:.1f}, SNR: {r.snr:.2f}\n")
            f.write(
                f"  Sync Headers: [{r.sync1_start}:{r.sync1_end}] len={r.sync1_len}, "
                f"[{r.sync2_start}:{r.sync2_end}] len={r.sync2_len}\n"
            )
            f.write("\n")

    print(f"Report saved to: {output_path}")


def generate_plots(results: List[ImageResult], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    groups = {}
    for r in results:
        key = f"{r.experiment}/{r.image_type}"
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, key in enumerate(sorted(groups.keys())):
        if idx >= 6:
            break
        group_results = groups[key]
        ax = axes[idx]

        gains = [r.equivalent_gain_db for r in group_results]
        bers = [r.ber for r in group_results]

        sorted_pairs = sorted(zip(gains, bers), key=lambda x: x[0])
        gains_sorted = [p[0] for p in sorted_pairs]
        bers_sorted = [p[1] for p in sorted_pairs]

        ax.scatter(gains_sorted, bers_sorted, alpha=0.6, s=30)
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5, label="BER=0 (perfect)")
        ax.axhline(
            y=0.5, color="orange", linestyle=":", alpha=0.5, label="BER=0.5 (failed)"
        )
        ax.set_xlabel("Equivalent Gain (dB)")
        ax.set_ylabel("BER")
        ax.set_title(f"{key}: BER vs Gain\n(n={len(group_results)})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if len(groups) < 6:
            for idx in range(len(groups), 6):
                axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "ber_vs_gain.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Plot saved: {os.path.join(output_dir, 'ber_vs_gain.png')}")

    fig, ax = plt.subplots(figsize=(12, 6))

    keys = sorted(groups.keys())
    n_groups = len(keys)
    x = np.arange(n_groups)

    perfect_rates = []
    avg_bers = []
    n_images = []

    for key in keys:
        group_results = groups[key]
        n = len(group_results)
        perfect = sum(1 for r in group_results if r.is_perfect)
        avg_ber = sum(r.ber for r in group_results) / n if n > 0 else 0

        n_images.append(n)
        perfect_rates.append(100 * perfect / n)
        avg_bers.append(avg_ber)

    width = 0.35
    bars1 = ax.bar(
        x - width / 2,
        perfect_rates,
        width,
        label="Perfect Rate (%)",
        color="green",
        alpha=0.7,
    )

    ax2 = ax.twinx()
    bars2 = ax2.bar(
        x + width / 2, avg_bers, width, label="Average BER", color="red", alpha=0.7
    )

    ax.set_xlabel("Experiment Group")
    ax.set_ylabel("Perfect Rate (%)", color="green")
    ax2.set_ylabel("Average BER", color="red")
    ax.set_title("Batch Demodulation Summary by Group")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=15)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "summary_bar_chart.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Plot saved: {os.path.join(output_dir, 'summary_bar_chart.png')}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, key in enumerate(sorted(groups.keys())):
        if idx >= 6:
            break
        group_results = groups[key]
        ax = axes[idx]

        bers = [r.ber for r in group_results]

        ax.hist(bers, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(x=0, color="r", linestyle="--", linewidth=2, label="BER=0")
        ax.set_xlabel("BER")
        ax.set_ylabel("Count")
        ax.set_title(f"{key}: BER Distribution\n(n={len(group_results)})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if len(groups) < 6:
        for idx in range(len(groups), 6):
            axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "ber_distribution.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Plot saved: {os.path.join(output_dir, 'ber_distribution.png')}")


def main():
    print("=" * 70)
    print("BATCH DEMODULATION AND BER ANALYSIS")
    print("=" * 70)

    dataset_dir = project_root / "ISO-Texp"
    truth_csv = project_root / "data" / "Mseq_32_original.csv"
    output_dir = project_root / "results" / "batch_demod"

    print(f"\nDataset directory: {dataset_dir}")
    print(f"Ground truth CSV: {truth_csv}")
    print(f"Output directory: {output_dir}\n")

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    if not truth_csv.exists():
        print(f"ERROR: Ground truth CSV not found: {truth_csv}")
        sys.exit(1)

    # Process all images
    print("Processing images...\n")
    results = process_all_images(str(dataset_dir), str(truth_csv))

    if not results:
        print("ERROR: No images processed successfully")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("GENERATING OUTPUTS")
    print("=" * 70 + "\n")

    generate_console_summary(results)

    csv_path = output_dir / "demod_results.csv"
    save_csv(results, str(csv_path))

    report_path = output_dir / "demod_report.txt"
    generate_report(results, str(report_path))

    plot_path = output_dir
    generate_plots(results, str(plot_path))

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
