"""
批量解调 BER - 使用 OOKDemodulator
"""

import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..demodulation import OOKDemodulator


def demodulate_image(
    image_path: str,
    tx_bits: np.ndarray,
) -> Tuple[float, int, int, int]:
    """
    对单张图像解调，返回 (ber, n_errors, n_bits, sync_start)。
    失败时抛出异常。
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    result = OOKDemodulator().demodulate(img)
    if not result.packets or len(result.packets[0]) == 0:
        raise ValueError("No packets detected")
    pkt = result.packets[0].astype(int)
    n = min(len(tx_bits), len(pkt))
    if n == 0:
        raise ValueError("Empty packet")
    n_errors = int(np.sum(tx_bits[:n] != pkt[:n]))
    sync_start = result.sync_positions_row[0] if result.sync_positions_row else -1
    return n_errors / n, n_errors, n, sync_start


def batch_demodulate(
    image_dir: str,
    label_csv: str,
    output_csv: Optional[str] = None,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    verbose: bool = True,
) -> List[dict]:
    """
    批量解调目录下所有图像，返回结果列表并可选保存 CSV。

    Args:
        image_dir:   图像目录
        label_csv:   发射序列 CSV 路径
        output_csv:  结果保存路径（None=不保存）
        extensions:  图像文件扩展名
        verbose:     是否打印进度

    Returns:
        每张图像的结果字典列表（包含 filename, ber, n_errors, n_bits, sync_start, error）
    """
    import pandas as pd

    df = pd.read_csv(label_csv, skiprows=5)
    tx_bits = df.iloc[:, 1].to_numpy()

    image_files = sorted(
        [f for f in Path(image_dir).iterdir() if f.suffix.lower() in extensions]
    )
    if not image_files:
        raise RuntimeError(f"目录 '{image_dir}' 中未找到图像文件")

    results = []
    for i, fpath in enumerate(image_files, 1):
        rec = {
            "filename": fpath.name,
            "ber": None,
            "n_errors": None,
            "n_bits": None,
            "sync_start": None,
            "error": None,
        }
        try:
            ber, n_err, n_bits, sync_start = demodulate_image(str(fpath), tx_bits)
            rec.update(
                {
                    "ber": ber,
                    "n_errors": n_err,
                    "n_bits": n_bits,
                    "sync_start": sync_start,
                }
            )
            if verbose:
                print(
                    f"  [{i}/{len(image_files)}] {fpath.name}: BER={ber:.4f} ({n_err}/{n_bits})"
                )
        except Exception as e:
            rec["error"] = str(e)
            if verbose:
                print(f"  [{i}/{len(image_files)}] {fpath.name}: 解调失败 — {e}")
        results.append(rec)

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        if verbose:
            print(f"\n结果已保存至: {output_csv}")

    return results
