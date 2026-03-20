"""
批量解调 BER
从 real/single_demod_all-exp.py 移植为包内模块。
对给定目录下所有图像批量解调，保存 CSV 结果。
"""

import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ── 解调核心函数 ───────────────────────────────────────────────────────────────

def _find_sync(rr, head_len=8, max_head_len=100, max_len_diff=5, last_header_len=None):
    runs = []
    p = 0
    while p < len(rr):
        if rr[p] == 1:
            q = p
            while q < len(rr) and rr[q] == 1:
                q += 1
            length = q - p
            if head_len <= length <= max_head_len:
                runs.append((p, q - 1, length))
            p = q
        else:
            p += 1
    if not runs:
        raise ValueError("未检测到任何有效同步段")
    runs_sorted = sorted(runs, key=lambda x: x[2], reverse=True)
    if last_header_len is not None:
        filtered = [r for r in runs_sorted if abs(r[2] - last_header_len) <= max_len_diff]
        if filtered:
            runs_sorted = filtered
    h1_s, h1_e, len1 = runs_sorted[0]
    h2_s = h2_e = len2 = None
    for run in runs_sorted[1:]:
        if abs(run[2] - len1) <= max_len_diff:
            h2_s, h2_e, len2 = run
            break
    if h2_s is not None:
        payload_start = min(h1_e, h2_e) + 1
        equ = int(round(abs((len1 + len2) / (head_len * 2))))
        sync_start = min(h1_s, h2_s)
    else:
        payload_start = h1_e + 1
        equ = int(round(len1 / head_len))
        sync_start = h1_s
    return equ, payload_start, sync_start


def _recover_data(rr, payload_start, equ_len):
    p = payload_start
    res = []
    for i in range(payload_start, len(rr) - 1):
        if rr[i + 1] != rr[i]:
            width = (i + 1) - p
            cnt = int(round(width / equ_len))
            res.extend([rr[i]] * cnt)
            p = i + 1
    return np.array(res)


def _polyfit_threshold(y, degree=3):
    x = np.arange(1, len(y) + 1)
    return np.polyval(np.polyfit(x, y, degree), x)


def demodulate_image(
    image_path: str,
    tx_bits: np.ndarray,
) -> Tuple[float, int, int, int]:
    """
    对单张图像解调，返回 (ber, n_errors, n_bits, sync_start)。
    失败时抛出异常。
    """
    from PIL import Image as PILImage
    img = np.array(PILImage.open(image_path).convert("L"), dtype=np.float64)
    column = np.mean(img, axis=1)
    mean, std = np.mean(column), np.std(column)
    if std < 1e-6:
        raise ValueError("方差为零")
    y = (column - mean) / std
    threshold = _polyfit_threshold(y, degree=3)
    rr = (y - threshold > 0).astype(int)
    equ, payload_start, sync_start = _find_sync(rr)
    rx = _recover_data(rr, payload_start, equ)
    rx = rx[1:]  # 与原脚本保持一致：跳过第一位
    n = min(len(tx_bits), len(rx))
    if n == 0:
        raise ValueError("恢复数据为空")
    n_errors = int(np.sum(tx_bits[:n] != rx[:n]))
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

    image_files = sorted([
        f for f in Path(image_dir).iterdir()
        if f.suffix.lower() in extensions
    ])
    if not image_files:
        raise RuntimeError(f"目录 '{image_dir}' 中未找到图像文件")

    results = []
    for i, fpath in enumerate(image_files, 1):
        rec = {"filename": fpath.name, "ber": None, "n_errors": None,
               "n_bits": None, "sync_start": None, "error": None}
        try:
            ber, n_err, n_bits, sync_start = demodulate_image(str(fpath), tx_bits)
            rec.update({"ber": ber, "n_errors": n_err, "n_bits": n_bits, "sync_start": sync_start})
            if verbose:
                print(f"  [{i}/{len(image_files)}] {fpath.name}: BER={ber:.4f} ({n_err}/{n_bits})")
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
