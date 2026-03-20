"""
单帧多算法顾问
输入一张图像和当前相机参数，输出三个算法推荐的下一组参数。
从 real/gain_advisor.py 移植为包内模块。

用法:
    from occ_gain_opt.experiments.advisor import run_advisor
    from occ_gain_opt.config import CameraParams

    result = run_advisor(
        image_path="real/new.jpeg",
        current_params=CameraParams(iso=35, exposure_us=27.9),
        label_csv="results/base_data/Mseq_32_original.csv",
    )
"""

import os
from typing import Optional, Tuple

import numpy as np

from ..config import CameraParams, ROIStrategy
from ..algorithms import get as algo_get
from ..data_sources.roi import (
    create_center_roi_mask,
    create_auto_roi_mask,
    create_sync_based_roi_mask,
    compute_roi_stats,
)


# ── ROI 亮度提取 ───────────────────────────────────────────────────────────────

def get_roi_brightness(
    image: np.ndarray,
    strategy: str = "sync_based",
) -> Tuple[float, str]:
    """
    返回 (mean_gray, method_used_str)。
    优先尝试 strategy，失败则退回 auto，再退回全图均值。
    """
    h, w = image.shape[:2]
    placeholder = np.zeros((h, w), dtype=np.uint8)

    for strat, label in [
        (strategy, strategy),
        ("auto", "auto_brightness"),
    ]:
        try:
            if strat == "sync_based":
                mask = create_sync_based_roi_mask(image)
            elif strat == "auto":
                mask = create_auto_roi_mask(image)
            else:
                mask = create_center_roi_mask(placeholder)
            stats = compute_roi_stats(image, mask)
            return float(stats["mean"]), label
        except Exception:
            continue

    return float(np.mean(image)), "full_image"


# ── BER 解调 ──────────────────────────────────────────────────────────────────

def _find_sync(rr, head_len=8, max_head_len=100, max_len_diff=5):
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
        raise ValueError("未检测到同步头")
    runs_sorted = sorted(runs, key=lambda x: x[2], reverse=True)
    h1_s, h1_e, len1 = runs_sorted[0]
    h2_s = h2_e = len2 = None
    for run in runs_sorted[1:]:
        if abs(run[2] - len1) <= max_len_diff:
            h2_s, h2_e, len2 = run
            break
    if h2_s is not None:
        payload_start = min(h1_e, h2_e) + 1
        equ = int(round(abs((len1 + len2) / (head_len * 2))))
    else:
        payload_start = h1_e + 1
        equ = int(round(len1 / head_len))
    return equ, payload_start


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
    coeffs = np.polyfit(x, y, degree)
    return np.polyval(coeffs, x)


def compute_ber(image: np.ndarray, label_csv_path: str) -> Tuple[float, int, int]:
    """
    对单帧图像进行 OOK 解调，返回 (ber, n_errors, n_bits)。
    失败时抛出异常。
    """
    import pandas as pd

    df = pd.read_csv(label_csv_path, skiprows=5)
    tx_bits = df.iloc[:, 1].to_numpy()

    img = image.astype(np.float64) if image.ndim == 2 else \
          __import__('cv2').cvtColor(image, __import__('cv2').COLOR_BGR2GRAY).astype(np.float64)

    column = np.mean(img, axis=1)
    mean = np.mean(column)
    std = np.std(column)
    if std < 1e-6:
        raise ValueError("图像行均值方差为零，无法解调")
    y = (column - mean) / std
    threshold = _polyfit_threshold(y, degree=3)
    rr = (y - threshold > 0).astype(int)

    equ, payload_start = _find_sync(rr)
    rx = _recover_data(rr, payload_start, equ)
    n = min(len(tx_bits), len(rx))
    if n == 0:
        raise ValueError("恢复数据长度为零")
    rx = rx[1:n + 1]
    n = min(len(tx_bits), len(rx))
    n_errors = int(np.sum(tx_bits[:n] != rx[:n]))
    return n_errors / n, n_errors, n


# ── 主函数 ─────────────────────────────────────────────────────────────────────

def run_advisor(
    image: np.ndarray,
    current_params: CameraParams,
    label_csv: Optional[str] = None,
    roi_strategy: str = "sync_based",
    alpha: float = 0.5,
    target_gray: float = 242.25,
    target_brightness: float = 125.0,
    ma_strategy: str = "exposure_priority",
    iso_min: float = 30.0,
    iso_max: float = 10000.0,
    verbose: bool = True,
) -> dict:
    """
    单帧三算法顾问：计算三种算法推荐的下一步相机参数。

    Args:
        image:             当前帧（BGR 或灰度 uint8）
        current_params:    当前相机参数
        label_csv:         发射序列 CSV 路径（用于 BER 解调，None=跳过）
        roi_strategy:      ROI 策略（"sync_based" / "auto" / "center"）
        alpha:             Matus 自适应学习率
        target_gray:       Matus 目标灰度
        target_brightness: Ma 目标亮度
        ma_strategy:       Ma 策略（"exposure_priority" / "gain_priority"）
        iso_min / iso_max: ISO 范围限制
        verbose:           是否打印结果表格

    Returns:
        字典，包含：
          mean_gray, roi_method, ber (or None),
          single_shot, adaptive_iter, adaptive_damping  (各为 CameraParams)
    """
    # ── ROI 亮度 ──
    mean_gray, roi_method = get_roi_brightness(image, roi_strategy)

    # ── BER ──
    ber = None
    if label_csv and os.path.isfile(label_csv):
        try:
            ber, _, _ = compute_ber(image, label_csv)
        except Exception:
            pass

    gain_db_min = 20.0 * np.log10(max(iso_min / 100.0, 1e-9))
    gain_db_max = 20.0 * np.log10(max(iso_max / 100.0, 1e-9))

    # ── 算法1：单次公式 ──
    algo1 = algo_get("single_shot")(
        target_gray=target_gray,
        gain_db_min=gain_db_min,
        gain_db_max=gain_db_max,
    )
    next1 = algo1.compute_next_params(current_params, mean_gray)

    # ── 算法2：自适应迭代 ──
    algo2 = algo_get("adaptive_iter")(
        alpha=alpha,
        target_gray=target_gray,
        gain_db_min=gain_db_min,
        gain_db_max=gain_db_max,
    )
    next2 = algo2.compute_next_params(current_params, mean_gray)

    # ── 算法3：自适应阻尼 ──
    algo3 = algo_get("adaptive_damping")(
        target_brightness=target_brightness,
        iso_min=iso_min,
        iso_max=iso_max,
        strategy=ma_strategy,
    )
    next3 = algo3.compute_next_params(current_params, mean_gray)

    if verbose:
        _print_table(current_params, mean_gray, roi_method, ber,
                     next1, next2, next3, alpha, target_gray, target_brightness, ma_strategy)

    return {
        "mean_gray": mean_gray,
        "roi_method": roi_method,
        "ber": ber,
        "single_shot": next1,
        "adaptive_iter": next2,
        "adaptive_damping": next3,
    }


def _print_table(current, mean_gray, roi_method, ber,
                 next1, next2, next3, alpha, target_gray, target_brightness, ma_strategy):
    SEP = "─" * 62
    print(f"\n{SEP}")
    print("  推荐参数 (下一次拍摄请设置)")
    print(SEP)
    print(f"  当前 ISO   : {current.iso:.0f}  ({current.gain_db:+.2f} dB)")
    print(f"  当前曝光   : {current.exposure_us:.2f} µs")
    print(f"  ROI 灰度均值: {mean_gray:.2f} / 255  (方法: {roi_method})")
    if ber is not None:
        print(f"  BER        : {ber:.4f}")
    print(SEP)
    fmt_exp = lambda us: f"{us:.2f}µs"
    converged_matus = abs(mean_gray - target_gray) < 5.0
    converged_ma = abs(mean_gray - target_brightness) < 5.0
    print(f"  {'算法':<28} {'ISO':>7} {'增益(dB)':>9} {'曝光':>12}  收敛?")
    print(SEP)
    exp_str = fmt_exp(current.exposure_us) + " (不变)"
    print(f"  {'1. 单次公式':<28} {next1.iso:>7.1f} {next1.gain_db:>+9.2f} {exp_str:>12}  {'是' if converged_matus else '否'}")
    print(f"  {'2. 自适应迭代 α=' + str(alpha):<28} {next2.iso:>7.1f} {next2.gain_db:>+9.2f} {exp_str:>12}  {'是' if converged_matus else '否'}")
    damp_label = f"3. 自适应阻尼 ({ma_strategy[:3]})"
    print(f"  {damp_label:<28} {next3.iso:>7.1f} {'(线性)':>9} {fmt_exp(next3.exposure_us):>12}  {'是' if converged_ma else '否'}")
    print(SEP)
