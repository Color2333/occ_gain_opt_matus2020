#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验室增益参数建议工具 (三算法对比)
====================================
输入当前拍摄图像 + 当前相机参数，输出三个算法推荐的下一组设置。

用法示例:
  python others/gain_advisor.py \\
      --image others/new.jpeg \\
      --current-iso 35 \\
      --current-exposure 27.9 \\
      --label-csv results/base_data/Mseq_32_original.csv

不带参数时交互式询问。
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

# 将 src/ 加入路径
_ROOT = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# 单位换算辅助
# ─────────────────────────────────────────────────────────────────────────────

def iso_to_db(iso: float) -> float:
    """ISO → 增益 dB (base ISO = 100)"""
    return 20.0 * np.log10(max(iso / 100.0, 1e-9))


def db_to_iso(db: float) -> float:
    """增益 dB → ISO"""
    return 100.0 * 10.0 ** (db / 20.0)


def iso_to_gain_linear(iso: float) -> float:
    return iso / 100.0


def gain_linear_to_iso(g: float) -> float:
    return g * 100.0


def fmt_exp(s: float) -> str:
    """格式化曝光时间（秒 → 可读字符串）"""
    if s >= 1.0:
        return f"{s:.4f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.3f} ms"
    return f"{s * 1e6:.2f} µs"


# ─────────────────────────────────────────────────────────────────────────────
# ROI 亮度提取
# ─────────────────────────────────────────────────────────────────────────────

def get_roi_brightness(image_path: str, strategy: str = "sync_based"):
    """
    返回 (mean_gray, roi_mask_or_None, method_used_str)。
    优先尝试 sync_based，失败则退回 auto_brightness，再退回全图均值。
    """
    from occ_gain_opt.data_acquisition import DataAcquisition
    from occ_gain_opt.config import ROIStrategy

    img_pil = Image.open(image_path).convert("L")
    img = np.array(img_pil, dtype=np.float64)
    h, w = img.shape
    da = DataAcquisition(width=w, height=h)

    strat_map = {
        "sync_based": ROIStrategy.SYNC_BASED,
        "auto": ROIStrategy.AUTO_BRIGHTNESS,
        "center": ROIStrategy.CENTER,
    }
    primary_strat = strat_map.get(strategy, ROIStrategy.SYNC_BASED)

    for strat, label in [
        (primary_strat, strategy),
        (ROIStrategy.AUTO_BRIGHTNESS, "auto_brightness"),
    ]:
        try:
            mask = da.select_roi(strategy=strat, image=img)
            stats = da.get_roi_statistics(img, mask)
            return float(stats["mean"]), mask, label
        except Exception:
            continue

    # 全图回退
    return float(np.mean(img)), None, "full_image"


# ─────────────────────────────────────────────────────────────────────────────
# BER 解调 (使用与 single_demod_all-exp.py 相同的简单流程)
# ─────────────────────────────────────────────────────────────────────────────

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


def compute_ber(image_path: str, label_csv_path: str):
    """
    对单张图像进行 OOK 解调，返回 (ber, n_errors, n_bits)。
    失败时抛出异常。
    """
    import pandas as pd

    df = pd.read_csv(label_csv_path, skiprows=5)
    tx_bits = df.iloc[:, 1].to_numpy()

    img = np.array(Image.open(image_path).convert("L"), dtype=np.float64)
    column = np.mean(img, axis=1)          # 每行均值
    mean = np.mean(column)
    std = np.std(column)
    if std < 1e-6:
        raise ValueError("图像行均值方差为零，无法解调")
    y = (column - mean) / std

    threshold = _polyfit_threshold(y, degree=3)
    yy = y - threshold
    rr = (yy > 0).astype(int)

    equ, payload_start = _find_sync(rr)
    rx = _recover_data(rr, payload_start, equ)
    n = min(len(tx_bits), len(rx))
    if n == 0:
        raise ValueError("恢复数据长度为零")
    rx = rx[1 : n + 1]          # 与 single_demod 保持一致：跳过第一位
    n = min(len(tx_bits), len(rx))
    n_errors = int(np.sum(tx_bits[:n] != rx[:n]))
    ber = n_errors / n
    return ber, n_errors, n


# ─────────────────────────────────────────────────────────────────────────────
# 算法 1 — Matus 单次公式 (paper eq. 7)
# ─────────────────────────────────────────────────────────────────────────────

def algo_matus_single(current_iso: float, mean_gray: float,
                      target_gray: float = 242.25,
                      gain_db_min: float = 0.0,
                      gain_db_max: float = 20.0):
    """
    G_opt(dB) = G_curr(dB) + 20·log10(Y_target / Y_curr)
    返回 (iso_next, gain_db_next)
    """
    g_db = iso_to_db(current_iso)
    if mean_gray <= 0:
        return current_iso, g_db
    g_opt_db = g_db + 20.0 * np.log10(target_gray / mean_gray)
    g_opt_db = float(np.clip(g_opt_db, gain_db_min, gain_db_max))
    iso_next = db_to_iso(g_opt_db)
    return iso_next, g_opt_db


# ─────────────────────────────────────────────────────────────────────────────
# 算法 2 — Matus 自适应迭代 (带学习率 α)
# ─────────────────────────────────────────────────────────────────────────────

def algo_matus_adaptive(current_iso: float, mean_gray: float,
                        alpha: float = 0.5,
                        target_gray: float = 242.25,
                        step_max_db: float = 5.0,
                        gain_db_min: float = 0.0,
                        gain_db_max: float = 20.0):
    """
    G_{k+1} = G_k + α × 20·log10(Y_target / Y_k)
    返回 (iso_next, gain_db_next)
    """
    g_db = iso_to_db(current_iso)
    if mean_gray <= 0:
        return current_iso, g_db
    delta_db = 20.0 * np.log10(target_gray / mean_gray)
    delta_db = float(np.clip(delta_db, -step_max_db, step_max_db))
    g_next_db = g_db + alpha * delta_db
    g_next_db = float(np.clip(g_next_db, gain_db_min, gain_db_max))
    iso_next = db_to_iso(g_next_db)
    return iso_next, g_next_db


# ─────────────────────────────────────────────────────────────────────────────
# 算法 3 — Ma 自适应阻尼 (单步推荐)
# ─────────────────────────────────────────────────────────────────────────────

def algo_ma_damping(current_iso: float,
                    current_exp_s: float,
                    mean_brightness: float,
                    target_brightness: float = 125.0,
                    strategy: str = "exposure_priority",
                    gain_min: float = 0.30,
                    gain_max: float = 100.0,
                    exp_min: float = 1e-6,
                    exp_max: float = 1e-3):
    """
    Ma算法首步推荐 (State II 比例法，阻尼 d=0 因为仅有一帧历史)。

    formula: r = target / current_brightness
    exposure_priority → e_new = r × e_curr,  g_new = g_curr
    gain_priority     → g_new = r × g_curr,  e_new = e_curr

    返回 (iso_next, exp_next_s, r)
    """
    g_curr = iso_to_gain_linear(current_iso)
    brightness = max(mean_brightness, 0.1)
    r = float(np.clip(target_brightness / brightness, 0.1, 10.0))

    if strategy == "exposure_priority":
        e_new = float(np.clip(r * current_exp_s, exp_min, exp_max))
        g_new = g_curr
    else:  # gain_priority
        g_new = float(np.clip(r * g_curr, gain_min, gain_max))
        e_new = current_exp_s

    iso_next = gain_linear_to_iso(g_new)
    return iso_next, e_new, r


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="增益参数建议工具：输入图像 → 三算法推荐下一步相机参数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image", default="others/new.jpeg",
        help="输入图像路径 (默认: others/new.jpeg)",
    )
    parser.add_argument(
        "--current-iso", type=float, default=None,
        help="当前相机 ISO (如 35、100、400)",
    )
    parser.add_argument(
        "--current-exposure", type=float, default=None,
        help="当前曝光时间 (µs，如 27.9)",
    )
    parser.add_argument(
        "--label-csv",
        default="results/base_data/Mseq_32_original.csv",
        help="发射序列 CSV 路径",
    )
    parser.add_argument(
        "--roi", default="sync_based",
        choices=["sync_based", "auto", "center"],
        help="ROI 策略 (默认: sync_based)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Matus 自适应学习率 α (0.3–0.7，默认 0.5)",
    )
    parser.add_argument(
        "--target-gray", type=float, default=242.25,
        help="Matus 目标灰度 (默认 255×0.95=242.25)",
    )
    parser.add_argument(
        "--target-brightness", type=float, default=125.0,
        help="Ma 算法目标亮度 (0–255，默认 125)",
    )
    parser.add_argument(
        "--ma-strategy", default="exposure_priority",
        choices=["exposure_priority", "gain_priority"],
        help="Ma 算法控制策略 (默认: exposure_priority)",
    )
    parser.add_argument(
        "--iso-min", type=float, default=30.0,
        help="相机最小 ISO (默认 30)",
    )
    parser.add_argument(
        "--iso-max", type=float, default=10000.0,
        help="相机最大 ISO (默认 10000)",
    )
    args = parser.parse_args()

    # ── 交互式询问缺失参数 ──
    if args.current_iso is None:
        try:
            args.current_iso = float(input("请输入当前相机 ISO (如 35): ").strip())
        except (ValueError, EOFError):
            args.current_iso = 35.0
            print(f"  使用默认 ISO {args.current_iso}")

    if args.current_exposure is None:
        try:
            args.current_exposure = float(
                input("请输入当前曝光时间 (µs，如 27.9): ").strip()
            )
        except (ValueError, EOFError):
            args.current_exposure = 27.9
            print(f"  使用默认曝光时间 {args.current_exposure} µs")

    current_exp_s = args.current_exposure * 1e-6   # µs → s

    # ── 图像分析 ──
    print(f"\n{'=' * 62}")
    print("  实验室增益参数建议工具  (三算法对比)")
    print(f"{'=' * 62}")
    print(f"  图像     : {args.image}")
    print(f"  当前 ISO : {args.current_iso:.0f}  "
          f"(增益 {iso_to_db(args.current_iso):+.2f} dB, "
          f"线性 {iso_to_gain_linear(args.current_iso):.3f}×)")
    print(f"  当前曝光 : {fmt_exp(current_exp_s)}")

    # ROI 亮度
    print(f"\n  [1/3] 分析 ROI 亮度 (策略: {args.roi}) ...")
    try:
        mean_gray, roi_mask, roi_method = get_roi_brightness(args.image, args.roi)
        print(f"        ROI方法 = {roi_method} | 平均灰度 = {mean_gray:.2f} / 255")
    except Exception as e:
        print(f"        ROI分析失败: {e}")
        img_arr = np.array(Image.open(args.image).convert("L"), dtype=np.float64)
        mean_gray = float(np.mean(img_arr))
        roi_method = "full_image (fallback)"
        print(f"        回退到全图均值: {mean_gray:.2f}")

    # BER 解调
    print(f"\n  [2/3] 解调 BER ({args.label_csv}) ...")
    ber_str = "N/A (解调失败)"
    try:
        ber, n_errors, n_bits = compute_ber(args.image, args.label_csv)
        ber_str = f"{ber:.4f}  ({n_errors}/{n_bits} errors)"
        print(f"        BER = {ber_str}")
    except Exception as e:
        print(f"        解调失败: {e}")

    # ── 三算法推荐 ──
    print(f"\n  [3/3] 计算三算法推荐参数 ...")

    gain_db_min = iso_to_db(args.iso_min)
    gain_db_max = iso_to_db(args.iso_max)

    iso1, db1 = algo_matus_single(
        args.current_iso, mean_gray, args.target_gray,
        gain_db_min=gain_db_min, gain_db_max=gain_db_max,
    )
    iso2, db2 = algo_matus_adaptive(
        args.current_iso, mean_gray, args.alpha, args.target_gray,
        gain_db_min=gain_db_min, gain_db_max=gain_db_max,
    )
    iso3, exp3, ratio3 = algo_ma_damping(
        args.current_iso, current_exp_s, mean_gray,
        args.target_brightness, args.ma_strategy,
        gain_min=iso_to_gain_linear(args.iso_min),
        gain_max=iso_to_gain_linear(args.iso_max),
    )

    # ── 收敛判断 ──
    converged_matus = abs(mean_gray - args.target_gray) < 5.0
    converged_ma = abs(mean_gray - args.target_brightness) < 5.0

    # ── 输出表格 ──
    SEP = "─" * 62
    print(f"\n{SEP}")
    print("  推荐参数 (下一次拍摄请设置)")
    print(SEP)
    print(f"  {'算法':<30} {'ISO':>7} {'增益(dB)':>10} {'曝光':>14}  收敛?")
    print(SEP)

    exp_unchanged = fmt_exp(current_exp_s) + " (不变)"
    print(f"  {'1. Matus 单次公式':<30} {iso1:>7.1f} {db1:>+10.2f} {exp_unchanged:>14}  {'是' if converged_matus else '否'}")
    print(f"  {'2. Matus 自适应 (α=' + str(args.alpha) + ')':<30} {iso2:>7.1f} {db2:>+10.2f} {exp_unchanged:>14}  {'是' if converged_matus else '否'}")
    ma_label = f"3. Ma 阻尼 ({args.ma_strategy[:3]})"
    print(f"  {ma_label:<30} {iso3:>7.1f} {'(线性)':>10} {fmt_exp(exp3):>14}  {'是' if converged_ma else '否'}")

    print(SEP)

    # ── 说明 ──
    print(f"""
  说明：
  ┌ Matus 算法 (算法1 & 2)
  │  目标灰度   : {args.target_gray:.1f}  (255 × 0.95)
  │  当前灰度   : {mean_gray:.1f}
  │  偏差       : {mean_gray - args.target_gray:+.1f}  ({'偏暗' if mean_gray < args.target_gray else '偏亮'})
  │  算法1 单次公式: G_opt = G_curr + 20·log10({args.target_gray:.0f}/{mean_gray:.1f})
  │  算法2 自适应 : G_next = G_curr + {args.alpha}×20·log10({args.target_gray:.0f}/{mean_gray:.1f})
  └  仅调整增益(ISO)，不改变曝光时间。

  ┌ Ma 算法 (算法3)
  │  目标亮度   : {args.target_brightness:.1f}  (线性 0–255 中间值)
  │  当前亮度   : {mean_gray:.1f}
  │  调节比例 r : {ratio3:.3f}  (target/current)
  │  策略       : {args.ma_strategy}
  └  调整{'曝光时间' if args.ma_strategy == 'exposure_priority' else 'ISO/增益'}，{'ISO 不变' if args.ma_strategy == 'exposure_priority' else '曝光不变'}。

  BER       : {ber_str}
  图像均值  : {mean_gray:.1f} / 255  (ROI: {roi_method})
""")

    # ── 简洁操作指令 ──
    d_iso1 = iso1 - args.current_iso
    d_iso2 = iso2 - args.current_iso
    d_exp3 = exp3 - current_exp_s
    d_iso3 = iso3 - args.current_iso

    print("  快速操作建议 (选择一个算法):")
    print(f"    算法1 [Matus单次]: ISO {args.current_iso:.0f} → {iso1:.0f}"
          f"  ({d_iso1:+.0f})，曝光不变")
    print(f"    算法2 [Matus自适应α={args.alpha}]: ISO {args.current_iso:.0f} → {iso2:.0f}"
          f"  ({d_iso2:+.0f})，曝光不变")
    if args.ma_strategy == "exposure_priority":
        print(f"    算法3 [Ma阻尼]: 曝光 {fmt_exp(current_exp_s)} → {fmt_exp(exp3)}"
              f"  ({d_exp3 * 1e6:+.2f} µs)，ISO不变")
    else:
        print(f"    算法3 [Ma阻尼]: ISO {args.current_iso:.0f} → {iso3:.0f}"
              f"  ({d_iso3:+.0f})，曝光不变")
    print()


if __name__ == "__main__":
    main()
