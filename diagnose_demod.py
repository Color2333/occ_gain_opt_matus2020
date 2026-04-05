"""
OCC 解调诊断脚本
================
功能:
  1. 遍历 GainLimit 0-100 (10 个等间距步长)，每步拍 1 张照片 (共 10 张)
  2. 对每张图用 3 种策略尝试解调:
       - legacy_sync  : 全图行均值 + polyfit阈值 + 多同步头检测
       - ook_demod    : OOKDemodulator (现有模块)
       - direct_slice : 手动按常见行数估算 bit_period 的暴力切分
  3. 记录每种策略的关键中间量 + BER
  4. 保存:
       - results/diagnose/<gain_limit>/frame.png
       - results/diagnose/<gain_limit>/row_mean.png  (行均值图)
       - results/diagnose/<gain_limit>/demod_debug.png  (3 格调试图)
  5. 打印统计汇总表

运行:
    cd "Code Repo/uocc-adaptive"
    python diagnose_demod.py
"""

from __future__ import annotations

import sys
import time
import json
import warnings
from csv import reader
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # 无显示器环境
import matplotlib.pyplot as plt
import numpy as np

# ── 路径 ──────────────────────────────────────────────────────────────────────
REPO_DIR = Path(__file__).parent
sys.path.insert(0, str(REPO_DIR / "src"))

from camera_isapi import HikvisionCamera  # noqa: E402
from occ_gain_opt.demodulation import OOKDemodulator  # noqa: E402

# ── 配置 ──────────────────────────────────────────────────────────────────────
RTSP_URL        = "rtsp://admin:abcd1234@192.168.1.19:554/Streaming/Channels/101"
CAMERA_IP       = "192.168.1.19"
CAMERA_USER     = "admin"
CAMERA_PASSWORD = "abcd1234"

TX_LABEL_CSV    = str(REPO_DIR / "data" / "Mseq_32_with_header.csv")
OUT_DIR         = REPO_DIR / "results" / "diagnose"

GAIN_LIMITS     = list(range(0, 101, 10))   # 0,10,20,...,100 共 11 步
SETTLE_S        = 1.5                        # 设完 GainLimit 后稳定等待秒数
RTSP_WARM_S     = 0.5                        # RTSP 打开后预热秒数

# ── CSV 工具 ──────────────────────────────────────────────────────────────────
def _load_truth_bits(label_csv: str) -> np.ndarray:
    p = Path(label_csv)
    if "_with_header" in p.stem:
        orig = p.with_name(p.name.replace("_with_header", "_original"))
        if orig.is_file():
            p = orig
    bits: List[int] = []
    with p.open(encoding="utf-8") as f:
        for idx, row in enumerate(reader(f)):
            if idx < 6 or len(row) < 2:
                continue
            try:
                bits.append(int(row[1]))
            except ValueError:
                pass
    return np.array(bits, dtype=np.uint8)


def _infer_sync_head_len(label_csv: str) -> int:
    p = Path(label_csv)
    if "_with_header" not in p.stem or not p.is_file():
        return 8
    bits: List[int] = []
    with p.open(encoding="utf-8") as f:
        for idx, row in enumerate(reader(f)):
            if idx < 6 or len(row) < 2:
                continue
            try:
                bits.append(int(row[1]))
            except ValueError:
                pass
    unit = bits[: len(bits) // 3] if len(bits) % 3 == 0 else bits
    run = 0
    started = False
    for b in unit:
        if not started:
            if b == 0:
                started = True
            continue
        if b == 1:
            run += 1
        else:
            break
    return run if run > 0 else 8


# ── 图像抓取 ──────────────────────────────────────────────────────────────────
def grab_frame(rtsp_url: str) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        cap.release()
        return None
    time.sleep(RTSP_WARM_S)
    frame = None
    for _ in range(10):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f.copy()
            break
    cap.release()
    return frame


# ── 解调工具函数 ──────────────────────────────────────────────────────────────
def _polyfit_threshold(y: np.ndarray, degree: int = 3) -> np.ndarray:
    x = np.arange(1, len(y) + 1, dtype=float)
    return np.polyval(np.polyfit(x, y, degree), x)


def _legacy_find_sync_info(rr: np.ndarray, head_len: int) -> Optional[Dict]:
    """检测同步头，返回 dict 或 None"""
    runs = []
    p = 0
    while p < len(rr):
        if rr[p] == 1:
            q = p
            while q < len(rr) and rr[q] == 1:
                q += 1
            length = q - p
            if head_len <= length <= 200:
                runs.append((p, q - 1, length))
            p = q
        else:
            p += 1

    if not runs:
        return None

    runs_sorted = sorted(runs, key=lambda x: x[2], reverse=True)
    h1_start, h1_end, len1 = runs_sorted[0]

    matched = [r for r in runs if abs(r[2] - len1) <= max(5, int(len1 * 0.2))]
    matched = sorted(matched, key=lambda x: x[0])

    h2 = None
    for run in runs_sorted[1:]:
        if abs(run[2] - len1) <= max(5, int(len1 * 0.2)):
            h2 = run
            break

    # 综合两种 equ 估计（权重 0.7 给间距法，0.3 给同步头长度法）
    BITS_PER_PACKET = 40
    SPACING_WEIGHT  = 0.7

    equ_sync = max(1.0, abs((len1 + h2[2]) / (head_len * 2))) if h2 is not None \
               else max(1.0, len1 / head_len)

    mr_complete = [r for r in matched if r[0] > 5]
    equ_spacing = None
    if len(mr_complete) >= 2:
        spacing = mr_complete[1][0] - mr_complete[0][0]
        if BITS_PER_PACKET * 5 < spacing < BITS_PER_PACKET * 30:
            equ_spacing = spacing / BITS_PER_PACKET

    if equ_spacing is not None:
        equ = SPACING_WEIGHT * equ_spacing + (1 - SPACING_WEIGHT) * equ_sync
    else:
        equ = equ_sync

    # 选第一个数据能完整装入帧的同步头
    frame_len = len(rr)
    needed_rows = int((1 + head_len + 32) * equ) + 5
    candidate_runs = mr_complete if mr_complete else matched
    best_sync = candidate_runs[0]
    for mr_r in candidate_runs:
        ps_candidate = mr_r[1] + 1
        if ps_candidate + needed_rows <= frame_len:
            best_sync = mr_r
            break

    payload_start = best_sync[1] + 1

    return {
        "equ": equ,
        "payload_start": payload_start,
        "matched_runs": matched,
        "header_len_rows": len1,
    }


def _legacy_recover_data(rr: np.ndarray, payload_start: int, equ: float) -> np.ndarray:
    """
    均匀采样解码：在 payload_start + (k+0.5)*equ 处采样第 k 个 bit。
    避免游程RLE的相位积累漂移问题。
    """
    res: List[int] = []
    k = 0
    while True:
        pos = int(payload_start + (k + 0.5) * equ)
        if pos >= len(rr):
            break
        res.append(int(rr[pos]))
        k += 1
    return np.array(res, dtype=np.uint8)


def _calc_ber(
    decoded: np.ndarray, truth: np.ndarray
) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    if len(decoded) == 0 or len(truth) == 0:
        return None, None, None
    n = int(min(len(decoded), len(truth)))
    errs = int(np.sum(decoded[:n] != truth[:n]))
    return errs / n, errs, n


# ── 策略 1: legacy_sync ───────────────────────────────────────────────────────
def demod_legacy(
    frame: np.ndarray,
    truth: np.ndarray,
    sync_head_len: int,
    data_bits: int,
) -> Dict:
    result: Dict = {"strategy": "legacy_sync", "ok": False, "error": None}
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        column = np.mean(gray.astype(np.float64), axis=1)
        std = float(np.std(column))
        result["row_std"] = float(std)
        result["row_mean_mean"] = float(np.mean(column))

        if std < 1e-6:
            result["error"] = "行均值方差为零"
            return result

        y = (column - float(np.mean(column))) / std
        threshold = _polyfit_threshold(y, degree=3)
        rr = (y - threshold > 0).astype(np.uint8)

        # 同步头统计（不同宽容度）
        all_runs = []
        p = 0
        while p < len(rr):
            if rr[p] == 1:
                q = p
                while q < len(rr) and rr[q] == 1:
                    q += 1
                all_runs.append((p, q - 1, q - p))
                p = q
            else:
                p += 1

        result["total_runs_detected"] = len(all_runs)
        result["run_lengths"] = sorted([r[2] for r in all_runs], reverse=True)[:20]

        info = _legacy_find_sync_info(rr, sync_head_len)
        if info is None:
            result["error"] = "未检测到有效同步段"
            return result

        # 用行均值自相关估算 bit_period（检测到正峰时最准确）
        BITS_PER_PACKET = 40
        equ_final = float(info["equ"])  # 默认兜底
        try:
            rr_f = column - float(np.mean(column))
            ac_full = np.correlate(rr_f, rr_f, mode='full')
            ac = ac_full[len(ac_full) // 2:]
            ac_norm = ac / (ac[0] + 1e-9)
            s_min, s_max = 430, 570
            if s_max < len(ac_norm):
                region = ac_norm[s_min:s_max]
                pos_peaks = []
                for i in range(1, len(region) - 1):
                    if region[i] > region[i-1] and region[i] > region[i+1] and region[i] > 0.02:
                        pos_peaks.append((i + s_min, float(region[i])))
                if pos_peaks:
                    best_peak = max(pos_peaks, key=lambda p: p[1])
                    equ_final = float(best_peak[0]) / BITS_PER_PACKET
        except Exception:
            pass

        result["sync_count"] = len(info["matched_runs"])
        result["bit_period"] = equ_final
        result["payload_start"] = int(info["payload_start"])
        result["header_len_rows"] = int(info["header_len_rows"])

        raw = _legacy_recover_data(rr, int(info["payload_start"]), equ_final)
        decoded = raw[1: data_bits + 1]  # 跳过起始位
        result["decoded_len"] = len(decoded)

        ber, errs, n = _calc_ber(decoded, truth)
        result["ber"] = ber
        result["errors"] = errs
        result["n_bits"] = n
        result["ok"] = ber is not None
        result["decoded_bits_head"] = decoded[:16].tolist() if len(decoded) >= 16 else decoded.tolist()
        result["truth_bits_head"] = truth[:16].tolist() if len(truth) >= 16 else truth.tolist()
        # 为画图保存中间量
        result["_column"] = column
        result["_y"] = y
        result["_threshold"] = threshold
        result["_rr"] = rr
        result["_matched_runs"] = info["matched_runs"]
    except Exception as e:
        result["error"] = str(e)
    return result


# ── 策略 2: OOKDemodulator ────────────────────────────────────────────────────
def demod_ook(
    frame: np.ndarray,
    truth: np.ndarray,
    data_bits: int,
) -> Dict:
    result: Dict = {"strategy": "ook_demod", "ok": False, "error": None}
    try:
        d = OOKDemodulator(config={"data_bits": data_bits}).demodulate(frame)
        result["sync_count"] = len(d.sync_positions_row)
        result["bit_period"] = float(d.bit_period)
        result["confidence"] = float(d.confidence)
        result["roi_pixels"] = int(np.sum(d.roi_mask))

        if d.packets:
            decoded = d.packets[0].astype(np.uint8)
            result["decoded_len"] = len(decoded)
            ber, errs, n = _calc_ber(decoded, truth)
            result["ber"] = ber
            result["errors"] = errs
            result["n_bits"] = n
            result["ok"] = ber is not None
            result["decoded_bits_head"] = decoded[:16].tolist() if len(decoded) >= 16 else decoded.tolist()
        else:
            result["error"] = "未提取出 packet"
        result["truth_bits_head"] = truth[:16].tolist() if len(truth) >= 16 else truth.tolist()
    except Exception as e:
        result["error"] = str(e)
    return result


# ── 策略 3: direct_slice (暴力按估算 bit_period 切片) ─────────────────────────
def demod_direct_slice(
    frame: np.ndarray,
    truth: np.ndarray,
    data_bits: int,
    sync_head_len: int,
    candidate_periods: Optional[List[float]] = None,
) -> Dict:
    """
    不依赖同步头检测:
    - 尝试一系列候选 bit_period
    - 对每个 period 在全图行均值的整数位置采样
    - 选 BER 最低的
    """
    result: Dict = {"strategy": "direct_slice", "ok": False, "error": None}
    if candidate_periods is None:
        candidate_periods = [float(p) for p in range(2, 30)]

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        column = np.mean(gray.astype(np.float64), axis=1)
        std = float(np.std(column))
        if std < 1e-6:
            result["error"] = "行均值方差为零"
            return result

        y = (column - float(np.mean(column))) / std
        threshold = _polyfit_threshold(y, degree=3)
        binary = (y - threshold > 0).astype(np.uint8)

        best_ber = 1.1
        best_period = None
        best_offset = None
        best_decoded = None

        total_bits_needed = sync_head_len + data_bits + 10
        for period in candidate_periods:
            n_samples = int(len(binary) / period)
            if n_samples < total_bits_needed:
                continue
            for offset in np.linspace(0, period, 8, endpoint=False):
                positions = np.arange(n_samples) * period + offset
                positions = positions[positions < len(binary)]
                sampled = binary[positions.astype(int)]
                # 跳过 sync_head_len+1 个位，取后 data_bits 个
                skip = sync_head_len + 1
                if len(sampled) < skip + data_bits:
                    continue
                decoded = sampled[skip: skip + data_bits]
                ber, errs, n = _calc_ber(decoded, truth)
                if ber is not None and ber < best_ber:
                    best_ber = ber
                    best_period = period
                    best_offset = offset
                    best_decoded = decoded

        if best_decoded is not None:
            result["bit_period"] = best_period
            result["best_offset"] = best_offset
            result["decoded_len"] = len(best_decoded)
            result["ber"] = best_ber
            result["errors"] = int(np.sum(best_decoded != truth[:len(best_decoded)]))
            result["n_bits"] = len(best_decoded)
            result["ok"] = True
            result["decoded_bits_head"] = best_decoded[:16].tolist()
        else:
            result["error"] = "所有候选 period 均失败"
        result["truth_bits_head"] = truth[:16].tolist() if len(truth) >= 16 else truth.tolist()
    except Exception as e:
        result["error"] = str(e)
    return result


# ── 画图 ──────────────────────────────────────────────────────────────────────
def save_debug_figure(
    out_path: Path,
    frame: np.ndarray,
    leg: Dict,
    title: str,
) -> None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    column = np.mean(gray.astype(np.float64), axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
    fig.suptitle(title, fontsize=12)

    # 上: 原始行均值
    axes[0].plot(column, color="steelblue", linewidth=0.8)
    axes[0].set_title("行均值 (原始灰度)")
    axes[0].set_ylabel("gray value")
    axes[0].grid(alpha=0.3)
    if "_matched_runs" in leg:
        for r in leg["_matched_runs"]:
            axes[0].axvspan(r[0], r[1], color="orange", alpha=0.3)

    # 中: 归一化 + 阈值
    if "_y" in leg and "_threshold" in leg:
        y = leg["_y"]
        thr = leg["_threshold"]
        axes[1].plot(y, color="steelblue", linewidth=0.8, label="normalized")
        axes[1].plot(thr, color="red", linewidth=1.0, label="polyfit threshold")
        axes[1].set_title("归一化信号 + 阈值")
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].grid(alpha=0.3)
    else:
        axes[1].set_title("(legacy_sync 解调失败，无此图)")

    # 下: 二值化 + 同步头标注
    if "_rr" in leg:
        rr = leg["_rr"]
        axes[2].step(np.arange(len(rr)), rr, where="post", color="black", linewidth=0.7)
        if "_matched_runs" in leg:
            for i, r in enumerate(leg["_matched_runs"]):
                axes[2].axvspan(r[0], r[1], color="orange", alpha=0.35,
                                label="sync run" if i == 0 else None)
        pp = leg.get("payload_start")
        bp = leg.get("bit_period")
        if pp is not None and bp is not None:
            pos = float(pp) + float(bp) / 2.0
            xs = []
            while pos < len(rr):
                xs.append(pos)
                pos += float(bp)
            if xs:
                axes[2].scatter(xs, np.full(len(xs), 0.5), s=8, color="red",
                                label="sample positions", zorder=5)
        axes[2].set_ylim(-0.2, 1.2)
        axes[2].set_title("二值化行信号 / 同步头 / 采样点")
        axes[2].legend(loc="upper right", fontsize=8)
        axes[2].grid(alpha=0.3)
    else:
        axes[2].set_title("(legacy_sync 解调失败，无此图)")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


def save_row_mean_only(out_path: Path, frame: np.ndarray, title: str) -> None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    column = np.mean(gray.astype(np.float64), axis=1)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(column, color="steelblue", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("gray value")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=90)
    plt.close(fig)


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUT_DIR}")
    print(f"GainLimit 扫描: {GAIN_LIMITS}")
    print(f"发射标签: {TX_LABEL_CSV}\n")

    truth = _load_truth_bits(TX_LABEL_CSV)
    sync_head_len = _infer_sync_head_len(TX_LABEL_CSV)
    data_bits = len(truth) if len(truth) > 0 else 32
    print(f"真值 bits 数: {data_bits}  同步头长度(推断): {sync_head_len}")
    print(f"真值前 16 位: {truth[:16].tolist()}\n")

    cam = HikvisionCamera(CAMERA_IP, CAMERA_USER, CAMERA_PASSWORD)

    all_records: List[Dict] = []

    for gl in GAIN_LIMITS:
        print(f"{'='*60}")
        print(f"  GainLimit = {gl}")
        # 设置相机
        ok = cam.set_gain_limit(gl)
        if not ok:
            print(f"  ⚠ 设置 GainLimit={gl} 失败，跳过")
            continue
        params = cam.get_current_params() or {}
        actual_gain_level = params.get("gain_level", "?")
        actual_shutter = params.get("shutter_level", "?")
        print(f"  实际 GainLevel={actual_gain_level}  快门={actual_shutter}")

        time.sleep(SETTLE_S)

        # 抓拍
        frame = grab_frame(RTSP_URL)
        if frame is None:
            print(f"  ✗ 抓拍失败，跳过")
            continue

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        row_mean = np.mean(gray.astype(float), axis=1)
        img_brightness = float(np.mean(gray))
        img_std = float(np.std(row_mean))

        print(f"  帧尺寸: {w}x{h}  平均亮度: {img_brightness:.1f}  行均值std: {img_std:.3f}")

        # 保存图像
        step_dir = OUT_DIR / f"gl_{gl:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(step_dir / "frame.png"), frame)
        save_row_mean_only(
            step_dir / "row_mean.png", frame,
            f"GainLimit={gl}  brightness={img_brightness:.1f}  row_std={img_std:.3f}"
        )

        # ── 策略 1: legacy_sync ──────────────────────────────────────────────
        res_leg = demod_legacy(frame, truth, sync_head_len, data_bits)
        _print_result(res_leg, "legacy_sync")

        # ── 策略 2: OOKDemodulator ───────────────────────────────────────────
        res_ook = demod_ook(frame, truth, data_bits)
        _print_result(res_ook, "ook_demod")

        # ── 策略 3: direct_slice ─────────────────────────────────────────────
        res_ds = demod_direct_slice(frame, truth, data_bits, sync_head_len)
        _print_result(res_ds, "direct_slice")

        # ── 保存调试图 (用 legacy_sync 的中间量) ─────────────────────────────
        save_debug_figure(
            step_dir / "demod_debug.png", frame, res_leg,
            f"GainLimit={gl}  legacy_sync  BER={res_leg.get('ber')}"
        )

        # ── 整合记录 ─────────────────────────────────────────────────────────
        record = {
            "gain_limit": gl,
            "actual_gain_level": actual_gain_level,
            "shutter": actual_shutter,
            "brightness": img_brightness,
            "row_std": img_std,
            # legacy_sync
            "leg_ok": res_leg["ok"],
            "leg_sync_count": res_leg.get("sync_count"),
            "leg_bit_period": res_leg.get("bit_period"),
            "leg_ber": res_leg.get("ber"),
            "leg_errors": res_leg.get("errors"),
            "leg_n_bits": res_leg.get("n_bits"),
            "leg_error_msg": res_leg.get("error"),
            "leg_header_len_rows": res_leg.get("header_len_rows"),
            "leg_total_runs": res_leg.get("total_runs_detected"),
            "leg_run_lengths_top5": (res_leg.get("run_lengths") or [])[:5],
            # ook_demod
            "ook_ok": res_ook["ok"],
            "ook_sync_count": res_ook.get("sync_count"),
            "ook_bit_period": res_ook.get("bit_period"),
            "ook_confidence": res_ook.get("confidence"),
            "ook_ber": res_ook.get("ber"),
            "ook_error_msg": res_ook.get("error"),
            # direct_slice
            "ds_ok": res_ds["ok"],
            "ds_bit_period": res_ds.get("bit_period"),
            "ds_ber": res_ds.get("ber"),
            "ds_error_msg": res_ds.get("error"),
        }
        all_records.append(record)

        # 保存单步 JSON
        _safe = {k: (v if not isinstance(v, np.integer) else int(v))
                 for k, v in record.items()}
        (step_dir / "metrics.json").write_text(
            json.dumps(_safe, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("汇总统计表")
    print(f"{'='*60}")
    hdr = (
        f"{'GL':>4} | {'亮度':>7} | {'行std':>6} | "
        f"{'leg_ok':>6} | {'leg_sync':>8} | {'leg_bp':>6} | {'leg_BER':>7} | "
        f"{'ook_ok':>6} | {'ook_sync':>8} | {'ook_BER':>7} | "
        f"{'ds_ok':>5} | {'ds_BER':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in all_records:
        def _fmt_ber(v):
            return f"{v:.4f}" if v is not None else "  N/A "
        print(
            f"{r['gain_limit']:>4} | "
            f"{r['brightness']:>7.1f} | "
            f"{r['row_std']:>6.3f} | "
            f"{'✓' if r['leg_ok'] else '✗':>6} | "
            f"{str(r['leg_sync_count']):>8} | "
            f"{str(r['leg_bit_period']):>6} | "
            f"{_fmt_ber(r['leg_ber']):>7} | "
            f"{'✓' if r['ook_ok'] else '✗':>6} | "
            f"{str(r['ook_sync_count']):>8} | "
            f"{_fmt_ber(r['ook_ber']):>7} | "
            f"{'✓' if r['ds_ok'] else '✗':>5} | "
            f"{_fmt_ber(r['ds_ber']):>7}"
        )

    # ── 保存汇总 ─────────────────────────────────────────────────────────────
    import csv as csv_mod
    summary_csv = OUT_DIR / "summary.csv"
    if all_records:
        keys = list(all_records[0].keys())
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv_mod.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_records:
                row_out = {}
                for k in keys:
                    v = r[k]
                    if isinstance(v, list):
                        row_out[k] = str(v)
                    elif isinstance(v, np.integer):
                        row_out[k] = int(v)
                    else:
                        row_out[k] = v
                w.writerow(row_out)
        print(f"\n汇总 CSV: {summary_csv}")

    summary_json = OUT_DIR / "summary.json"
    def _jsonify(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean_records = []
    for r in all_records:
        clean_records.append({
            k: _jsonify(v) for k, v in r.items()
            if not k.startswith("_")
        })
    summary_json.write_text(
        json.dumps(clean_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"汇总 JSON: {summary_json}")

    # ── 诊断建议 ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("诊断建议")
    print("="*60)

    low_std = [r for r in all_records if r["row_std"] < 1.0]
    if low_std:
        gls = [r["gain_limit"] for r in low_std]
        print(f"⚠  以下 GainLimit 的行均值 std < 1.0 (信号太弱/图像平坦): {gls}")
        print("   → 建议：提高 GainLimit 或调整 LED 频率/功率")

    ok_leg = [r for r in all_records if r["leg_ok"]]
    if ok_leg:
        best = min(ok_leg, key=lambda r: r["leg_ber"])
        print(f"✓  legacy_sync 最佳 BER={best['leg_ber']:.4f} @ GainLimit={best['gain_limit']}"
              f"  (同步头数={best['leg_sync_count']}, bit_period={best['leg_bit_period']})")
    else:
        print("✗  legacy_sync 在所有 GainLimit 下均失败")
        # 分析原因
        reasons: List[str] = []
        for r in all_records:
            if r.get("leg_error_msg"):
                reasons.append(f"  GainLimit={r['gain_limit']}: {r['leg_error_msg']}")
            elif r.get("leg_total_runs") is not None:
                reasons.append(
                    f"  GainLimit={r['gain_limit']}: 检测到 {r['leg_total_runs']} 个游程, "
                    f"最长前5: {r['leg_run_lengths_top5']}"
                )
        if reasons:
            print("   原因摘要:")
            for ln in reasons[:8]:
                print(ln)

    ok_ook = [r for r in all_records if r["ook_ok"]]
    if ok_ook:
        best = min(ok_ook, key=lambda r: r["ook_ber"])
        print(f"✓  OOKDemodulator 最佳 BER={best['ook_ber']:.4f} @ GainLimit={best['gain_limit']}")
    else:
        print("✗  OOKDemodulator 在所有 GainLimit 下均失败")

    ok_ds = [r for r in all_records if r["ds_ok"]]
    if ok_ds:
        best = min(ok_ds, key=lambda r: r["ds_ber"])
        print(f"✓  direct_slice 最佳 BER={best['ds_ber']:.4f} @ GainLimit={best['gain_limit']}"
              f"  bit_period={best['ds_bit_period']}")

    print(f"\n所有调试图保存在: {OUT_DIR}/gl_*/demod_debug.png")
    print("完成！")


def _print_result(r: Dict, tag: str) -> None:
    if r["ok"]:
        print(
            f"  [{tag}] ✓  BER={r.get('ber'):.4f} "
            f"({r.get('errors')}/{r.get('n_bits')})  "
            + (f"sync={r.get('sync_count')}  period={r.get('bit_period')}" if "sync_count" in r else
               f"period={r.get('bit_period')}")
        )
    else:
        err = r.get("error") or "(未知)"
        print(f"  [{tag}] ✗  失败: {err[:80]}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
