"""
OCC 实验实时面板。

运行方式:
    streamlit run realtime_experiment_app.py
"""

import json
import time
from csv import reader
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from camera_isapi import HikvisionCamera
from occ_gain_opt.algorithms import get as algo_get, list_algorithms
from occ_gain_opt.config import CameraParams
from occ_gain_opt.data_sources.roi import (
    compute_roi_stats,
    create_auto_roi_mask,
    create_center_roi_mask,
    create_sync_based_roi_mask,
)
from occ_gain_opt.demodulation import OOKDemodulator


RTSP_URL = "rtsp://admin:abcd1234@192.168.1.19:554/Streaming/Channels/101"
CAMERA_IP = "192.168.1.19"
CAMERA_USER = "admin"
CAMERA_PASSWORD = "abcd1234"
TX_LABEL_CSV = "data/Mseq_32_with_header.csv"
CAPTURE_DIR = Path("results/realtime_captures")
EXPERIMENT_DIR = Path("results/gainlimit_experiment")

GAIN_LEVEL_MIN, GAIN_LEVEL_MAX = 0, 100
ALPHA_MIN, ALPHA_MAX = 0.1, 1.0
TARGET_GRAY = 242.25
TARGET_GRAY_MIN, TARGET_GRAY_MAX = 150.0, 255.0
TOLERANCE_GRAY = 5.0
MAX_HISTORY = 120


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def gain_level_to_db(gain_level: int) -> float:
    return (gain_level / 100.0) * 40.0


def db_to_gain_level(gain_db: float) -> int:
    return int(round(clamp((gain_db / 40.0) * 100.0, GAIN_LEVEL_MIN, GAIN_LEVEL_MAX)))


def gain_level_to_iso(gain_level: int) -> float:
    gain_db = gain_level_to_db(gain_level)
    return 100.0 * (10.0 ** (gain_db / 20.0))


def gain_limit_to_db(gain_limit: int) -> float:
    return (gain_limit / 100.0) * 40.0


def db_to_gain_limit(gain_db: float) -> int:
    return int(round(clamp((gain_db / 40.0) * 100.0, GAIN_LEVEL_MIN, GAIN_LEVEL_MAX)))


def shutter_to_us(shutter: str) -> float:
    if "/" not in shutter:
        return 0.0
    try:
        denominator = int(shutter.split("/", maxsplit=1)[1])
        return 1000000.0 / denominator
    except (ValueError, ZeroDivisionError):
        return 0.0


def resolve_truth_label_path(label_csv_path: str) -> str:
    label_path = Path(label_csv_path)
    if "_with_header" not in label_path.stem:
        return str(label_path)
    original_path = label_path.with_name(label_path.name.replace("_with_header", "_original"))
    if original_path.is_file():
        return str(original_path)
    return str(label_path)


def load_truth_bits(label_csv_path: str) -> np.ndarray:
    truth_path = Path(resolve_truth_label_path(label_csv_path))
    if not truth_path.is_file():
        return np.array([], dtype=np.uint8)

    bits: List[int] = []
    with truth_path.open(encoding="utf-8") as f:
        csv_reader = reader(f)
        for idx, row in enumerate(csv_reader):
            if idx < 6:
                continue
            if len(row) < 2:
                continue
            try:
                bits.append(int(row[1]))
            except ValueError:
                continue
    return np.array(bits, dtype=np.uint8)


def infer_sync_head_len(label_csv_path: str) -> int:
    label_path = Path(label_csv_path)
    if "_with_header" not in label_path.stem or not label_path.is_file():
        return 8

    bits: List[int] = []
    with label_path.open(encoding="utf-8") as f:
        csv_reader = reader(f)
        for idx, row in enumerate(csv_reader):
            if idx < 6:
                continue
            if len(row) < 2:
                continue
            try:
                bits.append(int(row[1]))
            except ValueError:
                continue

    if not bits:
        return 8

    unit = bits[: len(bits) // 3] if len(bits) % 3 == 0 else bits
    run = 0
    started = False
    for bit in unit:
        if not started:
            if bit == 0:
                started = True
            continue
        if bit == 1:
            run += 1
        else:
            break
    return run if run > 0 else 8


def polyfit_threshold(y: np.ndarray, degree: int = 3) -> np.ndarray:
    x = np.arange(1, len(y) + 1)
    coeffs = np.polyfit(x, y, degree)
    return np.polyval(coeffs, x)


def estimate_bit_period_fft(column: np.ndarray, min_period: int = 5, max_period: int = 600) -> Optional[float]:
    """用 FFT 自相关估计行均值中的主要 bit_period（单位: 行）。"""
    y = column - float(np.mean(column))
    if float(np.std(y)) < 1e-6:
        return None
    n = len(y)
    f = np.fft.rfft(y, n=2 * n)
    acor = np.fft.irfft(f * np.conj(f))[:n].real
    if acor[0] <= 0:
        return None
    acor = acor / acor[0]
    search_start = max(1, min_period)
    search_end = min(max_period + 1, n)
    if search_end <= search_start:
        return None
    search = acor[search_start:search_end]
    if len(search) == 0:
        return None
    # 找局部峰值 (高度 > 0.15，间距 >= min_period//2)
    try:
        from scipy.signal import find_peaks
        peaks, props = find_peaks(search, height=0.15, distance=max(1, min_period // 2))
        if len(peaks) > 0:
            best = peaks[int(np.argmax(props["peak_heights"]))]
            return float(best + search_start)
    except Exception:
        pass
    # 退化: 直接取最大值
    best_lag = int(np.argmax(search))
    if search[best_lag] > 0.15:
        return float(best_lag + search_start)
    return None


def check_signal_quality(column: np.ndarray, bit_period_hint: Optional[float] = None) -> Dict[str, object]:
    """
    检查行均值信号质量。
    返回:
      snr_db     : 估计 SNR (dB)，< 0 表示信号被噪声淹没
      detectable : 是否可能存在有效 OCC 信号
      row_std    : 行均值标准差
      row_range  : 行均值极差 (max-min)
      warning    : 诊断警告文字
    """
    std = float(np.std(column))
    rng = float(np.max(column) - np.min(column))
    # 如果知道 bit_period，用理论模型估计 SNR
    snr_db: float = 0.0
    warning = ""
    if bit_period_hint is not None and bit_period_hint >= 5:
        # 理论上相邻 bit 的亮度差应该 ~ 2*std（双峰分布）
        # 用 bit_period 做移动平均平滑，然后看极差
        bp = int(round(bit_period_hint))
        if bp >= 2 and len(column) >= bp * 4:
            smooth = np.convolve(column, np.ones(bp) / bp, mode="valid")
            signal_range = float(np.max(smooth) - np.min(smooth))
            noise_std = float(np.std(column - np.interp(
                np.arange(len(column)),
                np.linspace(0, len(column) - 1, len(smooth)),
                smooth
            )))
            if noise_std > 0:
                snr_db = 20.0 * np.log10(max(signal_range / (2.0 * noise_std), 1e-6))
            else:
                snr_db = 40.0
    else:
        # 没有 hint，用行均值 std 与全局亮度做粗估
        snr_db = 20.0 * np.log10(max(std, 1e-3))

    detectable = std >= 3.0 and rng >= 10.0
    if std < 1.0:
        warning = "行均值方差极低（<1），图像几乎全平，无法解调"
    elif std < 3.0:
        warning = f"信号很弱(std={std:.1f})，解调结果不可靠；建议提高 GainLimit 或 LED 功率"
    elif bit_period_hint is not None and bit_period_hint > 200:
        n_bits_per_frame = len(column) / bit_period_hint
        warning = (
            f"估计 bit_period≈{bit_period_hint:.0f}行，"
            f"每帧仅 {n_bits_per_frame:.1f} 个 bit，"
            f"单帧无法完成完整包解调。建议将 AWG 频率提高到 "
            f"≥{int(25 * len(column) / 36)} Hz"
        )

    return {
        "snr_db": snr_db,
        "detectable": detectable,
        "row_std": std,
        "row_range": rng,
        "warning": warning,
    }


def legacy_find_sync_info(
    rr: np.ndarray,
    head_len: int,
    bit_period_hint: Optional[float] = None,
) -> Dict[str, object]:
    """
    在二值行序列中检测同步头。
    bit_period_hint: 由 FFT 估计的 bit 时长（行数）。提供后能正确设置游程长度范围。
    """
    # 动态设置游程长度范围
    if bit_period_hint is not None and bit_period_hint >= 2:
        min_run = max(2, int(head_len * bit_period_hint * 0.5))
        max_run = int(head_len * bit_period_hint * 2.0)
    else:
        # 无 hint：兜底范围（兼容原有行为）
        min_run = head_len
        max_run = max(100, head_len * 20)

    runs = []
    p = 0
    while p < len(rr):
        if rr[p] == 1:
            q = p
            while q < len(rr) and rr[q] == 1:
                q += 1
            length = q - p
            if min_run <= length <= max_run:
                runs.append((p, q - 1, length))
            p = q
        else:
            p += 1

    if not runs:
        raise ValueError(
            f"未检测到有效同步段 (游程范围 {min_run}-{max_run} 行，"
            f"共检测到 {sum(1 for i in range(len(rr)) if rr[i]==1)} 个1行)"
        )

    runs_sorted = sorted(runs, key=lambda x: x[2], reverse=True)
    h1_start, h1_end, len1 = runs_sorted[0]

    tol = max(5, int(len1 * 0.2))
    matched_runs = [r for r in runs if abs(r[2] - len1) <= tol]
    matched_runs = sorted(matched_runs, key=lambda x: x[0])

    # 找第二个匹配游程（用于计算 equ 和验证间距）
    h2_start = h2_end = len2 = None
    for run in runs_sorted[1:]:
        if abs(run[2] - len1) <= tol:
            h2_start, h2_end, len2 = run
            break

    # 计算 equ：综合两种估计，权重 0.7 给"间距法"、0.3 给"同步头长度法"
    # 原因：噪声会使同步头游程变短，导致长度法低估 equ；而间距法更稳定
    BITS_PER_PACKET = 40  # 本协议固定：1起始 + 6同步 + 1停止 + 32数据
    SPACING_WEIGHT  = 0.7

    equ_sync = max(1.0, abs((len1 + len2) / (head_len * 2))) if h2_start is not None \
               else max(1.0, len1 / head_len)

    # 找两个完整（非帧边截断）同步头，计算间距估算 equ
    mr_complete = [r for r in matched_runs if r[0] > 5]  # 排除从第0行开始的截断游程
    equ_spacing = None
    if len(mr_complete) >= 2:
        spacing = mr_complete[1][0] - mr_complete[0][0]
        if BITS_PER_PACKET * 5 < spacing < BITS_PER_PACKET * 30:  # 合理范围校验
            equ_spacing = spacing / BITS_PER_PACKET

    if equ_spacing is not None:
        equ = SPACING_WEIGHT * equ_spacing + (1 - SPACING_WEIGHT) * equ_sync
    else:
        equ = equ_sync

    # 选取 payload_start：从完整同步头中选第一个数据能完整装入帧的
    frame_len = len(rr)
    needed_rows = int((1 + head_len + 32) * equ) + 5
    candidate_runs = mr_complete if mr_complete else matched_runs
    best_sync = candidate_runs[0]
    for mr in candidate_runs:
        ps_candidate = mr[1] + 1
        if ps_candidate + needed_rows <= frame_len:
            best_sync = mr
            break

    payload_start = best_sync[1] + 1
    sync_header_start = matched_runs[0][0]

    return {
        "equ": equ,
        "payload_start": payload_start,
        "sync_start": sync_header_start,
        "matched_runs": matched_runs,
        "header_len_rows": len1,
    }


def legacy_recover_data(rr: np.ndarray, payload_start: int, equ_len: float) -> np.ndarray:
    """
    均匀采样解码：在 payload_start + (k+0.5)*equ_len 处直接采样第 k 个 bit。
    比原来的游程RLE方法更鲁棒：±1行的游程噪声不会导致相位积累漂移。
    """
    res: List[int] = []
    k = 0
    while True:
        pos = int(payload_start + (k + 0.5) * equ_len)
        if pos >= len(rr):
            break
        res.append(int(rr[pos]))
        k += 1
    return np.array(res, dtype=np.uint8)


def create_legacy_sync_roi_mask(
    image_shape: Tuple[int, int],
    sync_runs: List[Tuple[int, int, int]],
    packet_len_rows: int,
) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not sync_runs:
        return mask

    starts = [int(r[0]) for r in sync_runs]
    for idx, row_start in enumerate(starts):
        if idx + 1 < len(starts):
            row_end = starts[idx + 1]
        else:
            row_end = min(row_start + packet_len_rows, h)
        if row_end > row_start:
            mask[row_start:row_end, :] = 1
    return mask


def build_demod_debug_figure(metrics: Dict[str, object]):
    row_profile = np.array(metrics.get("row_profile", []), dtype=float)
    normalized_profile = np.array(metrics.get("normalized_profile", []), dtype=float)
    threshold_curve = np.array(metrics.get("threshold_curve", []), dtype=float)
    binary_profile = np.array(metrics.get("binary_profile", []), dtype=float)
    sync_runs = metrics.get("sync_runs", [])
    sample_positions = np.array(metrics.get("sample_positions", []), dtype=float)

    if row_profile.size == 0:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(row_profile, color="#00aa66", linewidth=1.2, label="row mean")
    axes[0].set_ylabel("gray")
    axes[0].set_title("行均值曲线")
    axes[0].grid(alpha=0.25)
    for idx, run in enumerate(sync_runs):
        start, end, _ = run
        axes[0].axvspan(start, end, color="orange", alpha=0.2,
                        label="sync run" if idx == 0 else None)
    if sync_runs:
        axes[0].legend(loc="upper right")

    if normalized_profile.size > 0:
        axes[1].plot(normalized_profile, color="#1f77b4", linewidth=1.1, label="normalized")
    if threshold_curve.size > 0:
        axes[1].plot(threshold_curve, color="#d62728", linewidth=1.1, label="threshold")
    axes[1].set_ylabel("normalized")
    axes[1].set_title("归一化信号与阈值")
    axes[1].grid(alpha=0.25)
    if normalized_profile.size > 0 or threshold_curve.size > 0:
        axes[1].legend(loc="upper right")

    if binary_profile.size > 0:
        axes[2].step(np.arange(len(binary_profile)), binary_profile, where="post",
                     color="black", linewidth=1.0, label="binary")
    if sample_positions.size > 0:
        axes[2].scatter(
            sample_positions,
            np.full_like(sample_positions, 0.5, dtype=float),
            s=10,
            color="red",
            label="sample positions",
        )
    for idx, run in enumerate(sync_runs):
        start, end, _ = run
        axes[2].axvspan(start, end, color="orange", alpha=0.2,
                        label="sync run" if idx == 0 else None)
    axes[2].set_ylim(-0.2, 1.2)
    axes[2].set_ylabel("binary")
    axes[2].set_xlabel("row index")
    axes[2].set_title("二值化行信号 / 采样点 / 同步头")
    axes[2].grid(alpha=0.25)
    if binary_profile.size > 0 or sample_positions.size > 0 or sync_runs:
        axes[2].legend(loc="upper right")

    plt.tight_layout()
    return fig


def init_state() -> None:
    defaults = {
        "hik_camera": HikvisionCamera(CAMERA_IP, CAMERA_USER, CAMERA_PASSWORD),
        "connected": False,
        "rtsp_url": RTSP_URL,
        "frame": None,
        "frame_rgb": None,
        "roi_overlay": None,
        "roi_crop": None,
        "metrics": {},
        "message": "",
        "records": [],
        "last_snapshot": "",
        "camera_params_cache": {},
        "last_params_refresh_ts": 0.0,
        "current_params": CameraParams(iso=35.0, exposure_us=100.0),
        "calibration_records": [],
        "experiment_records": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def push_message(message: str) -> None:
    st.session_state.message = message


def reset_camera() -> None:
    st.session_state.connected = False


def get_camera_client(ip: str, username: str, password: str) -> HikvisionCamera:
    current = st.session_state.get("hik_camera")
    if (
        current is None
        or current.ip != ip
        or current.username != username
        or current.password != password
    ):
        current = HikvisionCamera(ip=ip, username=username, password=password)
        st.session_state.hik_camera = current
    return current


def refresh_camera_params(hik_camera: HikvisionCamera) -> Dict[str, str]:
    params = hik_camera.get_current_params() or {}
    st.session_state.camera_params_cache = params
    st.session_state.last_params_refresh_ts = time.time()

    gain_level = int(params.get("gain_level", 0) or 0)
    shutter_us = shutter_to_us(params.get("shutter_level", ""))
    exposure_us = shutter_us if shutter_us > 0 else st.session_state.current_params.exposure_us
    st.session_state.current_params = CameraParams(
        iso=gain_level_to_iso(gain_level),
        exposure_us=exposure_us,
    )
    return params


def get_cached_or_refresh_camera_params(
    hik_camera: HikvisionCamera,
    *,
    force: bool = False,
    min_interval_s: float = 2.0,
) -> Dict[str, str]:
    last_ts = float(st.session_state.get("last_params_refresh_ts", 0.0))
    cached = st.session_state.get("camera_params_cache", {})
    if force or not cached or (time.time() - last_ts) >= min_interval_s:
        return refresh_camera_params(hik_camera)
    return cached


def calibration_to_arrays(calibration_records: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    if not calibration_records:
        return np.array([], dtype=float), np.array([], dtype=float)
    ordered = sorted(calibration_records, key=lambda r: r["gain_limit"])
    x = np.array([float(r["gain_limit"]) for r in ordered], dtype=float)
    y = np.array([float(r["equivalent_gain_db"]) for r in ordered], dtype=float)
    return x, y


def map_gain_limit_to_db(gain_limit: int, calibration_records: List[Dict[str, float]]) -> float:
    x, y = calibration_to_arrays(calibration_records)
    if len(x) >= 2:
        return float(np.interp(float(gain_limit), x, y))
    return gain_limit_to_db(gain_limit)


def map_db_to_gain_limit(gain_db: float, calibration_records: List[Dict[str, float]]) -> int:
    x, y = calibration_to_arrays(calibration_records)
    if len(x) >= 2:
        return int(round(float(np.interp(float(gain_db), y, x))))
    return db_to_gain_limit(gain_db)


def build_current_camera_params(
    params: Dict[str, str],
    calibration_records: List[Dict[str, float]],
    previous_exposure_us: float,
) -> CameraParams:
    gain_limit = int(params.get("gain_limit", params.get("gain_level", 0)) or 0)
    gain_db = map_gain_limit_to_db(gain_limit, calibration_records)
    shutter_us = shutter_to_us(params.get("shutter_level", ""))
    exposure_us = shutter_us if shutter_us > 0 else previous_exposure_us
    return CameraParams(iso=100.0 * (10.0 ** (gain_db / 20.0)), exposure_us=exposure_us)


def capture_frame_batch(rtsp_url: str, n_frames: int, settle_ms: int = 400) -> List[np.ndarray]:
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("无法打开 RTSP 流")

    time.sleep(max(settle_ms, 0) / 1000.0)
    frames: List[np.ndarray] = []
    attempts = 0
    max_attempts = max(n_frames * 4, 20)
    while len(frames) < n_frames and attempts < max_attempts:
        ok, frame = cap.read()
        attempts += 1
        if ok and frame is not None:
            frames.append(frame.copy())
    cap.release()

    if not frames:
        raise RuntimeError("批量抓拍失败，未获得有效帧")
    return frames


def analyze_frame_batch(
    frames: List[np.ndarray],
    roi_strategy: str,
    roi_size: int,
    label_csv_path: str,
) -> Dict[str, object]:
    frame_results = [analyze_frame(frame, roi_strategy, roi_size, label_csv_path) for frame in frames]
    brightnesses = [float(r["stats"]["mean"]) for r in frame_results]
    bers = [float(r["ber"]) for r in frame_results if r["ber"] is not None]
    best_idx = int(np.argmin(bers)) if bers else 0

    summary = {
        "frame_count": len(frame_results),
        "brightness_median": float(np.median(brightnesses)) if brightnesses else 0.0,
        "brightness_mean": float(np.mean(brightnesses)) if brightnesses else 0.0,
        "ber_median": float(np.median(bers)) if bers else None,
        "ber_best": float(np.min(bers)) if bers else None,
        "analysis_for_display": frame_results[best_idx if best_idx < len(frame_results) else 0],
        "all_results": frame_results,
    }
    return summary


def run_gainlimit_calibration(
    hik_camera: HikvisionCamera,
    rtsp_url: str,
    label_csv_path: str,
    *,
    roi_strategy: str,
    roi_size: int,
    limits: List[int],
    n_frames: int,
) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    baseline_brightness = None

    for gain_limit in limits:
        hik_camera.set_gain_limit(int(gain_limit))
        frames = capture_frame_batch(rtsp_url, n_frames)
        batch = analyze_frame_batch(frames, roi_strategy, roi_size, label_csv_path)
        brightness = float(batch["brightness_median"])
        if baseline_brightness is None:
            baseline_brightness = max(brightness, 1.0)
        equivalent_gain_db = 20.0 * np.log10(max(brightness, 1.0) / baseline_brightness)
        records.append(
            {
                "gain_limit": int(gain_limit),
                "brightness_median": brightness,
                "ber_median": batch["ber_median"],
                "equivalent_gain_db": float(equivalent_gain_db),
            }
        )
    return records


def compute_next_control_suggestion(
    algo_name: str,
    current_params: CameraParams,
    brightness: float,
    alpha: float,
    target_gray: float,
    calibration_records: List[Dict[str, float]],
    ber: Optional[float] = None,
) -> Dict[str, float]:
    gain_db_min = gain_level_to_db(GAIN_LEVEL_MIN)
    gain_db_max = gain_level_to_db(GAIN_LEVEL_MAX)
    if algo_name == "adaptive_iter":
        algo = algo_get(algo_name)(
            alpha=alpha,
            target_gray=target_gray,
            gain_db_min=gain_db_min,
            gain_db_max=gain_db_max,
        )
    elif algo_name == "single_shot":
        algo = algo_get(algo_name)(
            target_gray=target_gray,
            gain_db_min=gain_db_min,
            gain_db_max=gain_db_max,
        )
    else:
        algo = algo_get(algo_name)()

    next_params = algo.compute_next_params(current_params, brightness, ber)
    suggested_gain_limit = map_db_to_gain_limit(float(next_params.gain_db), calibration_records)
    return {
        "gain_db": float(next_params.gain_db),
        "gain_limit": int(suggested_gain_limit),
        "iso": float(next_params.iso),
        "exposure_us": float(next_params.exposure_us),
    }


def run_gainlimit_experiment(
    hik_camera: HikvisionCamera,
    rtsp_url: str,
    label_csv_path: str,
    *,
    algo_name: str,
    calibration_records: List[Dict[str, float]],
    steps: int,
    frames_per_step: int,
    roi_strategy: str,
    roi_size: int,
    alpha: float,
    target_gray: float,
) -> List[Dict[str, object]]:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []

    hik_camera.set_gain(0)
    hik_camera.set_gain_limit(0)
    current_params = CameraParams(iso=100.0, exposure_us=100.0)
    current_gain_limit = 0

    for step_idx in range(steps):
        frames = capture_frame_batch(rtsp_url, frames_per_step)
        batch = analyze_frame_batch(frames, roi_strategy, roi_size, label_csv_path)
        display = batch["analysis_for_display"]
        brightness = float(batch["brightness_median"])
        ber = batch["ber_median"]

        suggestion = compute_next_control_suggestion(
            algo_name,
            current_params,
            brightness,
            alpha,
            target_gray,
            calibration_records,
            ber=ber,
        )

        record = {
            "step": step_idx,
            "applied_gain_limit": current_gain_limit,
            "brightness_median": brightness,
            "ber_median": ber,
            "ber_best": batch["ber_best"],
            "suggested_gain_db": suggestion["gain_db"],
            "next_gain_limit": suggestion["gain_limit"],
            "sync_count": int(display.get("sync_count", 0) or 0),
            "bit_period": display.get("bit_period"),
        }
        records.append(record)

        current_gain_limit = int(suggestion["gain_limit"])
        hik_camera.set_gain_limit(current_gain_limit)
        current_params = CameraParams(
            iso=100.0 * (10.0 ** (suggestion["gain_db"] / 20.0)),
            exposure_us=float(suggestion["exposure_us"]),
        )

    return records


def read_single_frame(rtsp_url: str) -> np.ndarray:
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("无法打开 RTSP 流")

    ok = False
    frame = None
    for _ in range(5):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("无法从 RTSP 流读取图像")
    return frame


def get_roi_mask(image: np.ndarray, strategy: str, roi_size: int) -> Tuple[np.ndarray, str]:
    if strategy == "sync_based":
        try:
            return create_sync_based_roi_mask(image), "sync_based"
        except Exception:
            strategy = "auto"
    if strategy == "auto":
        try:
            return create_auto_roi_mask(image, roi_size=roi_size), "auto"
        except Exception:
            strategy = "center"
    return create_center_roi_mask(image, roi_size=roi_size), "center"


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask == 1)
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2 + 1, y2 + 1


def draw_roi_overlay(frame: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    overlay = frame.copy()
    if mask.ndim == 2 and np.any(mask > 0):
        color_layer = np.zeros_like(overlay)
        color_layer[:, :, 1] = 180
        overlay = np.where(mask[:, :, None] > 0, cv2.addWeighted(overlay, 0.65, color_layer, 0.35, 0), overlay)
    x1, y1, x2, y2 = mask_bbox(mask)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        overlay,
        f"ROI: {label}",
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return overlay


def crop_roi(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = mask_bbox(mask)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame
    return crop


def compute_row_profile(gray: np.ndarray, mask: np.ndarray) -> List[float]:
    x1, y1, x2, y2 = mask_bbox(mask)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return []
    return roi.mean(axis=1).astype(float).tolist()


def analyze_frame(
    frame: np.ndarray,
    roi_strategy: str,
    roi_size: int,
    label_csv_path: str,
) -> Dict[str, object]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    truth_bits = load_truth_bits(label_csv_path)
    data_bits = int(len(truth_bits)) if len(truth_bits) > 0 else 32

    demod_result = None
    demod_error = None
    decoded_bits: List[int] = []
    legacy_sync_runs: List[Tuple[int, int, int]] = []
    bit_period = None
    sync_count = 0
    confidence = 0.0
    normalized_profile: List[float] = []
    threshold_curve: List[float] = []
    binary_profile: List[int] = []
    sample_positions: List[float] = []
    payload_start = None
    signal_quality: Dict[str, object] = {}
    try:
        column = np.mean(gray.astype(np.float64), axis=1)
        std = float(np.std(column))
        if std < 1e-6:
            raise ValueError("图像行均值方差为零，无法解调")

        # ── FFT 估计 bit_period ──────────────────────────────────────────────
        fft_period = estimate_bit_period_fft(column, min_period=5, max_period=600)
        signal_quality = check_signal_quality(column, bit_period_hint=fft_period)

        y = (column - float(np.mean(column))) / std
        threshold = polyfit_threshold(y, degree=3)
        rr = (y - threshold > 0).astype(np.uint8)
        normalized_profile = y.astype(float).tolist()
        threshold_curve = threshold.astype(float).tolist()
        binary_profile = rr.astype(int).tolist()
        sync_head_len = infer_sync_head_len(label_csv_path)

        # ── 用行均值自相关精确估算 bit_period（包周期/40，信号够强时最准确）────────
        BITS_PER_PACKET = 40
        equ_from_autocor: Optional[float] = None
        try:
            rr_f = column - float(np.mean(column))
            ac_full = np.correlate(rr_f, rr_f, mode='full')
            ac = ac_full[len(ac_full) // 2:]
            ac_norm = ac / (ac[0] + 1e-9)
            # 在包周期期望范围内找正的局部极大值峰（abs>0.02时认为信号可检测）
            search_min, search_max = 430, 570
            if search_max < len(ac_norm):
                region = ac_norm[search_min:search_max]
                pos_peaks = []
                for i in range(1, len(region) - 1):
                    if region[i] > region[i-1] and region[i] > region[i+1] and region[i] > 0.02:
                        pos_peaks.append((i + search_min, float(region[i])))
                if pos_peaks:
                    # 选强度最大的正峰
                    best_peak = max(pos_peaks, key=lambda p: p[1])
                    equ_from_autocor = float(best_peak[0]) / BITS_PER_PACKET
        except Exception:
            pass

        # ── 传入 FFT 估计的 bit_period，修正游程长度范围 ─────────────────────
        sync_info = legacy_find_sync_info(rr, sync_head_len, bit_period_hint=fft_period)
        # 优先使用自相关 equ（信号足够强时最准确），兜底使用 sync_info 内的估计
        equ_final = equ_from_autocor if equ_from_autocor is not None else float(sync_info["equ"])
        raw = legacy_recover_data(rr, int(sync_info["payload_start"]), equ_final)
        decoded = raw[1 : data_bits + 1]
        decoded_bits = decoded.astype(np.uint8).tolist()
        legacy_sync_runs = list(sync_info["matched_runs"])
        bit_period = equ_final
        sync_count = len(legacy_sync_runs)
        confidence = 1.0 if sync_count >= 2 else (0.5 if sync_count == 1 else 0.0)
        payload_start = int(sync_info["payload_start"])

        packet_len_rows = 0
        if sync_count >= 2:
            starts = [int(r[0]) for r in legacy_sync_runs]
            packet_len_rows = int(np.median(np.diff(starts)))
        else:
            packet_len_rows = int(round((sync_head_len + data_bits) * bit_period))

        mask = create_legacy_sync_roi_mask(frame.shape, legacy_sync_runs, packet_len_rows)
        roi_used = "legacy_sync"
        profile = column.astype(float).tolist()
        if payload_start is not None and bit_period is not None:
            pos = payload_start + bit_period / 2.0
            while pos < len(rr):
                sample_positions.append(float(pos))
                pos += bit_period
    except Exception as exc:
        demod_error = str(exc)
        try:
            demod_result = OOKDemodulator(config={"data_bits": data_bits}).demodulate(frame)
        except Exception as inner_exc:
            demod_error = f"{demod_error}; fallback: {inner_exc}"

    if demod_result is not None and int(np.sum(demod_result.roi_mask)) > 0:
        mask = demod_result.roi_mask
        roi_used = "sync_based"
        profile = demod_result.row_profile.astype(float).tolist()
        bit_period = float(demod_result.bit_period)
        sync_count = len(demod_result.sync_positions_row)
        confidence = float(demod_result.confidence)
        normalized_profile = demod_result.row_profile.astype(float).tolist()
        threshold_curve = [float(demod_result.threshold)] * len(profile)
        binary_profile = demod_result.binary_profile.astype(int).tolist()
        sample_positions = demod_result.sample_positions.astype(float).tolist()
        legacy_sync_runs = [
            (int(start), int(start + round(bit_period * 8)), int(round(bit_period * 8)))
            for start in demod_result.sync_positions_row
        ]
    else:
        if "mask" not in locals():
            mask, roi_used = get_roi_mask(frame, roi_strategy, roi_size)
            profile = compute_row_profile(gray, mask)
            bit_period = None
            sync_count = 0
            confidence = 0.0

    stats = compute_roi_stats(gray, mask)

    ber = None
    errors = None
    n_bits = None
    ber_error = None
    if decoded_bits:
        if len(truth_bits) > 0:
            n_bits = int(min(len(decoded_bits), len(truth_bits)))
            if n_bits > 0:
                errors = int(np.sum(np.array(decoded_bits[:n_bits], dtype=np.uint8) != truth_bits[:n_bits]))
                ber = errors / n_bits
            else:
                ber_error = "恢复数据长度为零"
    elif demod_result is not None and demod_result.packets:
        packet = demod_result.packets[0].astype(np.uint8)
        if len(packet) > 0:
            decoded_bits = packet.tolist()
            if len(truth_bits) > 0:
                n_bits = int(min(len(packet), len(truth_bits)))
                if n_bits > 0:
                    errors = int(np.sum(packet[:n_bits] != truth_bits[:n_bits]))
                    ber = errors / n_bits
                else:
                    ber_error = "恢复数据长度为零"
    elif demod_result is not None:
        ber_error = "检测到同步头，但未提取出完整 payload"
    elif demod_error:
        ber_error = demod_error

    return {
        "mask": mask,
        "roi_used": roi_used,
        "stats": stats,
        "row_profile": profile,
        "ber": ber,
        "errors": errors,
        "n_bits": n_bits,
        "ber_error": ber_error,
        "bit_period": bit_period,
        "sync_count": sync_count,
        "confidence": confidence,
        "decoded_bits": decoded_bits,
        "truth_bits": truth_bits.tolist(),
        "normalized_profile": normalized_profile,
        "threshold_curve": threshold_curve,
        "binary_profile": binary_profile,
        "sync_runs": legacy_sync_runs,
        "sample_positions": sample_positions,
        "payload_start": payload_start,
        "signal_quality": signal_quality,
    }


def append_record(metrics: Dict[str, object], gain_level: int, gain_limit: int) -> None:
    record = {
        "ts": datetime.now().strftime("%H:%M:%S"),
        "brightness": float(metrics["stats"]["mean"]),
        "std": float(metrics["stats"]["std"]),
        "saturated_ratio": float(metrics["stats"]["saturated_ratio"]),
        "gain_level": int(gain_level),
        "gain_limit": int(gain_limit),
        "gain_db": float(gain_level_to_db(gain_level)),
        "ber": None if metrics["ber"] is None else float(metrics["ber"]),
    }
    records = st.session_state.records
    if not records or records[-1] != record:
        records.append(record)
    if len(records) > MAX_HISTORY:
        del records[:-MAX_HISTORY]


def save_snapshot(frame: np.ndarray, roi_crop_bgr: np.ndarray, metrics: Dict[str, object]) -> str:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_path = CAPTURE_DIR / f"frame_{stamp}.png"
    roi_path = CAPTURE_DIR / f"roi_{stamp}.png"
    meta_path = CAPTURE_DIR / f"metrics_{stamp}.json"

    cv2.imwrite(str(frame_path), frame)
    cv2.imwrite(str(roi_path), roi_crop_bgr)

    payload = {
        "timestamp": stamp,
        "metrics": {
            "roi_used": metrics["roi_used"],
            "ber": metrics["ber"],
            "errors": metrics["errors"],
            "n_bits": metrics["n_bits"],
            "ber_error": metrics["ber_error"],
            **metrics["stats"],
        },
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    st.session_state.last_snapshot = str(frame_path)
    return str(frame_path)


def compute_algo_suggestions(
    current_params: CameraParams,
    brightness: float,
    alpha: float,
    target_gray: float,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for name in list_algorithms():
        try:
            gain_db_min = gain_level_to_db(GAIN_LEVEL_MIN)
            gain_db_max = gain_level_to_db(GAIN_LEVEL_MAX)
            if name == "adaptive_iter":
                algo = algo_get(name)(
                    alpha=alpha,
                    target_gray=target_gray,
                    gain_db_min=gain_db_min,
                    gain_db_max=gain_db_max,
                )
            elif name == "single_shot":
                algo = algo_get(name)(
                    target_gray=target_gray,
                    gain_db_min=gain_db_min,
                    gain_db_max=gain_db_max,
                )
            else:
                algo = algo_get(name)()
            next_params = algo.compute_next_params(current_params, brightness)
            results[name] = {
                "iso": float(next_params.iso),
                "exposure_us": float(next_params.exposure_us),
                "gain_db": float(next_params.gain_db),
                "gain_level": db_to_gain_level(next_params.gain_db),
            }
        except Exception as exc:
            results[name] = {"error": str(exc)}
    return results


def connect_camera(rtsp_url: str) -> None:
    reset_camera()
    frame = read_single_frame(rtsp_url)
    st.session_state.frame = frame
    st.session_state.rtsp_url = rtsp_url
    st.session_state.connected = True
    push_message("相机连接成功。")


def fetch_latest_frame() -> Optional[np.ndarray]:
    rtsp_url = st.session_state.get("rtsp_url")
    if not rtsp_url:
        return None
    frame = read_single_frame(rtsp_url)
    st.session_state.frame = frame
    return frame


def main() -> None:
    st.set_page_config(
        page_title="OCC 实验实时面板",
        page_icon="📷",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()

    st.title("OCC 实验实时面板")
    st.caption("实时画面、ROI、抓拍、BER 分析与参数建议")

    st.sidebar.header("连接设置")
    rtsp_url = st.sidebar.text_input("RTSP 地址", value=RTSP_URL)
    camera_ip = CAMERA_IP
    username = CAMERA_USER
    password = CAMERA_PASSWORD
    with st.sidebar.expander("高级设置"):
        camera_ip = st.text_input("相机 IP", value=CAMERA_IP)
        username = st.text_input("用户名", value=CAMERA_USER)
        password = st.text_input("密码", value=CAMERA_PASSWORD, type="password")
    hik_camera = get_camera_client(camera_ip, username, password)

    btn_col1, btn_col2 = st.sidebar.columns(2)
    connect_btn = btn_col1.button("连接相机", width="stretch")
    disconnect_btn = btn_col2.button("断开", width="stretch")

    if connect_btn:
        try:
            connect_camera(rtsp_url)
        except Exception as exc:
            push_message(f"连接失败: {exc}")

    if disconnect_btn:
        reset_camera()
        push_message("已断开相机。")

    refresh_params_btn = st.sidebar.button("刷新相机参数", width="stretch")
    params = get_cached_or_refresh_camera_params(hik_camera, force=refresh_params_btn, min_interval_s=3.0)
    calibration_records = st.session_state.calibration_records
    st.session_state.current_params = build_current_camera_params(
        params,
        calibration_records,
        st.session_state.current_params.exposure_us,
    )

    st.sidebar.header("实验设置")
    tx_label_csv = st.sidebar.text_input("发射序列 CSV", value=TX_LABEL_CSV)
    st.sidebar.caption(f"BER 真值比对文件: `{resolve_truth_label_path(tx_label_csv)}`")
    selected_algo = st.sidebar.selectbox("推荐算法", list_algorithms())
    roi_strategy = "sync_based"
    roi_size = 280
    alpha = 0.5
    target_gray = TARGET_GRAY
    auto_refresh = False
    refresh_interval = 0.6
    with st.sidebar.expander("高级实验设置"):
        roi_strategy = st.selectbox("ROI 策略", ["sync_based", "auto", "center"], index=0)
        roi_size = st.slider("ROI 大小", 50, 800, 280, step=10)
        alpha = st.slider("学习率 α", ALPHA_MIN, ALPHA_MAX, 0.5, step=0.05)
        target_gray = st.slider("目标灰度", TARGET_GRAY_MIN, TARGET_GRAY_MAX, TARGET_GRAY, step=1.0)
        auto_refresh = st.checkbox("自动刷新", value=False)
        refresh_interval = st.slider("刷新间隔 (秒)", 0.2, 2.0, 0.6, step=0.1)

    st.sidebar.header("相机参数")
    gain_level = int(params.get("gain_level", 0) or 0)
    gain_limit = int(params.get("gain_limit", 0) or 0)
    shutter_level = params.get("shutter_level", "N/A")
    exposure_mode = params.get("exposure_type", "N/A")
    st.sidebar.metric("当前 GainLevel", gain_level)
    st.sidebar.metric("当前 GainLimit", gain_limit)
    st.sidebar.metric("快门", shutter_level)
    st.sidebar.metric("曝光模式", exposure_mode)
    st.sidebar.caption(
        f"算法估算参考: GainLimit≈{map_gain_limit_to_db(gain_limit, calibration_records):+.2f} dB, "
        f"GainLevel≈{gain_level_to_db(gain_level):+.2f} dB"
    )

    gain_limit_setting = st.sidebar.slider("设置 GainLimit", GAIN_LEVEL_MIN, GAIN_LEVEL_MAX, gain_limit, step=1)
    gain_level_setting = st.sidebar.slider("设置 GainLevel", GAIN_LEVEL_MIN, GAIN_LEVEL_MAX, gain_level, step=1)
    set_col1, set_col2 = st.sidebar.columns(2)
    set_limit_btn = set_col1.button("应用上限", width="stretch")
    set_gain_btn = set_col2.button("应用增益", width="stretch")
    if set_limit_btn:
        success = hik_camera.set_gain_limit(gain_limit_setting)
        push_message("增益上限设置成功。" if success else "增益上限设置失败。")
        if success:
            params = refresh_camera_params(hik_camera)
            gain_limit = int(params.get("gain_limit", gain_limit_setting) or gain_limit_setting)
    if set_gain_btn:
        success = hik_camera.set_gain(gain_level_setting)
        push_message("增益设置成功。" if success else "增益设置失败。")
        if success:
            params = refresh_camera_params(hik_camera)
            gain_level = int(params.get("gain_level", gain_level_setting) or gain_level_setting)

    frame = None
    analysis: Dict[str, object] = {}
    if st.session_state.connected:
        try:
            frame = fetch_latest_frame()
        except Exception as exc:
            push_message(f"取帧失败: {exc}")
            reset_camera()

    if frame is not None:
        analysis = analyze_frame(frame, roi_strategy, roi_size, tx_label_csv)
        st.session_state.metrics = analysis
        overlay_bgr = draw_roi_overlay(frame, analysis["mask"], str(analysis["roi_used"]))
        roi_crop_bgr = crop_roi(frame, analysis["mask"])
        st.session_state.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.session_state.roi_overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        st.session_state.roi_crop = cv2.cvtColor(roi_crop_bgr, cv2.COLOR_BGR2RGB)
        append_record(analysis, gain_level, gain_limit)

    if st.session_state.message:
        st.info(st.session_state.message)

    metrics = st.session_state.metrics or {}
    stats = metrics.get("stats", {})
    brightness = float(stats.get("mean", 0.0))
    roi_std = float(stats.get("std", 0.0))
    sat_ratio = float(stats.get("saturated_ratio", 0.0))
    target_error = brightness - target_gray
    bit_period = metrics.get("bit_period")
    sync_count = int(metrics.get("sync_count", 0) or 0)
    confidence = float(metrics.get("confidence", 0.0) or 0.0)

    top1, top2, top3, top4, top5 = st.columns(5)
    top1.metric("连接状态", "已连接" if st.session_state.connected else "未连接")
    top2.metric("ROI 亮度", f"{brightness:.2f}")
    top3.metric("ROI 标准差", f"{roi_std:.2f}")
    top4.metric("饱和比例", f"{sat_ratio * 100:.2f}%")
    top5.metric("当前 GainLimit", f"{gain_limit}")

    if frame is not None:
        h, w = frame.shape[:2]
        st.caption(f"当前帧分辨率: {w}x{h}")

    main_col, side_col = st.columns([2.2, 1.1])

    with main_col:
        st.subheader("实时画面")
        tab1, tab2, tab3 = st.tabs(["ROI 叠加", "ROI 放大", "原始帧"])
        with tab1:
            if st.session_state.roi_overlay is not None:
                st.image(st.session_state.roi_overlay, width="stretch")
            else:
                st.warning("尚未获取到图像。")
        with tab2:
            if st.session_state.roi_crop is not None:
                st.image(st.session_state.roi_crop, width="stretch")
            else:
                st.warning("尚未生成 ROI。")
        with tab3:
            if st.session_state.frame_rgb is not None:
                st.image(st.session_state.frame_rgb, width="stretch")
            else:
                st.warning("尚未获取到图像。")

        action1, action2, action3 = st.columns(3)
        refresh_btn = action1.button("刷新一帧", width="stretch")
        snapshot_btn = action2.button("抓拍保存", width="stretch")
        analyze_btn = action3.button("单步算法应用", width="stretch")

        if refresh_btn:
            st.rerun()
        if snapshot_btn and frame is not None and st.session_state.roi_crop is not None:
            roi_crop_bgr = cv2.cvtColor(st.session_state.roi_crop, cv2.COLOR_RGB2BGR)
            saved = save_snapshot(frame, roi_crop_bgr, metrics)
            push_message(f"已保存抓拍: {saved}")
            st.rerun()
        if analyze_btn and brightness > 0:
            try:
                suggestion = compute_next_control_suggestion(
                    selected_algo,
                    st.session_state.current_params,
                    brightness,
                    alpha,
                    target_gray,
                    calibration_records,
                    ber=metrics.get("ber"),
                )
                success = hik_camera.set_gain_limit(int(suggestion["gain_limit"]))
                if success:
                    st.session_state.current_params = CameraParams(
                        iso=float(suggestion["iso"]),
                        exposure_us=float(suggestion["exposure_us"]),
                    )
                    push_message(
                        f"已应用 {selected_algo}: {suggestion['gain_db']:+.2f} dB / GainLimit {suggestion['gain_limit']}"
                    )
                else:
                    push_message(f"{selected_algo} 应用失败。")
                st.rerun()
            except Exception as exc:
                push_message(f"算法应用失败: {exc}")
                st.rerun()

        st.subheader("条纹分析")
        profile = metrics.get("row_profile", [])
        if profile:
            st.line_chart(profile)
        else:
            st.caption("暂无行均值曲线。")

        st.subheader("解调调试图")
        debug_fig = build_demod_debug_figure(metrics)
        if debug_fig is not None:
            st.pyplot(debug_fig)
            plt.close(debug_fig)
        else:
            st.caption("暂无可视化调试数据。")

    with side_col:
        st.subheader("当前分析")
        st.write(f"ROI 方法: `{metrics.get('roi_used', 'N/A')}`")
        st.write(f"目标灰度: `{target_gray:.1f}`")
        st.write(f"亮度误差: `{target_error:+.2f}`")
        st.write(f"当前 GainLevel: `{gain_level}`")
        st.write(f"当前 GainLimit: `{gain_limit}`")
        st.write(f"GainLimit 对应估算增益: `{map_gain_limit_to_db(gain_limit, calibration_records):+.2f} dB`")
        st.write(f"快门时间: `{shutter_level}`")
        st.write(f"同步头个数: `{sync_count}`")
        st.write(f"位周期: `{bit_period:.2f}`" if bit_period else "位周期: `N/A`")
        st.write(f"解调置信度: `{confidence:.2f}`")
        payload_start = metrics.get("payload_start")
        st.write(f"payload 起点: `{payload_start}`" if payload_start is not None else "payload 起点: `N/A`")

        # ── 信号质量指示 ──────────────────────────────────────────────────────
        sq = metrics.get("signal_quality", {})
        if sq:
            sq_row_std = float(sq.get("row_std", 0))
            sq_row_rng = float(sq.get("row_range", 0))
            sq_snr = float(sq.get("snr_db", 0))
            sq_detectable = bool(sq.get("detectable", False))
            sq_warning = str(sq.get("warning", ""))
            st.markdown("---")
            st.caption("信号质量诊断")
            col_a, col_b = st.columns(2)
            col_a.metric("行std", f"{sq_row_std:.2f}")
            col_b.metric("行range", f"{sq_row_rng:.1f}")
            if sq_detectable:
                st.success(f"信号可检测  SNR≈{sq_snr:.1f} dB")
            else:
                st.error(f"信号极弱或不可检测  SNR≈{sq_snr:.1f} dB")
            if sq_warning:
                st.warning(sq_warning)

        if metrics.get("ber") is not None:
            st.success(
                f"BER={float(metrics['ber']):.4f}  错误位={metrics.get('errors')} / {metrics.get('n_bits')}"
            )
        elif metrics.get("ber_error"):
            st.warning(f"BER 分析失败: {metrics['ber_error']}")
        else:
            st.info("尚未进行 BER 分析。")

        decoded_bits = metrics.get("decoded_bits", [])
        truth_bits = metrics.get("truth_bits", [])
        if decoded_bits:
            st.markdown("---")
            st.caption("解调出的前 32 bit")
            st.code("".join(str(int(b)) for b in decoded_bits[:32]))
        if truth_bits:
            st.caption("真值前 32 bit")
            st.code("".join(str(int(b)) for b in truth_bits[:32]))

        st.markdown("---")
        st.subheader("抓拍记录")
        if st.session_state.last_snapshot:
            st.code(st.session_state.last_snapshot)
        else:
            st.caption("尚未抓拍。")

    st.subheader("历史结果")
    records = st.session_state.records
    if records:
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["亮度", "BER", "GainLimit"])
        with chart_tab1:
            st.line_chart([r["brightness"] for r in records])
        with chart_tab2:
            ber_values = [0.0 if r["ber"] is None else r["ber"] for r in records]
            st.line_chart(ber_values)
        with chart_tab3:
            st.line_chart([r["gain_limit"] for r in records])
        st.dataframe(records[-20:], width="stretch")
    else:
        st.caption("历史记录为空。")

    st.subheader("算法建议")
    if brightness > 0:
        suggestions = compute_algo_suggestions(st.session_state.current_params, brightness, alpha, target_gray)
        cols = st.columns(len(suggestions))
        for col, (name, suggestion) in zip(cols, suggestions.items()):
            with col:
                st.markdown(f"**{name}**")
                if "error" in suggestion:
                    st.error(suggestion["error"])
                    continue
                suggested_gain_limit = map_db_to_gain_limit(float(suggestion["gain_db"]), calibration_records)
                st.write(f"建议增益(算法): `{suggestion['gain_db']:+.2f} dB`")
                st.write(f"建议 GainLimit: `{suggested_gain_limit}`")
                st.write(f"建议 ISO(算法参考): `{suggestion['iso']:.1f}`")
                if st.button(f"应用 {name}", key=f"apply_{name}", width="stretch"):
                    success = hik_camera.set_gain_limit(int(suggested_gain_limit))
                    push_message(f"应用 {name} 成功。" if success else f"应用 {name} 失败。")
                    if success:
                        st.session_state.current_params = CameraParams(
                            iso=float(suggestion["iso"]),
                            exposure_us=float(suggestion["exposure_us"]),
                        )
                    st.rerun()
    else:
        st.caption("获取到稳定 ROI 后会显示算法建议。")

    st.subheader("GainLimit 标定")
    cal_col1, cal_col2, cal_col3 = st.columns(3)
    calibration_points = cal_col1.text_input("标定档位", value="0,10,20,30,40,50,60,70,80,90,100")
    calibration_frames = int(cal_col2.number_input("每档帧数", min_value=3, max_value=30, value=8, step=1))
    run_calibration_btn = cal_col3.button("开始标定", width="stretch")

    if run_calibration_btn:
        try:
            limits = [int(x.strip()) for x in calibration_points.split(",") if x.strip()]
            st.session_state.calibration_records = run_gainlimit_calibration(
                hik_camera,
                rtsp_url,
                tx_label_csv,
                roi_strategy=roi_strategy,
                roi_size=roi_size,
                limits=limits,
                n_frames=calibration_frames,
            )
            push_message("GainLimit 标定完成。")
            st.rerun()
        except Exception as exc:
            push_message(f"GainLimit 标定失败: {exc}")
            st.rerun()

    calibration_records = st.session_state.calibration_records
    if calibration_records:
        st.dataframe(calibration_records, width="stretch")

    st.subheader("10 步实验")
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    experiment_steps = int(exp_col1.number_input("步数", min_value=1, max_value=20, value=10, step=1))
    experiment_frames = int(exp_col2.number_input("每步帧数", min_value=5, max_value=50, value=20, step=1))
    run_experiment_btn = exp_col3.button("运行实验", width="stretch")

    if run_experiment_btn:
        try:
            st.session_state.experiment_records = run_gainlimit_experiment(
                hik_camera,
                rtsp_url,
                tx_label_csv,
                algo_name=selected_algo,
                calibration_records=calibration_records,
                steps=experiment_steps,
                frames_per_step=experiment_frames,
                roi_strategy=roi_strategy,
                roi_size=roi_size,
                alpha=alpha,
                target_gray=target_gray,
            )
            push_message("10 步 GainLimit 实验完成。")
            st.rerun()
        except Exception as exc:
            push_message(f"实验执行失败: {exc}")
            st.rerun()

    if st.session_state.experiment_records:
        st.dataframe(st.session_state.experiment_records, width="stretch")
        exp_tabs = st.tabs(["步骤亮度", "步骤 BER", "步骤 GainLimit"])
        with exp_tabs[0]:
            st.line_chart([r["brightness_median"] for r in st.session_state.experiment_records])
        with exp_tabs[1]:
            st.line_chart([0.0 if r["ber_median"] is None else r["ber_median"] for r in st.session_state.experiment_records])
        with exp_tabs[2]:
            st.line_chart([r["applied_gain_limit"] for r in st.session_state.experiment_records])

    with st.expander("使用说明"):
        st.markdown(
            """
            - `连接相机` 后会持续读取 RTSP 最新帧。
            - `ROI 策略` 支持 `sync_based / auto / center`，推荐先用 `auto` 找亮区。
            - `抓拍保存` 会同时保存原始帧、ROI 裁剪和对应指标 JSON。
            - `BER` 基于发射序列 CSV 单帧解调；若失败，会显示失败原因。
            - `单步算法应用` 会按当前 ROI 亮度计算下一组建议参数，并直接写入相机增益。
            """
        )

    if auto_refresh and st.session_state.connected:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
