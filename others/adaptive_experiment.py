#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCC 自适应参数实验控制器 (三算法对比)
=====================================

流程:
  Round 0: 初始参数 -> 采集N张 -> 解调 -> BER_0  (三算法共享)
  Round k: 算法1推荐参数 -> 采集N张 -> BER_k_A1
           算法2推荐参数 -> 采集N张 -> BER_k_A2
           算法3推荐参数 -> 采集N张 -> BER_k_A3
  输出: 三算法 BER 收敛曲线对比图 + 状态 JSON

算法:
  1. Matus 单次公式   G_opt = G_curr + 20*log10(Y_target/Y_curr)  [仅调ISO]
  2. Matus 自适应 α   G_next = G_curr + α*20*log10(Y_target/Y_curr) [仅调ISO]
  3. Ma 自适应阻尼    完整状态机 I-V，可调曝光+ISO

用法示例:
  # RTSP 自动采集模式
  python others/adaptive_experiment.py \\
      --rtsp-url rtsp://admin:abcd1234@192.168.1.19/Streaming/Channels/1 \\
      --initial-iso 35 --initial-exposure 27.9 \\
      --n-frames 50 --max-rounds 5 \\
      --label-csv results/base_data/Mseq_32_original.csv \\
      --save-dir exp_data/session_001

  # 手动上传图像模式（无相机时）
  python others/adaptive_experiment.py \\
      --rtsp-url none \\
      --initial-iso 35 --initial-exposure 27.9 \\
      --save-dir exp_data/session_001

相机控制模式 (--camera-mode):
  manual      每轮提示用户手动调整相机参数 (默认)
  hikvision   通过 HIKVISION ISAPI HTTP 接口自动设置 (需先确认字段名)
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# ── 将 src/ 加入路径 ──────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_ROOT))


# =============================================================================
# 字体设置
# =============================================================================

def _setup_fonts() -> None:
    system = platform.system()
    candidates = {
        "Darwin": ["Hiragino Sans GB", "Arial Unicode MS", "Arial"],
        "Windows": ["Microsoft YaHei", "SimHei", "Arial"],
    }.get(system, ["DejaVu Sans", "Arial"])
    for name in candidates:
        try:
            path = fm.findfont(fm.FontProperties(family=name))
            if path:
                matplotlib.rcParams["font.family"] = name
                matplotlib.rcParams["font.sans-serif"] = [name]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
        except Exception:
            continue


_setup_fonts()


# =============================================================================
# 单位换算
# =============================================================================

def iso_to_db(iso: float) -> float:
    return 20.0 * np.log10(max(iso / 100.0, 1e-9))


def db_to_iso(db: float) -> float:
    return 100.0 * 10.0 ** (db / 20.0)


def fmt_exp(s: float) -> str:
    if s >= 1.0:
        return f"{s:.4f}s"
    if s >= 1e-3:
        return f"{s * 1e3:.3f}ms"
    return f"{s * 1e6:.2f}us"


# =============================================================================
# 相机采集（基于 comm_capture.py 逻辑）
# =============================================================================

class _ThreadedCamera:
    def __init__(self, source: str):
        self.capture = cv2.VideoCapture(source)
        self.running = True
        self.status = False
        self.frame = None
        self._thread = Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self) -> None:
        while self.running:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(0.01)

    def grab_frame(self):
        return self.frame if self.status else None

    def stop(self) -> None:
        self.running = False
        self._thread.join(timeout=3)
        self.capture.release()


class _ImageWriterThread(Thread):
    def __init__(self, queue: Queue):
        super().__init__(daemon=True)
        self.queue = queue
        self.start()

    def run(self) -> None:
        while True:
            filepath, frame = self.queue.get()
            if filepath is None:
                break
            cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            self.queue.task_done()


def collect_frames(rtsp_url: Optional[str], save_dir: str, n_frames: int) -> List[str]:
    """
    采集 n_frames 张图像保存到 save_dir，返回文件路径列表。
    rtsp_url 为 None 或 'none' 时进入手动上传模式。
    """
    os.makedirs(save_dir, exist_ok=True)

    if rtsp_url is None or rtsp_url.lower() == "none":
        print(f"\n  [手动采集] 请将 {n_frames} 张图像放入目录:")
        print(f"  {save_dir}")
        print("  完成后按 Enter 继续...", end="", flush=True)
        input()
        paths = sorted(
            [str(p) for p in Path(save_dir).glob("*.jpg")]
            + [str(p) for p in Path(save_dir).glob("*.jpeg")]
            + [str(p) for p in Path(save_dir).glob("*.png")]
        )
        print(f"  检测到 {len(paths)} 张图像")
        return paths

    # RTSP 采集
    print(f"\n  [RTSP] 正在连接相机: {rtsp_url} ...")
    streamer = _ThreadedCamera(rtsp_url)
    time.sleep(2)

    write_queue: Queue = Queue(maxsize=n_frames + 200)
    writer = _ImageWriterThread(write_queue)

    paths: List[str] = []
    count = 0
    print(f"  开始采集 {n_frames} 张...", flush=True)
    while count < n_frames:
        frame = streamer.grab_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        filename = os.path.join(save_dir, f"frame_{count + 1:05d}.jpg")
        write_queue.put((filename, frame))
        paths.append(filename)
        count += 1
        if count % 10 == 0:
            print(f"    {count}/{n_frames}", flush=True)

    streamer.stop()
    write_queue.put((None, None))
    writer.join(timeout=30)
    print(f"  采集完成，共 {count} 张")
    return sorted(paths)


# =============================================================================
# OOK 解调 —— 直接调用 src/occ_gain_opt/demodulation.py 的 OOKDemodulator
#
# 相比简单的行均值+三阶多项式方法，OOKDemodulator 有三项关键优势：
#   1. 先用 Otsu 分割检测 LED 所在列范围，只对 LED 列做行均值（过滤背景）
#   2. 使用 green 通道（LED 信号更强）+ 高斯平滑（抑制行噪声）
#   3. Otsu 二值化（鲁棒）+ 精确位周期对齐采样（比边沿计数更准确）
# =============================================================================

from occ_gain_opt.demodulation import OOKDemodulator as _OOKDemodulator

_demodulator = _OOKDemodulator()


def demodulate_image(image_path: str,
                     tx_bits: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    解调单张图像，返回 (ber, roi_brightness)，失败返回 None。

    使用 OOKDemodulator：
      - LED 列 ROI（Otsu 自动检测列范围）
      - green 通道 + 高斯平滑
      - Otsu 二值化
      - 精确位周期对齐采样
    roi_brightness 用 OOKDemodulator 生成的 roi_mask 区域灰度均值，
    精确反映 LED 信号区域亮度，用于算法参数更新。
    """
    try:
        # OOKDemodulator 需要 BGR 格式（cv2 默认）
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            # 回退：PIL 读取再转换
            img_bgr = cv2.cvtColor(
                np.array(Image.open(image_path).convert("RGB")),
                cv2.COLOR_RGB2BGR
            )

        result = _demodulator.demodulate(img_bgr)

        # 优先用完整数据包；若没有（图像只含一个同步头），
        # 回退到从比特序列中同步头位置之后手动提取
        if result.packets:
            packet = result.packets[0]
        elif result.sync_positions_bit:
            sync_pos = result.sync_positions_bit[0]
            sync_len = (len(result.sync_pattern)
                        if result.sync_pattern is not None else 8)
            data_start = sync_pos + sync_len
            packet = result.bit_sequence[data_start: data_start + len(tx_bits)]
        else:
            return None

        n = min(len(tx_bits), len(packet))
        if n == 0:
            return None
        ber = float(np.sum(tx_bits[:n] != packet[:n])) / n

        # 亮度用 ROI mask 区域的灰度均值（LED 信号精确区域）
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        roi_pixels = gray[result.roi_mask == 1]
        roi_brightness = (float(np.mean(roi_pixels))
                          if len(roi_pixels) > 0 else float(np.mean(gray)))

        return ber, roi_brightness
    except Exception:
        return None


def batch_demodulate(image_paths: List[str],
                     tx_bits: np.ndarray) -> Dict[str, Any]:
    """
    批量解调，返回成功帧的平均 BER 和平均亮度。
    """
    bers: List[float] = []
    brightnesses: List[float] = []
    for p in image_paths:
        result = demodulate_image(p, tx_bits)
        if result is not None:
            bers.append(result[0])
            brightnesses.append(result[1])
    n_success = len(bers)
    return {
        "mean_ber": float(np.mean(bers)) if bers else None,
        "mean_brightness": float(np.mean(brightnesses)) if brightnesses else None,
        "n_success": n_success,
        "n_total": len(image_paths),
        "success_rate": n_success / len(image_paths) if image_paths else 0.0,
    }


# =============================================================================
# 算法 1: Matus 单次公式
# =============================================================================

class MatusSingleAlgo:
    """G_opt = G_curr + 20*log10(Y_target / Y_curr)，仅调整 ISO。"""

    def __init__(self, initial_iso: float, initial_exp_s: float,
                 target_gray: float = 242.25,
                 iso_min: float = 30.0, iso_max: float = 10000.0):
        self.current_iso = initial_iso
        self.current_exp_s = initial_exp_s
        self.target_gray = target_gray
        self._db_min = iso_to_db(iso_min)
        self._db_max = iso_to_db(iso_max)
        self.iso_history: List[float] = [initial_iso]
        self.exp_history: List[float] = [initial_exp_s]
        self.ber_history: List[float] = []
        self.brightness_history: List[float] = []

    def update(self, mean_brightness: float, ber: float) -> Tuple[float, float]:
        """用当前亮度+BER更新，返回 (next_iso, next_exp_s)。"""
        self.ber_history.append(ber)
        self.brightness_history.append(mean_brightness)
        if mean_brightness > 0:
            g_db = iso_to_db(self.current_iso)
            g_opt = float(np.clip(
                g_db + 20.0 * np.log10(self.target_gray / mean_brightness),
                self._db_min, self._db_max
            ))
            self.current_iso = db_to_iso(g_opt)
        self.iso_history.append(self.current_iso)
        self.exp_history.append(self.current_exp_s)
        return self.current_iso, self.current_exp_s

    def get_state(self) -> Dict:
        return dict(
            current_iso=self.current_iso,
            current_exp_s=self.current_exp_s,
            iso_history=self.iso_history,
            exp_history=self.exp_history,
            ber_history=self.ber_history,
            brightness_history=self.brightness_history,
        )

    def load_state(self, s: Dict) -> None:
        self.current_iso = s["current_iso"]
        self.current_exp_s = s["current_exp_s"]
        self.iso_history = s["iso_history"]
        self.exp_history = s["exp_history"]
        self.ber_history = s["ber_history"]
        self.brightness_history = s["brightness_history"]


# =============================================================================
# 算法 2: Matus 自适应 (α)
# =============================================================================

class MatusAdaptiveAlgo:
    """G_next = G_curr + α * 20*log10(Y_target / Y_curr)，仅调整 ISO。"""

    def __init__(self, initial_iso: float, initial_exp_s: float,
                 alpha: float = 0.5, target_gray: float = 242.25,
                 step_max_db: float = 5.0,
                 iso_min: float = 30.0, iso_max: float = 10000.0):
        self.current_iso = initial_iso
        self.current_exp_s = initial_exp_s
        self.alpha = alpha
        self.target_gray = target_gray
        self.step_max_db = step_max_db
        self._db_min = iso_to_db(iso_min)
        self._db_max = iso_to_db(iso_max)
        self.iso_history: List[float] = [initial_iso]
        self.exp_history: List[float] = [initial_exp_s]
        self.ber_history: List[float] = []
        self.brightness_history: List[float] = []

    def update(self, mean_brightness: float, ber: float) -> Tuple[float, float]:
        self.ber_history.append(ber)
        self.brightness_history.append(mean_brightness)
        if mean_brightness > 0:
            g_db = iso_to_db(self.current_iso)
            delta = float(np.clip(
                20.0 * np.log10(self.target_gray / mean_brightness),
                -self.step_max_db, self.step_max_db
            ))
            g_next = float(np.clip(g_db + self.alpha * delta, self._db_min, self._db_max))
            self.current_iso = db_to_iso(g_next)
        self.iso_history.append(self.current_iso)
        self.exp_history.append(self.current_exp_s)
        return self.current_iso, self.current_exp_s

    def get_state(self) -> Dict:
        return dict(
            current_iso=self.current_iso,
            current_exp_s=self.current_exp_s,
            alpha=self.alpha,
            iso_history=self.iso_history,
            exp_history=self.exp_history,
            ber_history=self.ber_history,
            brightness_history=self.brightness_history,
        )

    def load_state(self, s: Dict) -> None:
        self.current_iso = s["current_iso"]
        self.current_exp_s = s["current_exp_s"]
        self.alpha = s.get("alpha", self.alpha)
        self.iso_history = s["iso_history"]
        self.exp_history = s["exp_history"]
        self.ber_history = s["ber_history"]
        self.brightness_history = s["brightness_history"]


# =============================================================================
# 算法 3: Ma 自适应阻尼 (完整状态机 I-V)
# =============================================================================

class MaDampingAlgo:
    """
    完整状态机:
      I   初始化规整
      II  比例法快速单向收敛
      III 局部线性拟合收敛
      IV  增益单向收敛
      V   增益夹逼收敛
    BER 由外部真实解调提供，不再模拟。
    """
    _ISO_BASE = 100.0

    def __init__(self, initial_iso: float, initial_exp_s: float,
                 target_brightness: float = 125.0,
                 brightness_tolerance: float = 5.0,
                 exposure_strategy: str = "exposure_priority",
                 exp_min_s: float = 1e-6, exp_max_s: float = 1e-3,
                 iso_min: float = 30.0, iso_max: float = 10000.0):
        self.target_br = target_brightness
        self.tolerance = brightness_tolerance
        self.exp_strategy = exposure_strategy
        self.exp_min = exp_min_s
        self.exp_max = exp_max_s
        self.gain_min = iso_min / self._ISO_BASE
        self.gain_max = iso_max / self._ISO_BASE

        self.current_exp = initial_exp_s
        self.current_gain = initial_iso / self._ISO_BASE
        self.current_l: Optional[float] = None

        # 状态机
        self.state = "I"
        self.references: Dict[str, Dict] = {
            s: {"L": None, "E": None, "G": None} for s in ["I", "II", "III", "IV", "V"]
        }
        self.state_ii_cnt = 0
        self.state_iii_cnt = 0
        self.state_iv_cnt = 0

        # 历史
        self.iso_history: List[float] = [initial_iso]
        self.exp_history: List[float] = [initial_exp_s]
        self.ber_history: List[float] = []
        self.brightness_history: List[float] = []
        self.damping_history: List[float] = []
        self.state_history: List[str] = ["I"]

    @property
    def current_iso(self) -> float:
        return self.current_gain * self._ISO_BASE

    @property
    def current_exp_s(self) -> float:
        return self.current_exp

    # ── 算法1：比例法 ────────────────────────────────────────────────────────
    def _proportional(self, use_exp: bool) -> Tuple[float, float]:
        r = float(np.clip(self.target_br / max(self.current_l, 0.1), 0.1, 10.0))
        if use_exp:
            e = float(np.clip(r * self.current_exp, self.exp_min, self.exp_max))
            return e, self.current_gain
        else:
            g = float(np.clip(r * self.current_gain, self.gain_min, self.gain_max))
            return self.current_exp, g

    # ── 算法2：局部线性拟合 ──────────────────────────────────────────────────
    def _linear_fit(self, ref1: Dict, ref2: Dict,
                    adjust_exp: bool) -> Tuple[float, float]:
        l1, e1, g1 = ref1["L"], ref1["E"], ref1["G"]
        l2, e2, g2 = ref2["L"], ref2["E"], ref2["G"]
        if None in (l1, e1, g1, l2, e2, g2) or abs(l2 - l1) < 1e-6:
            return self._proportional(adjust_exp)
        if adjust_exp:
            e_new = e1 + (e2 - e1) * (self.target_br - l1) / (l2 - l1)
            return float(np.clip(e_new, self.exp_min, self.exp_max)), g1
        else:
            g_new = g1 + (g2 - g1) * (self.target_br - l1) / (l2 - l1)
            return e1, float(np.clip(g_new, self.gain_min, self.gain_max))

    # ── 自适应阻尼 ───────────────────────────────────────────────────────────
    def _calc_damping(self, l_prev: float) -> float:
        if l_prev < 1.0 or self.current_l < 1.0:
            return 0.3
        lam = min(l_prev / self.current_l, self.current_l / l_prev)
        d_base = 1.0 - lam
        err = abs(self.current_l - self.target_br)
        factor = 0.5 if err > 50 else (0.7 if err > 20 else (0.9 if err > 10 else 1.0))
        return float(np.clip(d_base * factor, 0.0, 0.8))

    def _apply_damping(self, e_old: float, g_old: float,
                       e_new: float, g_new: float, d: float) -> Tuple[float, float]:
        e = float(np.clip(d * e_old + (1 - d) * e_new, self.exp_min, self.exp_max))
        g = float(np.clip(d * g_old + (1 - d) * g_new, self.gain_min, self.gain_max))
        return e, g

    # ── 各状态执行 ───────────────────────────────────────────────────────────
    def _execute_state(self) -> Tuple[float, float]:
        use_exp = self.exp_strategy in ("exposure_priority", "exposure_only")

        if self.state == "I":
            if use_exp:
                g_new = self.gain_min
                e_new = ((self.current_gain / self.gain_min) * self.current_exp
                         if self.current_gain > self.gain_min
                         else self.current_exp * 1.5)
            else:
                e_new = self.exp_min
                g_new = ((self.current_exp / self.exp_min) * self.current_gain
                         if self.current_exp > self.exp_min
                         else self.current_gain * 1.5)
            return (float(np.clip(e_new, self.exp_min, self.exp_max)),
                    float(np.clip(g_new, self.gain_min, self.gain_max)))

        if self.state == "II":
            return self._proportional(use_exp)

        if self.state == "III":
            r1, r2 = self.references["I"], self.references["II"]
            if (r1["L"] is None or r2["L"] is None or
                    (r1["L"] - self.target_br) * (r2["L"] - self.target_br) > 0):
                return self._proportional(use_exp)
            return self._linear_fit(r1, r2, use_exp)

        if self.state == "IV":
            return self._proportional(False)   # 增益收敛阶段

        if self.state == "V":
            r3, r4 = self.references["III"], self.references["IV"]
            if (r3["L"] is None or r4["L"] is None or
                    (r3["L"] - self.target_br) * (r4["L"] - self.target_br) > 0):
                return self._proportional(False)
            return self._linear_fit(r3, r4, False)

        return self.current_exp, self.current_gain

    # ── 状态转移 ─────────────────────────────────────────────────────────────
    def _transition(self) -> None:
        cur = {"L": self.current_l, "E": self.current_exp, "G": self.current_gain}

        if self.state == "I":
            self.references["I"] = cur
            self.state = "II"

        elif self.state == "II":
            self.state_ii_cnt += 1
            l_ref = self.references["I"]["L"]
            self.references["II"] = cur
            if l_ref is not None:
                crossed = ((l_ref < self.target_br < self.current_l) or
                           (self.current_l < self.target_br < l_ref))
                if crossed or self.state_ii_cnt >= 8:
                    self.state = "III"

        elif self.state == "III":
            self.state_iii_cnt += 1
            self.references["III"] = cur
            if self.state_iii_cnt >= 5:
                self.state = "IV"

        elif self.state == "IV":
            self.state_iv_cnt += 1
            l_ref = self.references["III"]["L"]
            self.references["IV"] = cur
            if l_ref is not None:
                crossed = ((l_ref < self.target_br < self.current_l) or
                           (self.current_l < self.target_br < l_ref))
                if crossed or self.state_iv_cnt >= 8:
                    self.state = "V"

        elif self.state == "V":
            self.references["V"] = cur

    # ── 公开接口 ─────────────────────────────────────────────────────────────
    def update(self, mean_brightness: float, ber: float) -> Tuple[float, float]:
        """输入真实亮度和 BER，返回 (next_iso, next_exp_s)。"""
        self.ber_history.append(ber)
        self.brightness_history.append(mean_brightness)

        l_prev = self.current_l
        self.current_l = mean_brightness

        e_new, g_new = self._execute_state()

        # 应用阻尼（至少有一次历史记录才有意义）
        if l_prev is not None:
            d = self._calc_damping(l_prev)
            e_act, g_act = self._apply_damping(
                self.current_exp, self.current_gain, e_new, g_new, d
            )
        else:
            d = 0.0
            e_act, g_act = e_new, g_new

        self._transition()
        self.current_exp = e_act
        self.current_gain = g_act

        self.iso_history.append(self.current_iso)
        self.exp_history.append(self.current_exp)
        self.damping_history.append(d)
        self.state_history.append(self.state)

        return self.current_iso, self.current_exp

    def get_state(self) -> Dict:
        return dict(
            current_gain=self.current_gain,
            current_exp_s=self.current_exp,
            current_l=self.current_l,
            state=self.state,
            references=self.references,
            state_ii_cnt=self.state_ii_cnt,
            state_iii_cnt=self.state_iii_cnt,
            state_iv_cnt=self.state_iv_cnt,
            iso_history=self.iso_history,
            exp_history=self.exp_history,
            ber_history=self.ber_history,
            brightness_history=self.brightness_history,
            damping_history=self.damping_history,
            state_history=self.state_history,
        )

    def load_state(self, s: Dict) -> None:
        self.current_gain = s["current_gain"]
        self.current_exp = s["current_exp_s"]
        self.current_l = s.get("current_l")
        self.state = s["state"]
        self.references = s["references"]
        self.state_ii_cnt = s["state_ii_cnt"]
        self.state_iii_cnt = s["state_iii_cnt"]
        self.state_iv_cnt = s["state_iv_cnt"]
        self.iso_history = s["iso_history"]
        self.exp_history = s["exp_history"]
        self.ber_history = s["ber_history"]
        self.brightness_history = s["brightness_history"]
        self.damping_history = s.get("damping_history", [])
        self.state_history = s.get("state_history", [])


# =============================================================================
# 相机控制接口
# =============================================================================

class CameraController:
    """
    手动模式：打印提示，等待用户调好相机后按 Enter。
    hikvision 模式：通过 ISAPI HTTP 接口自动设置（TODO，需确认字段名）。
    """

    def __init__(self, mode: str = "manual",
                 camera_ip: str = "192.168.1.19",
                 user: str = "admin", password: str = "abcd1234"):
        self.mode = mode
        self.camera_ip = camera_ip
        self.user = user
        self.password = password

    def set_params(self, iso: float, exp_us: float) -> bool:
        """
        设置相机参数。
        返回 True 表示继续实验，False 表示用户选择退出。
        """
        if self.mode == "hikvision":
            return self._hikvision_set(iso, exp_us)
        return self._manual_prompt(iso, exp_us)

    def _manual_prompt(self, iso: float, exp_us: float) -> bool:
        print(f"\n  {'─' * 52}")
        print(f"  请将相机设置为:")
        print(f"    ISO      : {iso:.0f}")
        print(f"    曝光时间 : {exp_us:.2f} us  ({fmt_exp(exp_us * 1e-6)})")
        print(f"  设置完成后按 Enter 继续，输入 'q' 退出: ", end="", flush=True)
        resp = input().strip().lower()
        return resp != "q"

    def _hikvision_set(self, iso: float, exp_us: float) -> bool:
        """
        HIKVISION ISAPI 自动控制。
        使用前需先 GET /ISAPI/Image/channels/1/imagingSettings
        确认实际 XML 字段名，然后在此处填写。
        """
        # TODO: 根据实际字段名完善以下实现
        # import requests
        # from requests.auth import HTTPDigestAuth
        # url = f"http://{self.camera_ip}/ISAPI/Image/channels/1/imagingSettings"
        # xml = f"""<ImagingData>
        #     <exposureTime>{int(exp_us)}</exposureTime>
        #     <gain>{iso}</gain>
        # </ImagingData>"""
        # resp = requests.put(url, data=xml,
        #                     auth=HTTPDigestAuth(self.user, self.password),
        #                     timeout=5)
        # if resp.status_code == 200:
        #     print(f"  [ISAPI] 参数已设置: ISO={iso:.0f}, 曝光={exp_us:.2f}us")
        #     return True
        # print(f"  [ISAPI] 设置失败 ({resp.status_code})，回退手动模式")
        print("  [ISAPI] 尚未实现，回退手动模式")
        return self._manual_prompt(iso, exp_us)


# =============================================================================
# 实时可视化
# =============================================================================

_ALGO_META = {
    "matus_single":   ("#2196F3", "Matus 单次"),
    "matus_adaptive": ("#4CAF50", f"Matus 自适应"),
    "ma_damping":     ("#FF5722", "Ma 阻尼"),
}


class RealtimeVisualizer:
    def __init__(self, save_dir: str, alpha: float = 0.5):
        self.save_dir = save_dir
        self.alpha = alpha
        plt.ion()
        self.fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        self.fig.suptitle("OCC 自适应参数实验 — 三算法对比", fontsize=14, fontweight="bold")
        self.ax_ber    = axes[0, 0]
        self.ax_bright = axes[0, 1]
        self.ax_iso    = axes[1, 0]
        self.ax_exp    = axes[1, 1]
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    def update(self, algorithms: Dict[str, Any]) -> None:
        """每轮结束后调用，刷新四张子图。"""
        for ax in (self.ax_ber, self.ax_bright, self.ax_iso, self.ax_exp):
            ax.cla()

        for name, algo in algorithms.items():
            color, label = _ALGO_META[name]
            ber_hist = algo.ber_history
            bright_hist = algo.brightness_history
            iso_hist = algo.iso_history
            exp_hist = [e * 1e6 for e in algo.exp_history]   # s → µs
            rounds = list(range(len(ber_hist)))

            if ber_hist:
                self.ax_ber.plot(rounds, ber_hist, "o-", color=color,
                                 label=label, markersize=5, linewidth=1.5)
            if bright_hist:
                self.ax_bright.plot(list(range(len(bright_hist))), bright_hist,
                                    "o-", color=color, label=label,
                                    markersize=5, linewidth=1.5)
            if iso_hist:
                self.ax_iso.plot(iso_hist, "s--", color=color,
                                 label=label, markersize=4)
            # 只有曝光时间真正变化时才绘制（Matus 两算法曝光恒定）
            if exp_hist and max(exp_hist) - min(exp_hist) > 0.01:
                self.ax_exp.plot(exp_hist, "s--", color=color,
                                 label=label, markersize=4)

        # BER 图
        self.ax_ber.set_title("误码率 BER vs 轮次")
        self.ax_ber.set_xlabel("轮次")
        self.ax_ber.set_ylabel("BER")
        self.ax_ber.legend()
        self.ax_ber.grid(True, alpha=0.3)
        self.ax_ber.set_ylim(bottom=0)

        # 亮度图
        self.ax_bright.set_title("ROI 平均亮度 vs 轮次")
        self.ax_bright.set_xlabel("轮次")
        self.ax_bright.set_ylabel("灰度均值 (0-255)")
        self.ax_bright.axhline(125,    color="salmon",    linestyle=":", alpha=0.7,
                               label="Ma 目标 (125)")
        self.ax_bright.axhline(242.25, color="steelblue", linestyle=":", alpha=0.7,
                               label="Matus 目标 (242)")
        self.ax_bright.legend(fontsize=8)
        self.ax_bright.grid(True, alpha=0.3)

        # ISO 图
        self.ax_iso.set_title("ISO 调整历史")
        self.ax_iso.set_xlabel("迭代步")
        self.ax_iso.set_ylabel("ISO")
        self.ax_iso.legend()
        self.ax_iso.grid(True, alpha=0.3)

        # 曝光图（仅 Ma 算法有效变化时有内容）
        self.ax_exp.set_title("曝光时间调整历史")
        self.ax_exp.set_xlabel("迭代步")
        self.ax_exp.set_ylabel("曝光 (us)")
        self.ax_exp.legend()
        self.ax_exp.grid(True, alpha=0.3)

        # 右上角标注 alpha
        self.fig.text(0.99, 0.01,
                      f"Matus-adaptive alpha={self.alpha}",
                      ha="right", va="bottom", fontsize=9, color="gray")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.pause(0.5)

    def save(self) -> None:
        path = os.path.join(self.save_dir, "final_comparison.png")
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n  图表已保存: {path}")
        plt.ioff()
        plt.show(block=False)


# =============================================================================
# 状态持久化
# =============================================================================

class ExperimentState:
    def __init__(self, save_dir: str):
        self.path = os.path.join(save_dir, "state.json")

    def save(self, data: Dict) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> Optional[Dict]:
        if not os.path.exists(self.path):
            return None
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)


# =============================================================================
# 主实验控制器
# =============================================================================

_ALGO_ORDER = ["matus_single", "matus_adaptive", "ma_damping"]


class AdaptiveExperiment:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 加载标签序列
        df = pd.read_csv(args.label_csv, skiprows=5)
        self.tx_bits: np.ndarray = df.iloc[:, 1].to_numpy()
        print(f"  标签加载成功，比特数 = {len(self.tx_bits)}")

        # 初始化三个算法
        self.algorithms: Dict[str, Any] = {
            "matus_single": MatusSingleAlgo(
                args.initial_iso, args.initial_exposure * 1e-6,
                iso_min=args.iso_min, iso_max=args.iso_max,
            ),
            "matus_adaptive": MatusAdaptiveAlgo(
                args.initial_iso, args.initial_exposure * 1e-6,
                alpha=args.alpha,
                iso_min=args.iso_min, iso_max=args.iso_max,
            ),
            "ma_damping": MaDampingAlgo(
                args.initial_iso, args.initial_exposure * 1e-6,
                target_brightness=args.target_brightness,
                exposure_strategy=args.ma_strategy,
                exp_min_s=args.exp_min_us * 1e-6,
                exp_max_s=args.exp_max_us * 1e-6,
                iso_min=args.iso_min, iso_max=args.iso_max,
            ),
        }

        self.camera = CameraController(
            mode=args.camera_mode,
            camera_ip=args.camera_ip,
        )
        self.state_store = ExperimentState(str(self.save_dir))
        self.visualizer = RealtimeVisualizer(str(self.save_dir), alpha=args.alpha)
        self.current_round = 0

    # ── 采集+解调 ────────────────────────────────────────────────────────────
    def _run_one_slot(self, tag: str,
                      iso: float, exp_us: float) -> Optional[Dict[str, Any]]:
        """
        设置参数 -> 采集 N 帧 -> 批量解调。
        tag 用于命名保存目录（如 'init'、'matus_single'）。
        返回解调结果字典，用户选择退出时返回 None。
        """
        ok = self.camera.set_params(iso, exp_us)
        if not ok:
            return None

        folder = self.save_dir / f"round_{self.current_round:02d}_{tag}"
        paths = collect_frames(
            rtsp_url=self.args.rtsp_url if self.args.rtsp_url.lower() != "none" else None,
            save_dir=str(folder),
            n_frames=self.args.n_frames,
        )
        if not paths:
            print("  未获取到图像，跳过本轮本算法")
            return None

        result = batch_demodulate(paths, self.tx_bits)
        sr = result["success_rate"] * 100
        ber_str = f"{result['mean_ber']:.4f}" if result["mean_ber"] is not None else "N/A"
        bright_str = (f"{result['mean_brightness']:.1f}"
                      if result["mean_brightness"] is not None else "N/A")
        print(f"    解调: {result['n_success']}/{result['n_total']} 帧成功 ({sr:.0f}%)"
              f"  |  BER={ber_str}  |  亮度={bright_str}")
        return result

    # ── 状态保存/恢复 ────────────────────────────────────────────────────────
    def _save(self) -> None:
        self.state_store.save({
            "current_round": self.current_round,
            "args": vars(self.args),
            "algorithms": {n: a.get_state() for n, a in self.algorithms.items()},
        })

    def _try_resume(self) -> bool:
        data = self.state_store.load()
        if data is None:
            return False
        print(f"\n  发现中断状态 (Round {data['current_round']})，是否恢复？[y/n]: ",
              end="", flush=True)
        if input().strip().lower() != "y":
            return False
        self.current_round = data["current_round"]
        for name, state in data["algorithms"].items():
            self.algorithms[name].load_state(state)
        print(f"  已恢复至 Round {self.current_round}")
        return True

    # ── 主流程 ───────────────────────────────────────────────────────────────
    def run(self) -> None:
        print(f"\n{'=' * 60}")
        print("  OCC 自适应参数实验控制器  (三算法对比)")
        print(f"{'=' * 60}")
        print(f"  初始参数  : ISO={self.args.initial_iso}, "
              f"曝光={self.args.initial_exposure} us")
        print(f"  每轮帧数  : {self.args.n_frames}")
        print(f"  最大轮次  : {self.args.max_rounds}")
        print(f"  保存目录  : {self.save_dir}")
        print(f"  相机模式  : {self.args.camera_mode}")
        print(f"{'=' * 60}")

        # 尝试中断恢复
        resumed = self._try_resume()

        # ── Round 0：三算法共享初始采集 ─────────────────────────────────────
        if not resumed:
            print(f"\n{'─' * 60}")
            print(f"  Round 0  —  初始采集 (三算法共享)")
            result = self._run_one_slot(
                "init", self.args.initial_iso, self.args.initial_exposure
            )
            if result is None:
                print("  实验中止")
                return
            if result["mean_ber"] is not None:
                for algo in self.algorithms.values():
                    algo.update(result["mean_brightness"], result["mean_ber"])
            self.current_round = 1
            self._save()
            self.visualizer.update(self.algorithms)

        # ── Round 1..N：各算法独立采集 ──────────────────────────────────────
        for rnd in range(self.current_round, self.args.max_rounds + 1):
            self.current_round = rnd
            print(f"\n{'─' * 60}")
            print(f"  Round {rnd} / {self.args.max_rounds}")

            any_valid = False
            for name in _ALGO_ORDER:
                algo = self.algorithms[name]
                next_iso = algo.current_iso
                # current_exp_s 属性在三个算法上均存在
                next_exp_us = algo.current_exp_s * 1e6

                print(f"\n  [{_ALGO_META[name][1]}]"
                      f"  ISO={next_iso:.0f},  曝光={next_exp_us:.2f} us")

                result = self._run_one_slot(name, next_iso, next_exp_us)
                if result is None:
                    print("  实验中止")
                    self._save()
                    self.visualizer.save()
                    return

                if result["mean_ber"] is not None:
                    algo.update(result["mean_brightness"], result["mean_ber"])
                    any_valid = True

            self._save()
            if any_valid:
                self.visualizer.update(self.algorithms)

            # 本轮摘要
            print(f"\n  Round {rnd} 摘要:")
            for name, algo in self.algorithms.items():
                if algo.ber_history:
                    state_str = (f"  state={algo.state}"
                                 if hasattr(algo, "state") else "")
                    print(f"    {_ALGO_META[name][1]:18s}"
                          f"  BER={algo.ber_history[-1]:.4f}"
                          f"  ISO={algo.current_iso:.0f}"
                          f"  曝光={algo.current_exp_s * 1e6:.1f}us"
                          f"{state_str}")

        print(f"\n{'=' * 60}")
        print("  实验完成！")
        self._save()
        self.visualizer.save()


# =============================================================================
# CLI 入口
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OCC 自适应参数实验控制器 (三算法对比)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # 采集
    p.add_argument("--rtsp-url", default="none",
                   help="相机 RTSP 地址，填 'none' 则手动上传图像")
    p.add_argument("--n-frames", type=int, default=50,
                   help="每轮每算法采集帧数 (默认 50)")
    p.add_argument("--max-rounds", type=int, default=5,
                   help="最大迭代轮次 (默认 5)")
    # 初始参数
    p.add_argument("--initial-iso", type=float, default=35.0,
                   help="初始 ISO (默认 35)")
    p.add_argument("--initial-exposure", type=float, default=27.9,
                   help="初始曝光时间 us (默认 27.9)")
    # 数据
    p.add_argument("--label-csv",
                   default="results/base_data/Mseq_32_original.csv",
                   help="发射序列 CSV 路径")
    p.add_argument("--save-dir", default="exp_data/session_001",
                   help="结果保存目录")
    # 算法参数
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Matus 自适应学习率 alpha (默认 0.5)")
    p.add_argument("--target-brightness", type=float, default=125.0,
                   help="Ma 算法目标亮度 (默认 125)")
    p.add_argument("--ma-strategy", default="exposure_priority",
                   choices=["exposure_priority", "gain_priority"],
                   help="Ma 算法控制策略 (默认 exposure_priority)")
    # 参数范围
    p.add_argument("--iso-min", type=float, default=30.0)
    p.add_argument("--iso-max", type=float, default=10000.0)
    p.add_argument("--exp-min-us", type=float, default=1.0,
                   help="曝光最小值 us (默认 1)")
    p.add_argument("--exp-max-us", type=float, default=1000.0,
                   help="曝光最大值 us (默认 1000)")
    # 相机控制
    p.add_argument("--camera-mode", default="manual",
                   choices=["manual", "hikvision"],
                   help="相机控制模式 (默认 manual)")
    p.add_argument("--camera-ip", default="192.168.1.19",
                   help="相机 IP (hikvision 模式用)")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    exp = AdaptiveExperiment(args)
    exp.run()


if __name__ == "__main__":
    main()
