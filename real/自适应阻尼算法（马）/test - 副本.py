#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import platform

import cv2
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

# ========================= 用户可配置参数 =========================

# 输入/输出
INPUT_PATH = "C:/Users/A/Desktop/222"        # 输入路径（文件或文件夹）
OUTPUT_DIR = "results1"                          # 输出目录
BATCH_MODE = False                               # 强制批量模式
COMPARE = True                                   # 比较有/无阻尼

# 图像处理
ROI = None                                       # 感兴趣区域 (x,y,w,h)
EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

# 算法核心
TARGET_BRIGHTNESS = 125.0                        # 目标亮度
BRIGHTNESS_TOLERANCE = 5.0                       # 亮度容差
# 误码率目标已移除，算法将尽量降低误码率

# 曝光时间范围（支持分数，如"1/1000"）
EXPOSURE_MIN = 0.000001
EXPOSURE_MAX = 0.001

# 增益范围
GAIN_MIN = 0.30
GAIN_MAX = 100

# 初始参数（None则自动）
INITIAL_EXPOSURE = 0.0000279
INITIAL_ISO = 35

# 曝光策略: exposure_only, exposure_priority, gain_only, gain_priority
EXPOSURE_STRATEGY = 'exposure_priority'

# 算法控制
DEBUG = True
MAX_ITERATIONS = 50
MIN_ITERATIONS = 10
NO_VISUALIZATION = False                          # 不生成可视化图表

# 解调所需标签文件路径（必须正确设置）
LABEL_CSV_PATH = "C:/Users/A/Desktop/000/Mseq_32_original.csv"  # noqa

# ========================= 脚本名称 =========================
SCRIPT_NAME = Path(__file__).name


# ========================= 字体设置（英文） =========================
def setup_fonts_for_english() -> bool:
    """配置英文字体避免乱码"""
    system = platform.system()
    if system == 'Windows':
        font_list = ['Arial', 'Times New Roman', 'Calibri', 'Verdana']  # noqa
    elif system == 'Darwin':  # macOS
        font_list = ['Arial', 'Helvetica', 'Times New Roman']
    else:  # Linux
        font_list = ['DejaVu Sans', 'Liberation Sans', 'FreeSans']

    for font_name in font_list:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path:
                matplotlib.rcParams['font.family'] = font_name
                matplotlib.rcParams['font.sans-serif'] = [font_name]
                matplotlib.rcParams['axes.unicode_minus'] = False
                print(f"English font set: {font_name}")
                return True
        except (ValueError, OSError, RuntimeError):
            continue

    try:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        matplotlib.rcParams['axes.unicode_minus'] = False
        print("Using default English font")
        return True
    except Exception:  # noqa: E722
        print("Warning: Unable to set English font")
        return False


setup_fonts_for_english()


# ========================= 辅助函数 =========================
def parse_fraction(value: Any) -> float:
    """解析分数或科学计数法为浮点数"""
    if isinstance(value, (int, float)):
        return float(value)
    if '/' in str(value):
        numerator, denominator = str(value).split('/')
        return float(numerator) / float(denominator)
    return float(value)


# ========================= 解调相关函数 =========================
def find_sync(
    rr: np.ndarray,
    head_len: int = 8,
    max_head_len: int = 100,
    max_len_diff: int = 5,
    last_header_len: Optional[int] = None
) -> Tuple[int, int, int]:
    """检测同步头，返回等效比特长度、有效载荷起始位置、同步头起始位置"""
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
        filtered_runs = [r for r in runs_sorted
                         if abs(r[2] - last_header_len) <= max_len_diff]
        if filtered_runs:
            runs_sorted = filtered_runs

    header1_start, header1_end, len1 = runs_sorted[0]
    header2_start, header2_end, len2 = None, None, None

    for run in runs_sorted[1:]:
        if abs(run[2] - len1) <= max_len_diff:
            header2_start, header2_end, len2 = run
            break

    if header2_start is not None:
        payload_start = min(header1_end, header2_end) + 1
        equ = int(round(abs(((len1 + len2) / (head_len * 2)))))
        sync_header_start = min(header1_start, header2_start)
    else:
        payload_start = header1_end + 1
        equ = int(round(len1 / head_len))
        sync_header_start = header1_start

    return equ, payload_start, sync_header_start


def recover_data(rr: np.ndarray, payload_start: int, equ_len: int) -> np.ndarray:
    """根据等效比特长度恢复比特流"""
    p = payload_start
    res = []
    for i in range(payload_start, len(rr) - 1):
        if rr[i + 1] != rr[i]:
            q = i + 1
            width = q - p
            cnt = int(round(width / equ_len))
            res.extend([rr[i]] * cnt)
            p = q
    return np.array(res)


def evaluate(tx: np.ndarray, rx: np.ndarray) -> Tuple[int, float]:
    """计算误码数及误码率"""
    tx = tx[:len(rx)]
    num_errors = np.sum(tx != rx)
    ber = num_errors / len(tx)
    return num_errors, ber


def polyfit_threshold(y: np.ndarray, degree: int = 3) -> np.ndarray:
    """三阶多项式拟合阈值"""
    x = np.arange(1, len(y) + 1)
    coeffs = np.polyfit(x, y, degree)  # noqa
    return np.polyval(coeffs, x)


# ========================= 自适应阻尼自动曝光算法核心类 =========================
class AdaptiveDampingAutoExposure:
    """自适应阻尼自动曝光算法（以最小化误码率为目标）"""

    def __init__(
        self,
        target_brightness: float = 120.0,
        brightness_tolerance: float = 5.0,
        exposure_min: float = 1e-6,
        exposure_max: float = 1.0,
        gain_min: float = 1.0,
        gain_max: float = 32.0,
        initial_exposure: Optional[float] = None,
        initial_iso: Optional[float] = None,
        exposure_strategy: str = 'exposure_priority',
        debug_mode: bool = True
    ):
        # 参数初始化
        self.target_br = target_brightness
        self.tolerance = brightness_tolerance
        self.exp_min = exposure_min
        self.exp_max = exposure_max
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.exp_strategy = exposure_strategy
        self.debug = debug_mode
        self.iso_base = 100

        # 初始曝光时间
        if initial_exposure is not None:
            self.current_exp = max(min(initial_exposure, exposure_max), exposure_min)
            if self.debug:
                print(f"Initial exposure time: {self.format_exposure_time(self.current_exp)}")
        else:
            self.current_exp = np.sqrt(exposure_min * exposure_max)
            if self.debug:
                print(f"Using default exposure time: {self.format_exposure_time(self.current_exp)}")

        # 初始增益/ISO
        if initial_iso is not None:
            initial_gain = initial_iso / self.iso_base
            self.current_gain = max(min(initial_gain, gain_max), gain_min)
            if self.debug:
                print(f"Initial ISO {initial_iso} -> Gain {self.current_gain:.2f}")
        else:
            self.current_gain = np.sqrt(gain_min * gain_max)
            if self.debug:
                print(f"Using default gain: {self.current_gain:.2f}")

        # 保存初始值，以便reset时恢复
        self._initial_exp_value = self.current_exp
        self._initial_gain_value = self.current_gain

        # 状态机相关
        self.state = 'I'
        self.prev_state = 'I'
        self.references: Dict[str, Dict[str, Optional[float]]] = {
            'I': {'L': None, 'E': None, 'G': None},
            'II': {'L': None, 'E': None, 'G': None},
            'III': {'L': None, 'E': None, 'G': None},
            'IV': {'L': None, 'E': None, 'G': None},
            'V': {'L': None, 'E': None, 'G': None}
        }
        self.current_l: Optional[float] = None
        self.brightness_history: List[float] = []
        self.exposure_history: List[float] = []
        self.gain_history: List[float] = []
        self.damping_history: List[float] = []
        self.state_history: List[str] = []
        self.ber_history: List[float] = []
        self.iteration_cnt = 0
        self.max_iters = 50
        self.min_iters = 10
        self.state_ii_cnt = 0
        self.state_iii_cnt = 0
        self.state_iv_cnt = 0
        self.params_changed = False
        self.converged_flag = False
        self.initial_real_ber = None  # 用于存储第一次迭代的真实BER

        if self.debug:
            print("Auto Exposure Algorithm Initialized (BER minimization):")
            print(f"  Target Brightness: {self.target_br} ± {self.tolerance}")
            print(f"  Exposure Range: {self.format_exposure_time(self.exp_min)} to "
                  f"{self.format_exposure_time(self.exp_max)}")
            print(f"  Gain Range: {self.gain_min:.1f} to {self.gain_max:.1f}")
            print(f"  Initial Exposure: {self.format_exposure_time(self.current_exp)}")
            print(f"  Initial Gain: {self.current_gain:.2f}")
            print(f"  Initial ISO: {self.gain_to_iso(self.current_gain):.0f}")
            print(f"  Exposure Strategy: {self.exp_strategy}")

    # ========== 辅助方法 ==========
    def iso_to_gain(self, iso: float) -> float:
        return iso / self.iso_base

    def gain_to_iso(self, gain: float) -> float:
        return self.iso_base * gain

    @staticmethod
    def format_exposure_time(exposure_seconds: float) -> str:
        """格式化显示曝光时间"""
        if exposure_seconds >= 1:
            return f"{exposure_seconds:.3f} s"
        if exposure_seconds >= 0.001:
            return f"{exposure_seconds * 1000:.3f} ms"
        if exposure_seconds >= 1e-6:
            return f"{exposure_seconds * 1e6:.3f} µs"
        return f"{exposure_seconds:.3e} s"

    @staticmethod
    def calculate_roi_brightness(
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> float:
        """计算图像ROI平均亮度"""
        if roi is not None:
            x, y, w, h = roi
            x = max(0, min(x, image.shape[1]))
            y = max(0, min(y, image.shape[0]))
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            roi_image = image[y:y + h, x:x + w]
        else:
            roi_image = image
        if len(roi_image.shape) == 3:
            gray_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = roi_image
        return float(np.mean(gray_image))

    def need_auto_exposure(self, current_brightness: float) -> bool:
        """判断是否需要自动曝光"""
        return abs(current_brightness - self.target_br) > self.tolerance

    # ========== 算法1：比例法 ==========
    def algorithm1_proportional(
        self,
        l_curr: float,
        e_curr: Optional[float] = None,
        g_curr: Optional[float] = None
    ) -> Tuple[float, float]:
        """根据亮度比例计算新参数"""
        if l_curr < 0.1:
            l_curr = 0.1
        r = self.target_br / l_curr
        r = min(max(r, 0.1), 10.0)
        if self.debug:
            print("    Algorithm1 - Proportional:")
            print(f"      Current L: {l_curr:.1f}, Target L: {self.target_br}")
            print(f"      Ratio r = {self.target_br} / {l_curr:.1f} = {r:.3f}")
        if e_curr is not None:
            e_new = r * e_curr
            e_new = min(max(e_new, self.exp_min), self.exp_max)
            if self.debug:
                print(f"      New E: {self.format_exposure_time(e_new)} "
                      f"(from {self.format_exposure_time(e_curr)})")
            return e_new, g_curr if g_curr is not None else self.current_gain
        if g_curr is not None:
            g_new = r * g_curr
            g_new = min(max(g_new, self.gain_min), self.gain_max)
            if self.debug:
                print(f"      New G: {g_new:.2f} (from {g_curr:.2f})")
            return e_curr if e_curr is not None else self.current_exp, g_new
        return self.current_exp, self.current_gain

    # ========== 算法2：局部线性拟合 ==========
    def algorithm2_linear_fitting(
        self,
        l1: float, e1: float, g1: float,
        l2: float, e2: float, g2: float
    ) -> Tuple[float, float]:
        """利用两点线性插值计算新参数"""
        if abs(l2 - l1) < 1e-6:
            if self.debug:
                print("    Algorithm2 - Brightness difference too small, using Algorithm1")
            return self.algorithm1_proportional(
                (l1 + l2) / 2, (e1 + e2) / 2, (g1 + g2) / 2
            )
        if self.debug:
            print("    Algorithm2 - Local Linear Fitting:")
            print(f"      Point1: L={l1:.1f}, E={e1:.6f}, G={g1:.2f}")
            print(f"      Point2: L={l2:.1f}, E={e2:.6f}, G={g2:.2f}")
        if abs(e2 - e1) > 1e-10 and abs(g2 - g1) < 1e-6:
            e_new = e1 + (e2 - e1) * (self.target_br - l1) / (l2 - l1)
            e_new = min(max(e_new, self.exp_min), self.exp_max)
            if self.debug:
                print("      Adjusting exposure time:")
                print(f"      E_new = {e1:.6f} + ({e2:.6f} - {e1:.6f}) * "
                      f"({self.target_br} - {l1:.1f}) / ({l2:.1f} - {l1:.1f})")
                print(f"      E_new = {self.format_exposure_time(e_new)}")
            return e_new, g1
        if abs(g2 - g1) > 1e-6 and abs(e2 - e1) < 1e-10:
            g_new = g1 + (g2 - g1) * (self.target_br - l1) / (l2 - l1)
            g_new = min(max(g_new, self.gain_min), self.gain_max)
            if self.debug:
                print("      Adjusting gain:")
                print(f"      G_new = {g1:.2f} + ({g2:.2f} - {g1:.2f}) * "
                      f"({self.target_br} - {l1:.1f}) / ({l2:.1f} - {l1:.1f})")
                print(f"      G_new = {g_new:.2f}")
            return e1, g_new
        if self.debug:
            print("    Algorithm2 - Both exposure and gain changed, using Algorithm1")
        return self.algorithm1_proportional(
            (l1 + l2) / 2, (e1 + e2) / 2, (g1 + g2) / 2
        )

    # ========== 自适应阻尼计算 ==========
    def calculate_adaptive_damping(self, l_prev: float, l_curr: float) -> float:
        """根据亮度变化计算阻尼系数（范围 0～1）"""
        if l_prev < 1.0 or l_curr < 1.0:
            return 0.3
        lam = min(l_prev / l_curr, l_curr / l_prev)
        d_base = 1 - lam
        error_curr = abs(l_curr - self.target_br)
        if error_curr > 50:
            d_factor = 0.5
        elif error_curr > 20:
            d_factor = 0.7
        elif error_curr > 10:
            d_factor = 0.9
        else:
            d_factor = 1.0
        d_val = d_base * d_factor
        # 允许阻尼在 0～1 之间，但算法本身会倾向于不超过 0.5
        d_val = max(0.0, min(d_val, 1.0))
        if self.debug:
            print("    Adaptive Damping Calculation:")
            print(f"      L_prev: {l_prev:.1f}, L_curr: {l_curr:.1f}")
            print(f"      λ = min({l_prev:.1f}/{l_curr:.1f}, {l_curr:.1f}/{l_prev:.1f}) = {lam:.3f}")
            print(f"      d_base = 1 - λ = {d_base:.3f}")
            print(f"      Error: {error_curr:.1f}, Factor: {d_factor:.2f}")
            print(f"      Final d = {d_val:.3f}")
        return d_val

    # ========== 应用阻尼 ==========
    def apply_damping(
        self,
        e_old: float, g_old: float,
        e_new: float, g_new: float,
        d_val: float
    ) -> Tuple[float, float]:
        """对参数施加阻尼"""
        e_change = abs(e_new - e_old) / max(e_old, 1e-10)
        g_change = abs(g_new - g_old) / max(g_old, 1e-10)
        if e_change < 0.001 and e_old > 0:
            e_new = e_old * 1.001 if e_new > e_old else e_old * 0.999
        if g_change < 0.001 and g_old > 0:
            g_new = g_old * 1.001 if g_new > g_old else g_old * 0.999
        e_actual = d_val * e_old + (1 - d_val) * e_new
        g_actual = d_val * g_old + (1 - d_val) * g_new
        e_actual = min(max(e_actual, self.exp_min), self.exp_max)
        g_actual = min(max(g_actual, self.gain_min), self.gain_max)
        if self.debug:
            print(f"    Applying Damping (d={d_val:.3f}):")
            print(f"      E: {self.format_exposure_time(e_old)} → "
                  f"{self.format_exposure_time(e_actual)}")
            print(f"      G: {g_old:.2f} → {g_actual:.2f}")
        return e_actual, g_actual

    # ========== 亮度模拟（线性模型） ==========
    def simulate_brightness_change(
        self,
        current_l: float,
        e_old: float, g_old: float,
        e_new: float, g_new: float
    ) -> float:
        """根据参数变化线性模拟新亮度"""
        old_product = max(e_old * g_old, 1e-12)
        new_product = max(e_new * g_new, 1e-12)
        brightness_ratio = new_product / old_product
        new_l = current_l * brightness_ratio
        noise = np.random.normal(0, 1.0)
        new_l = new_l + noise
        new_l = min(max(new_l, 0), 255)
        if self.debug:
            print("    Brightness Simulation (Linear):")
            print(f"      Current L: {current_l:.1f}")
            print(f"      Old product: E×G = {e_old:.6f}×{g_old:.2f} = {old_product:.6f}")
            print(f"      New product: E×G = {e_new:.6f}×{g_new:.2f} = {new_product:.6f}")
            print(f"      Brightness ratio = {brightness_ratio:.3f}")
            print(f"      Simulated new L: {new_l:.1f}")
        return new_l

    # ========== 误码率模拟（仅基于亮度误差，无目标BER） ==========
    def calculate_ber_from_brightness(self, brightness: float) -> float:
        """根据亮度模拟BER，亮度误差越小BER越低"""
        brightness_error = abs(brightness - self.target_br)
        base_ber = (brightness_error / 255.0) * 0.1  # 不再加固定目标BER
        noise = np.random.uniform(-0.005, 0.005)
        ber = max(0.001, min(base_ber + noise, 0.5))
        if self.debug:
            print("    BER Calculation (minimization):")
            print(f"      Brightness: {brightness:.1f}, Target: {self.target_br}")
            print(f"      Error: {brightness_error:.1f}")
            print(f"      Base BER: {base_ber:.6f}, Noise: {noise:.6f}")
            print(f"      Final BER: {ber:.6f}")
        return ber

    # ========== 状态I：初始化规整 ==========
    def state_i_initialization(self) -> Tuple[float, float]:
        """状态I：初始化参数规整"""
        if self.debug:
            print("    State I: Initialization and Normalization")
        self.params_changed = True
        if self.exp_strategy in ['exposure_priority', 'exposure_only']:
            g_new = self.gain_min
            if self.current_gain > self.gain_min:
                e_new = (self.current_gain / self.gain_min) * self.current_exp
                if self.debug:
                    print(f"      Normalizing gain: {self.current_gain:.2f} → {g_new:.2f}")
                    print(
                        f"      Adjusting exposure: {self.format_exposure_time(self.current_exp)} → "
                        f"{self.format_exposure_time(e_new)}"
                    )
            else:
                e_new = self.current_exp * 1.5
                if self.debug:
                    print(f"      Gain already at minimum: {g_new:.2f}")
                    print(
                        f"      Initial exposure adjustment: {self.format_exposure_time(self.current_exp)} → "
                        f"{self.format_exposure_time(e_new)}"
                    )
        elif self.exp_strategy in ['gain_priority', 'gain_only']:
            e_new = self.exp_min
            if self.current_exp > self.exp_min:
                g_new = (self.current_exp / self.exp_min) * self.current_gain
                if self.debug:
                    print(
                        f"      Normalizing exposure: {self.format_exposure_time(self.current_exp)} → "
                        f"{self.format_exposure_time(e_new)}"
                    )
                    print(f"      Adjusting gain: {self.current_gain:.2f} → {g_new:.2f}")
            else:
                g_new = self.current_gain * 1.5
                if self.debug:
                    print(f"      Exposure already at minimum: {self.format_exposure_time(e_new)}")
                    print(f"      Initial gain adjustment: {self.current_gain:.2f} → {g_new:.2f}")
        else:
            e_new, g_new = self.current_exp, self.current_gain
        e_new = min(max(e_new, self.exp_min), self.exp_max)
        g_new = min(max(g_new, self.gain_min), self.gain_max)
        self.references['I'] = {'L': self.current_l, 'E': self.current_exp, 'G': self.current_gain}
        if self.debug:
            print(f"    State I completed: E={self.format_exposure_time(e_new)}, G={g_new:.2f}")
        return e_new, g_new

    # ========== 状态II：比例法快速单向收敛 ==========
    def state_ii_proportional_convergence(self) -> Tuple[float, float]:
        """状态II：比例法单向收敛"""
        if self.debug:
            print("    State II: Proportional Method for Fast Unidirectional Convergence")
        self.state_ii_cnt += 1
        self.params_changed = True
        if self.exp_strategy in ['exposure_priority', 'exposure_only']:
            e_new, g_new = self.algorithm1_proportional(self.current_l, self.current_exp, None)
            g_new = self.current_gain
        elif self.exp_strategy in ['gain_priority', 'gain_only']:
            e_new, g_new = self.algorithm1_proportional(self.current_l, None, self.current_gain)
            e_new = self.current_exp
        else:
            e_new, g_new = self.algorithm1_proportional(self.current_l, self.current_exp, None)
            g_new = self.current_gain
        if abs(e_new - self.current_exp) / max(self.current_exp, 1e-10) < 0.01:
            e_new = self.current_exp * 1.05 if e_new > self.current_exp else self.current_exp * 0.95
        if abs(g_new - self.current_gain) / max(self.current_gain, 1e-10) < 0.01:
            g_new = self.current_gain * 1.05 if g_new > self.current_gain else self.current_gain * 0.95
        return e_new, g_new

    # ========== 状态III：局部线性拟合法收敛 ==========
    def state_iii_linear_fitting_convergence(self) -> Tuple[float, float]:
        """状态III：线性拟合收敛"""
        if self.debug:
            print("    State III: Local Linear Fitting Method")
        self.state_iii_cnt += 1
        self.params_changed = True
        l1 = self.references['I']['L']
        e1 = self.references['I']['E']
        g1 = self.references['I']['G']
        l2 = self.references['II']['L']
        e2 = self.references['II']['E']
        g2 = self.references['II']['G']
        if None in (l1, e1, g1, l2, e2, g2):
            if self.debug:
                print("    State III: Incomplete reference values, using proportional method")
            return self.state_ii_proportional_convergence()
        if (l1 - self.target_br) * (l2 - self.target_br) > 0:
            if self.debug:
                print("    State III: Reference points on same side, using proportional method")
            return self.state_ii_proportional_convergence()
        e_new, g_new = self.algorithm2_linear_fitting(l1, e1, g1, l2, e2, g2)
        return e_new, g_new

    # ========== 状态IV：增益单向收敛 ==========
    def state_iv_gain_convergence(self) -> Tuple[float, float]:
        """状态IV：增益单向收敛"""
        if self.debug:
            print("    State IV: Gain Unidirectional Convergence")
        self.state_iv_cnt += 1
        self.params_changed = True
        e_new = self.current_exp
        _, g_new = self.algorithm1_proportional(self.current_l, None, self.current_gain)
        if abs(g_new - self.current_gain) / max(self.current_gain, 1e-10) < 0.01:
            g_new = self.current_gain * 1.05 if g_new > self.current_gain else self.current_gain * 0.95
        return e_new, g_new

    # ========== 状态V：增益夹逼收敛 ==========
    def state_v_gain_linear_convergence(self) -> Tuple[float, float]:
        """状态V：增益线性拟合收敛"""
        if self.debug:
            print("    State V: Gain Clamping Convergence")
        self.params_changed = True
        e_new = self.current_exp
        l1 = self.references['III']['L']
        e1 = self.references['III']['E']
        g1 = self.references['III']['G']
        l2 = self.references['IV']['L']
        e2 = self.references['IV']['E']
        g2 = self.references['IV']['G']
        if None in (l1, e1, g1, l2, e2, g2):
            if self.debug:
                print("    State V: Incomplete reference values, using proportional method")
            _, g_new = self.algorithm1_proportional(self.current_l, None, self.current_gain)
            return e_new, g_new
        if (l1 - self.target_br) * (l2 - self.target_br) > 0:
            if self.debug:
                print("    State V: Reference points on same side, using proportional method")
            _, g_new = self.algorithm1_proportional(self.current_l, None, self.current_gain)
            return e_new, g_new
        _, g_new = self.algorithm2_linear_fitting(l1, e1, g1, l2, e2, g2)
        return e_new, g_new

    # ========== 状态转移 ==========
    def update_state_transition(self) -> None:
        """更新状态机"""
        prev_state = self.state
        if self.state == 'I':
            self.state = 'II'
            if self.debug:
                print("    State transition: I → II (Initialization complete)")
        elif self.state == 'II':
            if self.references['I']['L'] is not None and self.current_l is not None:
                l_ref = self.references['I']['L']
                l_curr = self.current_l
                if (l_ref < self.target_br < l_curr) or (l_curr < self.target_br < l_ref):
                    self.references['II'] = {'L': l_curr, 'E': self.current_exp, 'G': self.current_gain}
                    self.state = 'III'
                    if self.debug:
                        print("    State transition: II → III (Brightness crossover detected)")
                else:
                    self.references['II'] = {'L': l_curr, 'E': self.current_exp, 'G': self.current_gain}
                    if self.state_ii_cnt >= 8:
                        self.state = 'III'
                        if self.debug:
                            print(f"    State transition: II → III (Max iterations reached: {self.state_ii_cnt})")
        elif self.state == 'III':
            if self.state_iii_cnt >= 5:
                self.references['III'] = {'L': self.current_l, 'E': self.current_exp, 'G': self.current_gain}
                self.state = 'IV'
                if self.debug:
                    print(f"    State transition: III → IV (Reached iteration count: {self.state_iii_cnt})")
            else:
                self.references['III'] = {'L': self.current_l, 'E': self.current_exp, 'G': self.current_gain}
        elif self.state == 'IV':
            if self.references['III']['L'] is not None and self.current_l is not None:
                l_ref = self.references['III']['L']
                l_curr = self.current_l
                if (l_ref < self.target_br < l_curr) or (l_curr < self.target_br < l_ref):
                    self.references['IV'] = {'L': l_curr, 'E': self.current_exp, 'G': self.current_gain}
                    self.state = 'V'
                    if self.debug:
                        print("    State transition: IV → V (Brightness crossover detected)")
                else:
                    self.references['IV'] = {'L': l_curr, 'E': self.current_exp, 'G': self.current_gain}
                    if self.state_iv_cnt >= 8:
                        self.state = 'V'
                        if self.debug:
                            print(f"    State transition: IV → V (Max iterations reached: {self.state_iv_cnt})")
        elif self.state == 'V':
            self.references['V'] = {'L': self.current_l, 'E': self.current_exp, 'G': self.current_gain}
        if prev_state != self.state and self.debug:
            print(f"    State changed: {prev_state} → {self.state}")

    # ========== 单帧处理 ==========
    def process_frame(self, image: np.ndarray, use_damping: bool = True) -> Dict[str, Any]:
        """处理一帧图像，返回结果"""
        if self.iteration_cnt == 0:
            self.current_l = self.calculate_roi_brightness(image)
            if self.initial_real_ber is not None:
                current_ber = self.initial_real_ber
                self.ber_history.append(current_ber)
            else:
                current_ber = self.calculate_ber_from_brightness(self.current_l)
        else:
            current_ber = self.ber_history[-1]

        if self.debug:
            print(f"\n  Iteration {self.iteration_cnt}:")
            print(f"    Current brightness: {self.current_l:.1f}")
            print(f"    Current BER: {current_ber:.6f}")
            print(f"    Current exposure: {self.format_exposure_time(self.current_exp)}, "
                  f"Gain: {self.current_gain:.2f}, ISO: {self.gain_to_iso(self.current_gain):.0f}")
            print(f"    Current state: {self.state}")

        if self.iteration_cnt == 0:
            self.brightness_history.append(self.current_l)
            self.exposure_history.append(self.current_exp)
            self.gain_history.append(self.current_gain)
            self.damping_history.append(0.1 if use_damping else 0.0)
            self.state_history.append(self.state)
            if self.initial_real_ber is None:
                self.ber_history.append(current_ber)

        if self.iteration_cnt >= self.max_iters:
            if self.debug:
                print(f"    Maximum iterations reached ({self.max_iters})")
            self.converged_flag = False
            return {
                'brightness': self.current_l,
                'ber': current_ber,
                'exposure': self.current_exp,
                'gain': self.current_gain,
                'iso': self.gain_to_iso(self.current_gain),
                'converged': False,
                'reason': 'max_iterations_reached'
            }

        if not self.need_auto_exposure(self.current_l) and self.iteration_cnt > self.min_iters:
            if self.debug:
                print(f"    Brightness within tolerance ({self.current_l:.1f} within {self.target_br} ± {self.tolerance})")
            self.converged_flag = True
            return {
                'brightness': self.current_l,
                'ber': current_ber,
                'exposure': self.current_exp,
                'gain': self.current_gain,
                'iso': self.gain_to_iso(self.current_gain),
                'damping': 0.0,
                'state': self.state,
                'converged': True
            }

        e_new, g_new = self.current_exp, self.current_gain
        if self.state == 'I':
            e_new, g_new = self.state_i_initialization()
        elif self.state == 'II':
            e_new, g_new = self.state_ii_proportional_convergence()
        elif self.state == 'III':
            e_new, g_new = self.state_iii_linear_fitting_convergence()
        elif self.state == 'IV':
            e_new, g_new = self.state_iv_gain_convergence()
        elif self.state == 'V':
            e_new, g_new = self.state_v_gain_linear_convergence()

        # ========== 确保计算出的参数在允许范围内 ==========
        e_new = min(max(e_new, self.exp_min), self.exp_max)
        g_new = min(max(g_new, self.gain_min), self.gain_max)

        if self.debug:
            print(f"    Calculated new parameters (clipped): E={self.format_exposure_time(e_new)}, G={g_new:.2f}")
        # =================================================

        self.params_changed = False
        if use_damping and len(self.brightness_history) > 0:
            l_prev = self.brightness_history[-1]
            d_val = self.calculate_adaptive_damping(l_prev, self.current_l)
        else:
            d_val = 0.0
            if self.debug:
                print("    No damping mode")
        if use_damping and d_val > 0:
            if self.debug:
                print(f"    Applying damping: d={d_val:.3f}")
            e_actual, g_actual = self.apply_damping(
                self.current_exp, self.current_gain, e_new, g_new, d_val
            )
        else:
            e_actual, g_actual = e_new, g_new
            if self.debug:
                print("    No damping applied: using calculated parameters directly")
        e_change = abs(e_actual - self.current_exp) / max(self.current_exp, 1e-10)
        g_change = abs(g_actual - self.current_gain) / max(self.current_gain, 1e-10)
        if e_change < 0.001 and g_change < 0.001 and self.iteration_cnt > 5:
            if self.debug:
                print("    Warning: Parameters not changing, forcing change")
            if e_actual <= self.current_exp:
                e_actual = self.current_exp * 1.05
            else:
                e_actual = self.current_exp * 0.95
            if g_actual <= self.current_gain:
                g_actual = self.current_gain * 1.05
            else:
                g_actual = self.current_gain * 0.95
            e_actual = min(max(e_actual, self.exp_min), self.exp_max)
            g_actual = min(max(g_actual, self.gain_min), self.gain_max)
            self.params_changed = True
        new_brightness = self.simulate_brightness_change(
            self.current_l, self.current_exp, self.current_gain, e_actual, g_actual
        )
        new_ber = self.calculate_ber_from_brightness(new_brightness)
        old_e, old_g = self.current_exp, self.current_gain
        self.current_exp = e_actual
        self.current_gain = g_actual
        self.current_l = new_brightness
        if self.debug:
            print(f"    Parameters updated:")
            print(f"      E: {self.format_exposure_time(old_e)} → {self.format_exposure_time(self.current_exp)}")
            print(f"      G: {old_g:.2f} → {self.current_gain:.2f}")
            print(f"      ISO: {self.gain_to_iso(old_g):.0f} → {self.gain_to_iso(self.current_gain):.0f}")
            prev_l = self.brightness_history[-1] if self.brightness_history else 0
            print(f"      L: {prev_l:.1f} → {self.current_l:.1f}")
            print(f"      BER: {current_ber:.6f} → {new_ber:.6f}")
        self.update_state_transition()
        self.brightness_history.append(self.current_l)
        self.exposure_history.append(self.current_exp)
        self.gain_history.append(self.current_gain)
        self.damping_history.append(d_val)
        self.state_history.append(self.state)
        self.ber_history.append(new_ber)
        self.iteration_cnt += 1
        self.converged_flag = not self.need_auto_exposure(self.current_l) and self.iteration_cnt >= self.min_iters
        if self.converged_flag and self.debug:
            print("    Convergence condition satisfied!")
        return {
            'brightness': self.current_l,
            'ber': new_ber,
            'exposure': self.current_exp,
            'gain': self.current_gain,
            'iso': self.gain_to_iso(self.current_gain),
            'damping': d_val,
            'state': self.state,
            'converged': self.converged_flag,
            'params_changed': self.params_changed,
            'iteration': self.iteration_cnt
        }

    # ========== 完整图像处理 ==========
    def process_image(
        self,
        image: np.ndarray,
        use_damping: bool = True,
        initial_real_ber: Optional[float] = None
    ) -> Dict[str, Any]:
        """对单张图像执行自动曝光流程，可传入初始真实BER"""
        self.reset()
        self.initial_real_ber = initial_real_ber
        if self.debug:
            print(f"\n{'=' * 60}")
            print("Starting Auto-Exposure Algorithm (BER minimization)")
            print(f"{'=' * 60}")
            print(f"  Target brightness: {self.target_br} ± {self.tolerance}")
            print("  Initial parameters:")
            print(f"    Exposure time: {self.format_exposure_time(self.current_exp)}")
            print(f"    Gain: {self.current_gain:.2f}")
            print(f"    ISO: {self.gain_to_iso(self.current_gain):.0f}")
            if initial_real_ber is not None:
                print(f"  Initial real BER: {initial_real_ber:.6f}")
            print(f"  Exposure strategy: {self.exp_strategy}")
            print(f"  Damping: {'Enabled' if use_damping else 'Disabled'}")
        last_change_iteration = 0
        convergence_data = []
        for i in range(self.max_iters):
            if self.debug:
                print(f"\n{'=' * 40}")
            result = self.process_frame(image, use_damping)
            convergence_data.append(result)
            if result.get('params_changed', False):
                last_change_iteration = i
            if i - last_change_iteration > 10 and i >= self.min_iters:
                if self.debug:
                    print("\n    No parameter change for 10 iterations, stopping")
                break
            if result['converged']:
                if self.debug:
                    print(f"\n    Converged at iteration {i + 1}")
                break
            if i == self.max_iters - 1:
                if self.debug:
                    print(f"\n    Reached maximum iterations {self.max_iters}")
        final_brightness_error = abs(self.brightness_history[-1] - self.target_br) if self.brightness_history else 0
        brightness_stability = np.std(self.brightness_history) if len(self.brightness_history) > 1 else 0
        final_ber = self.ber_history[-1] if self.ber_history else 0.0
        return {
            'brightness_history': self.brightness_history,
            'exposure_history': self.exposure_history,
            'gain_history': self.gain_history,
            'iso_history': [self.gain_to_iso(g) for g in self.gain_history],
            'damping_history': self.damping_history if use_damping else [0.0] * len(self.brightness_history),
            'state_history': self.state_history,
            'ber_history': self.ber_history,
            'iterations': self.iteration_cnt,
            'final_brightness': self.brightness_history[-1] if self.brightness_history else self.current_l,
            'final_ber': final_ber,
            'final_exposure': self.exposure_history[-1] if self.exposure_history else self.current_exp,
            'final_gain': self.gain_history[-1] if self.gain_history else self.current_gain,
            'final_iso': (self.gain_to_iso(self.gain_history[-1])
                          if self.gain_history else self.gain_to_iso(self.current_gain)),
            'initial_exposure': self.exposure_history[0] if self.exposure_history else self.current_exp,
            'initial_gain': self.gain_history[0] if self.gain_history else self.current_gain,
            'initial_iso': (self.gain_to_iso(self.gain_history[0])
                            if self.gain_history else self.gain_to_iso(self.current_gain)),
            'brightness_error': final_brightness_error,
            'brightness_stability': brightness_stability,
            'converged': self.converged_flag,
            'convergence_data': convergence_data
        }

    # ========== 重置状态 ==========
    def reset(self) -> None:
        """重置算法状态，并恢复初始曝光和增益"""
        self.state = 'I'
        self.prev_state = 'I'
        for state in self.references:
            self.references[state] = {'L': None, 'E': None, 'G': None}
        self.brightness_history = []
        self.exposure_history = []
        self.gain_history = []
        self.damping_history = []
        self.state_history = []
        self.ber_history = []
        self.iteration_cnt = 0
        self.converged_flag = False
        self.state_ii_cnt = 0
        self.state_iii_cnt = 0
        self.state_iv_cnt = 0
        self.params_changed = False
        self.current_l = None
        self.initial_real_ber = None
        # 恢复初始曝光和增益
        self.current_exp = self._initial_exp_value
        self.current_gain = self._initial_gain_value


# ========================= 自动曝光系统封装类 =========================
class AutoExposureSystem:
    """自动曝光系统，封装算法并提供处理接口"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.default_config = {
            'target_brightness': 120.0,
            'brightness_tolerance': 5.0,
            'exposure_min': 1e-6,
            'exposure_max': 1.0,
            'gain_min': 1.0,
            'gain_max': 32.0,
            'exposure_strategy': 'exposure_priority',
            'debug_mode': True,
            'initial_exposure': None,
            'initial_iso': None
        }
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        self.algorithm = AdaptiveDampingAutoExposure(
            target_brightness=self.config['target_brightness'],
            brightness_tolerance=self.config['brightness_tolerance'],
            exposure_min=self.config['exposure_min'],
            exposure_max=self.config['exposure_max'],
            gain_min=self.config['gain_min'],
            gain_max=self.config['gain_max'],
            initial_exposure=self.config['initial_exposure'],
            initial_iso=self.config['initial_iso'],
            exposure_strategy=self.config['exposure_strategy'],
            debug_mode=self.config['debug_mode']
        )
        # 加载标签数据用于解调
        self.input_bits = None
        if LABEL_CSV_PATH and os.path.exists(LABEL_CSV_PATH):
            try:
                df = pd.read_csv(LABEL_CSV_PATH, skiprows=5)
                self.input_bits = df.iloc[:, 1].to_numpy()
                print(f"✅ 已加载统一标签，比特长度 = {len(self.input_bits)}")
            except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError) as e:
                print(f"⚠️ 加载标签文件失败: {e}")
        else:
            print("⚠️ 标签文件路径未设置或不存在，解调功能将不可用")

        print("Auto Exposure System Initialized (BER minimization)")
        print(f"Configuration: {self.config}")

    def demodulate_single(self, image_path: str) -> Optional[float]:
        """对单张图像进行解调，返回BER，失败返回None"""
        if self.input_bits is None:
            print("❌ 标签数据未加载，无法解调")
            return None
        try:
            img_gray = Image.open(image_path).convert('L')
            img = np.array(img_gray, dtype=np.float64)

            column = np.mean(img, axis=1)
            mean = np.mean(column)
            std = np.std(column)
            y = (column - mean) / std

            threshold = polyfit_threshold(y, degree=3)
            yy = y - threshold
            rr = (yy > 0).astype(int)

            equ, payload_start, sync_start = find_sync(rr)

            res = recover_data(rr, payload_start, equ)
            res = res[1:len(self.input_bits) + 1]

            if len(res) == len(self.input_bits):
                num, ber = evaluate(self.input_bits, res)
                return ber
            print(f"⚠️ 恢复数据长度不足，期望 {len(self.input_bits)}，实际 {len(res)}")
            return None
        except (ValueError, IndexError, OSError) as e:
            print(f"❌ 解调失败: {e}")
            return None

    def process_single_image(self, image_path: str, use_damping: bool = True) -> Dict[str, Any]:
        """处理单张图像，先解调得到初始BER，再运行自动曝光"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"\n{'=' * 60}")
        print(f"Demodulating image: {image_path}")
        initial_ber = self.demodulate_single(image_path)
        if initial_ber is not None:
            print(f"Initial real BER = {initial_ber:.6f}")
        else:
            print("⚠️ 解调失败，将使用模拟初始BER")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        print(f"\n{'=' * 60}")
        print(f"Processing single image: {image_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        print(f"Damping: {'Enabled' if use_damping else 'Disabled'}")
        print(f"{'=' * 60}")

        result = self.algorithm.process_image(image, use_damping, initial_real_ber=initial_ber)
        return result

    def process_folder(
        self,
        folder_path: str,
        use_damping: bool = True,
        extensions: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """批量处理文件夹内图像，每张图像先解调再自动曝光"""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        if not folder.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")
        print(f"\n{'=' * 60}")
        print(f"Processing folder: {folder_path}")
        print(f"Supported extensions: {extensions}")
        print(f"Damping: {'Enabled' if use_damping else 'Disabled'}")
        print(f"{'=' * 60}")
        image_files = []
        for ext in extensions:
            image_files.extend(list(folder.glob(f"*{ext}")))
            image_files.extend(list(folder.glob(f"*{ext.upper()}")))
        image_files = list(set(image_files))
        if not image_files:
            raise ValueError(f"No image files found in folder: {folder_path}")
        print(f"Found {len(image_files)} image files:")
        for img_file in image_files:
            print(f"  - {img_file.name}")
        all_results = {}
        for i, img_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {img_file.name}")
            try:
                result = self.process_single_image(str(img_file), use_damping)
                all_results[str(img_file)] = result
                self.algorithm.reset()
            except Exception as e:
                print(f"Error processing {img_file.name}: {str(e)}")
                all_results[str(img_file)] = {'error': str(e)}
        return all_results

    # ========== 策略比较 ==========
    def compare_strategies_single(self, image_path: str) -> Dict[str, Any]:
        """比较单张图像有/无阻尼效果（同样先解调）"""
        print(f"\n{'=' * 60}")
        print("Comparing Damping Strategies (Single Image)")
        print(f"Image: {image_path}")
        print(f"{'=' * 60}")
        print("\n1. With Damping Strategy:")
        result_with = self.process_single_image(image_path, use_damping=True)
        self.algorithm.reset()
        print("\n2. Without Damping Strategy:")
        result_without = self.process_single_image(image_path, use_damping=False)
        comparison = {
            'with_damping': result_with,
            'without_damping': result_without,
            'comparison': self._compare_results(result_with, result_without)
        }
        return comparison

    def compare_strategies_folder(
        self,
        folder_path: str,
        extensions: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """批量比较有/无阻尼"""
        print(f"\n{'=' * 60}")
        print("Comparing Damping Strategies (Folder)")
        print(f"Folder: {folder_path}")
        print(f"{'=' * 60}")
        print("\n1. Processing with damping strategy:")
        results_with = self.process_folder(folder_path, use_damping=True, extensions=extensions)
        self.algorithm.reset()
        print("\n2. Processing without damping strategy:")
        results_without = self.process_folder(folder_path, use_damping=False, extensions=extensions)
        all_comparisons = {}
        for img_path in results_with:
            if img_path in results_without:
                comparison = self._compare_results(results_with[img_path], results_without[img_path])
                all_comparisons[img_path] = {
                    'with_damping': results_with[img_path],
                    'without_damping': results_without[img_path],
                    'comparison': comparison
                }
        return all_comparisons

    def _compare_results(self, result_with: Dict[str, Any], result_without: Dict[str, Any]) -> Dict[str, Any]:
        """比较两组结果，使用重新计算的亮度误差"""
        if 'error' in result_with or 'error' in result_without:
            return {
                'error': f"With damping: {result_with.get('error', 'OK')}, "
                         f"Without damping: {result_without.get('error', 'OK')}"
            }
        target = self.config['target_brightness']
        speed_with = result_with['iterations']
        speed_without = result_without['iterations']
        accuracy_with = abs(result_with['final_brightness'] - target)
        accuracy_without = abs(result_without['final_brightness'] - target)
        stability_with = result_with['brightness_stability']
        stability_without = result_without['brightness_stability']
        ber_with = result_with['final_ber']
        ber_without = result_without['final_ber']
        score_with = self._calculate_score(speed_with, accuracy_with, stability_with, ber_with)
        score_without = self._calculate_score(speed_without, accuracy_without, stability_without, ber_without)
        better_strategy = 'with_damping' if score_with > score_without else 'without_damping'
        return {
            'convergence_speed': {
                'with_damping': speed_with,
                'without_damping': speed_without,
                'improvement': (speed_without - speed_with) / max(speed_without, 1) * 100
            },
            'accuracy': {
                'with_damping': accuracy_with,
                'without_damping': accuracy_without,
                'improvement': (accuracy_without - accuracy_with) / max(accuracy_without, 0.1) * 100
            },
            'stability': {
                'with_damping': stability_with,
                'without_damping': stability_without,
                'improvement': (stability_without - stability_with) / max(stability_without, 0.1) * 100
            },
            'ber_performance': {
                'with_damping': ber_with,
                'without_damping': ber_without,
                'improvement': (ber_without - ber_with) / max(ber_without, 0.001) * 100
            },
            'overall_score': {
                'with_damping': score_with,
                'without_damping': score_without
            },
            'better_strategy': better_strategy,
            'recommendation': self._get_recommendation(better_strategy, score_with, score_without)
        }

    @staticmethod
    def _calculate_score(speed: int, accuracy: float, stability: float, ber: float) -> float:
        """计算综合评分"""
        w_speed = 0.25
        w_accuracy = 0.35
        w_stability = 0.20
        w_ber = 0.20
        speed_score = max(0, 100 - speed * 2)
        accuracy_score = max(0, 100 - accuracy * 10)
        stability_score = max(0, 100 - stability * 100)
        ber_score = max(0, 100 - ber * 2000)  # BER越低评分越高
        total_score = (speed_score * w_speed +
                       accuracy_score * w_accuracy +
                       stability_score * w_stability +
                       ber_score * w_ber)
        return total_score

    @staticmethod
    def _get_recommendation(better_strategy: str, score_with: float, score_without: float) -> str:
        """给出推荐建议"""
        if better_strategy == 'with_damping':
            return (
                "基于分析结果，推荐使用有阻尼策略。"
                f"有阻尼策略的综合评分({score_with:.1f})高于无阻尼策略({score_without:.1f})，"
                "在收敛速度和稳定性方面表现更好。"
            )
        return (
            "基于分析结果，推荐使用无阻尼策略。"
            f"无阻尼策略的综合评分({score_without:.1f})高于有阻尼策略({score_with:.1f})，"
            "在简单场景下收敛更快。"
        )

    # ========== 可视化 ==========
    def visualize_results(
        self,
        result: Dict[str, Any],
        save_path: str = None,
        title: str = None
    ) -> None:
        """生成结果可视化图表，并在曝光时间和增益曲线上标注最佳值"""
        if save_path is None:
            save_path = "auto_exposure_results.png"
        if title is None:
            title = 'Adaptive Damping Auto-Exposure Analysis (BER Minimization)'
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 亮度曲线
        ax1 = axes[0, 0]
        if result['brightness_history']:
            iterations = range(len(result['brightness_history']))
            ax1.plot(iterations, result['brightness_history'], 'b-o', markersize=4, label='Brightness')
            ax1.axhline(
                y=self.config['target_brightness'], color='r', linestyle='--',
                label=f'Target: {self.config["target_brightness"]}'
            )
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Brightness')
            ax1.set_title('Brightness Adjustment')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 曝光时间曲线
        ax2 = axes[0, 1]
        if result['exposure_history']:
            iterations = range(len(result['exposure_history']))
            ax2.plot(iterations, result['exposure_history'], 'g-o', markersize=4, label='Exposure time')
            # 标记最佳曝光时间（最终值）
            best_exp = result['final_exposure']
            best_iter = len(result['exposure_history']) - 1
            ax2.plot(
                best_iter, best_exp, 'r*', markersize=12,
                label=f'Best: {self.algorithm.format_exposure_time(best_exp)}'
            )
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Exposure Time (s)')
            ax2.set_title('Exposure Time Adjustment')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')

        # 增益曲线
        ax3 = axes[0, 2]
        if result['gain_history']:
            iterations = range(len(result['gain_history']))
            ax3.plot(iterations, result['gain_history'], 'c-o', markersize=4, label='Gain')
            # 标记最佳增益（最终值）
            best_gain = result['final_gain']
            best_iter = len(result['gain_history']) - 1
            ax3.plot(best_iter, best_gain, 'r*', markersize=12, label=f'Best: {best_gain:.2f}')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Gain')
            ax3.set_title('Gain Adjustment')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 阻尼曲线
        ax4 = axes[1, 0]
        if result['damping_history']:
            iterations = range(len(result['damping_history']))
            ax4.plot(iterations, result['damping_history'], 'y-o', markersize=4)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Damping Factor')
            ax4.set_title('Damping Factor Change')
            ax4.grid(True, alpha=0.3)

        # BER曲线
        ax5 = axes[1, 1]
        if result['ber_history']:
            iterations = range(len(result['ber_history']))
            ax5.plot(iterations, result['ber_history'], 'm-o', markersize=4, label='BER')
            ax5.set_xlabel('Iteration')
            ax5.set_ylabel('BER')
            ax5.set_title('BER Performance')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            if len(result['ber_history']) > 0:
                initial_ber = result['ber_history'][0]
                final_ber = result['ber_history'][-1]
                ax5.annotate(
                    f'Initial BER: {initial_ber:.6f}', xy=(0, initial_ber),
                    xytext=(5, initial_ber + 0.01), fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='blue')
                )
                ax5.annotate(
                    f'Final BER: {final_ber:.6f}', xy=(len(iterations) - 1, final_ber),
                    xytext=(len(iterations) - 5, final_ber + 0.01), fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='green')
                )

        # 摘要信息
        ax6 = axes[1, 2]
        ax6.axis('off')
        info_text = (
            f"Summary\n\nIterations: {result['iterations']}\n"
            f"Final Brightness: {result['final_brightness']:.1f}\n"
            f"Final BER: {result['final_ber']:.6f}\n"
            f"Converged: {'Yes' if result['converged'] else 'No'}"
        )
        ax6.text(
            0.5, 0.5, info_text, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Results visualization saved to: {save_path}")

    # ========== 生成报告 ==========
    def generate_summary_report(
        self,
        all_results: Dict[str, Dict[str, Any]],
        output_path: str
    ) -> Dict[str, Any]:
        """生成批量处理总结报告，使用重新计算的亮度误差"""
        if not all_results:
            print("Warning: No results to generate report")
            return {}
        stats = {
            'total_images': len(all_results),
            'converged_images': 0,
            'failed_images': 0,
            'iterations': [],
            'brightness_errors': [],
            'bers': [],
            'scores': []
        }
        report_lines = [
            "=" * 80,
            "自适应阻尼平滑自动曝光算法批量处理总结报告（误码率最小化）",
            "=" * 80,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"处理图像总数: {len(all_results)}",
            f"目标亮度: {self.config['target_brightness']} ± {self.config['brightness_tolerance']}",
            f"曝光时间范围: {self.algorithm.format_exposure_time(self.config['exposure_min'])} "
            f"到 {self.algorithm.format_exposure_time(self.config['exposure_max'])}",
            f"增益范围: {self.config['gain_min']} 到 {self.config['gain_max']}",
            f"曝光策略: {self.config['exposure_strategy']}",
            "",
            "一、处理结果统计",
            "-" * 40,
            f"{'图像名称':<30} {'迭代次数':<10} {'初始曝光':<15} {'最终曝光':<15} "
            f"{'初始增益':<10} {'最终增益':<10} {'亮度误差':<10} {'最终BER':<12} {'收敛状态':<8}",
            "-" * 80
        ]
        for img_path, result in all_results.items():
            img_name = Path(img_path).name
            if 'error' in result:
                stats['failed_images'] += 1
                report_lines.append(
                    f"{img_name:<30} {'ERROR':<10} {'ERROR':<15} {'ERROR':<15} "
                    f"{'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12} {'ERROR':<8}"
                )
            else:
                stats['converged_images'] += 1 if result['converged'] else 0
                stats['iterations'].append(result['iterations'])
                brightness_error = abs(result['final_brightness'] - self.config['target_brightness'])
                stats['brightness_errors'].append(brightness_error)
                stats['bers'].append(result['final_ber'])
                score = self._calculate_score(
                    result['iterations'],
                    brightness_error,
                    result['brightness_stability'],
                    result['final_ber']
                )
                stats['scores'].append(score)

                initial_exp_str = self.algorithm.format_exposure_time(result['initial_exposure'])
                final_exp_str = self.algorithm.format_exposure_time(result['final_exposure'])
                initial_gain_str = f"{result['initial_gain']:.2f}"
                final_gain_str = f"{result['final_gain']:.2f}"

                report_lines.append(
                    f"{img_name:<30} {result['iterations']:<10} {initial_exp_str:<15} {final_exp_str:<15} "
                    f"{initial_gain_str:<10} {final_gain_str:<10} {brightness_error:<10.2f} "
                    f"{result['final_ber']:<12.6f} {'是' if result['converged'] else '否':<8}"
                )
        if stats['iterations']:
            stats['convergence_rate'] = stats['converged_images'] / len(all_results) * 100
            stats['avg_iterations'] = np.mean(stats['iterations'])
            stats['avg_brightness_error'] = np.mean(stats['brightness_errors'])
            stats['avg_ber'] = np.mean(stats['bers'])
            stats['avg_score'] = np.mean(stats['scores'])
            stats['min_score'] = np.min(stats['scores'])
            stats['max_score'] = np.max(stats['scores'])
        else:
            stats['convergence_rate'] = 0
            stats['avg_iterations'] = 0
            stats['avg_brightness_error'] = 0
            stats['avg_ber'] = 0
            stats['avg_score'] = 0
            stats['min_score'] = 0
            stats['max_score'] = 0
        report_lines.extend([
            "",
            "二、统计信息",
            "-" * 40,
            f"收敛图像数量: {stats['converged_images']}/{len(all_results)} ({stats['convergence_rate']:.1f}%)",
            f"平均迭代次数: {stats['avg_iterations']:.1f}",
            f"平均亮度误差: {stats['avg_brightness_error']:.2f}",
            f"平均最终BER: {stats['avg_ber']:.6f}",
            f"平均综合评分: {stats['avg_score']:.1f}/100",
            f"最低评分: {stats['min_score']:.1f}/100",
            f"最高评分: {stats['max_score']:.1f}/100",
            "",
            "三、结论与建议",
            "-" * 40
        ])
        if stats['convergence_rate'] > 80 and stats['avg_score'] > 80:
            conclusion = "算法在大多数图像上表现优秀，收敛成功率高且综合评分高。"
        elif stats['convergence_rate'] > 60 and stats['avg_score'] > 60:
            conclusion = "算法在多数图像上表现良好，但在某些图像上可能需要优化。"
        else:
            conclusion = "算法在某些图像上表现不佳，建议检查参数设置或图像质量。"
        report_lines.append(conclusion)
        report_lines.extend([
            "",
            "四、优化建议",
            "-" * 40,
            "1. 对于高动态范围场景，建议调整曝光时间范围",
            "2. 对于低光照条件，可以适当提高增益上限",
            "3. 对于复杂场景，可以尝试不同的曝光策略",
            "4. 根据具体应用调整目标亮度和容差范围",
            "",
            "=" * 80,
            "报告结束",
            "=" * 80
        ])
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"Summary report saved to: {output_path}")
        return stats

    # ========== 保存CSV ==========
    def save_results_to_csv(self, all_results: Dict[str, Dict[str, Any]], output_path: str) -> None:
        """将结果保存为CSV文件，使用重新计算的亮度误差"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Image Name', 'Iterations', 'Converged',
                'Initial Brightness', 'Final Brightness', 'Brightness Error',
                'Initial Exposure (s)', 'Final Exposure (s)',
                'Initial Gain', 'Final Gain',
                'Initial ISO', 'Final ISO',
                'Initial BER', 'Final BER',
                'Brightness Stability', 'Score'
            ])
            for img_path, result in all_results.items():
                img_name = Path(img_path).name
                if 'error' in result:
                    writer.writerow([img_name, 'ERROR', 'ERROR', 'ERROR', 'ERROR',
                                     'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR',
                                     'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR'])
                else:
                    brightness_error = abs(result['final_brightness'] - self.config['target_brightness'])
                    score = self._calculate_score(
                        result['iterations'],
                        brightness_error,
                        result['brightness_stability'],
                        result['final_ber']
                    )
                    writer.writerow([
                        img_name,
                        result['iterations'],
                        'Yes' if result['converged'] else 'No',
                        result['brightness_history'][0] if result['brightness_history'] else 0,
                        result['final_brightness'],
                        f"{brightness_error:.2f}",
                        result['initial_exposure'],
                        result['final_exposure'],
                        result['initial_gain'],
                        result['final_gain'],
                        result['initial_iso'],
                        result['final_iso'],
                        result['ber_history'][0] if result['ber_history'] else 0,
                        result['final_ber'],
                        result['brightness_stability'],
                        f"{score:.1f}"
                    ])
        print(f"Results saved to CSV: {output_path}")


# ========================= 辅助函数：运行单图像模式 =========================
def _run_single_mode(system: AutoExposureSystem, timestamp: str, output_dir: Path) -> None:
    """处理单图像模式"""
    if COMPARE:
        comparison = system.compare_strategies_single(INPUT_PATH)
        if not NO_VISUALIZATION:
            img_name = Path(INPUT_PATH).stem
            vis_with_path = output_dir / f"single_with_damping_{img_name}_{timestamp}.png"
            system.visualize_results(
                comparison['with_damping'],
                str(vis_with_path),
                f"With Damping - {img_name}"
            )
            vis_without_path = output_dir / f"single_without_damping_{img_name}_{timestamp}.png"
            system.visualize_results(
                comparison['without_damping'],
                str(vis_without_path),
                f"Without Damping - {img_name}"
            )
        comp = comparison['comparison']
        print(f"\n{'=' * 60}")
        print("策略比较结果 (单图像)")
        print(f"{'=' * 60}")
        print(f"收敛速度:")
        print(f"  有阻尼: {comp['convergence_speed']['with_damping']} 次迭代")
        print(f"  无阻尼: {comp['convergence_speed']['without_damping']} 次迭代")
        print(f"  改进: {comp['convergence_speed']['improvement']:.1f}%")
        print("\n精度:")
        print(f"  有阻尼: {comp['accuracy']['with_damping']:.2f} 亮度误差")
        print(f"  无阻尼: {comp['accuracy']['without_damping']:.2f} 亮度误差")
        print(f"  改进: {comp['accuracy']['improvement']:.1f}%")
        print("\nBER性能:")
        print(f"  有阻尼: {comp['ber_performance']['with_damping']:.6f}")
        print(f"  无阻尼: {comp['ber_performance']['without_damping']:.6f}")
        print(f"  改进: {comp['ber_performance']['improvement']:.1f}%")
        print("\n综合评分:")
        print(f"  有阻尼: {comp['overall_score']['with_damping']:.1f}/100")
        print(f"  无阻尼: {comp['overall_score']['without_damping']:.1f}/100")
        print(f"\n推荐策略: {comp['better_strategy']}")
        print(f"\n{comp['recommendation']}")
        print(f"\n结果已保存到: {output_dir}")
    else:
        result = system.process_single_image(INPUT_PATH, use_damping=True)
        if not NO_VISUALIZATION:
            img_name = Path(INPUT_PATH).stem
            vis_path = output_dir / f"single_visualization_{img_name}_{timestamp}.png"
            system.visualize_results(result, str(vis_path), f"Auto-Exposure Result - {img_name}")
        print("\n处理完成!")
        print(f"最终亮度: {result['final_brightness']:.1f} "
              f"(目标: {TARGET_BRIGHTNESS}, 误差: {abs(result['final_brightness'] - TARGET_BRIGHTNESS):.2f})")
        print(f"最终BER: {result['final_ber']:.6f}")
        print(f"最终曝光: {system.algorithm.format_exposure_time(result['final_exposure'])}")
        print(f"最终增益: {result['final_gain']:.2f} (ISO: {result['final_iso']:.0f})")
        print(f"迭代次数: {result['iterations']}")
        print(f"收敛: {'是' if result['converged'] else '否'}")
        print(f"\n结果已保存到: {output_dir}")


# ========================= 辅助函数：运行批量模式 =========================
def _run_batch_mode(system: AutoExposureSystem, timestamp: str, output_dir: Path) -> None:
    """处理批量模式"""
    if COMPARE:
        all_comparisons = system.compare_strategies_folder(INPUT_PATH, EXTENSIONS)
        results_with = {}
        for img_path, comparison in all_comparisons.items():
            if 'with_damping' in comparison:
                results_with[img_path] = comparison['with_damping']
        if results_with:
            summary_path = output_dir / f"batch_comparison_summary_{timestamp}.txt"
            stats = system.generate_summary_report(results_with, str(summary_path))
            csv_path = output_dir / f"batch_comparison_results_{timestamp}.csv"
            system.save_results_to_csv(results_with, str(csv_path))

            # 统一可视化比较结果
            if not NO_VISUALIZATION:
                for img_path, comparison in all_comparisons.items():
                    img_name = Path(img_path).stem
                    if 'with_damping' in comparison and 'error' not in comparison['with_damping']:
                        vis_with_path = output_dir / f"batch_with_damping_{img_name}_{timestamp}.png"
                        system.visualize_results(
                            comparison['with_damping'],
                            str(vis_with_path),
                            f"With Damping - {img_name}"
                        )
                    if 'without_damping' in comparison and 'error' not in comparison['without_damping']:
                        vis_without_path = output_dir / f"batch_without_damping_{img_name}_{timestamp}.png"
                        system.visualize_results(
                            comparison['without_damping'],
                            str(vis_without_path),
                            f"Without Damping - {img_name}"
                        )
            print(f"\n{'=' * 60}")
            print("批量比较处理完成!")
            print(f"{'=' * 60}")
            print(f"处理图像总数: {stats['total_images']}")
            print(f"收敛图像数量: {stats['converged_images']} ({stats['convergence_rate']:.1f}%)")
            print(f"平均迭代次数: {stats['avg_iterations']:.1f}")
            print(f"平均亮度误差: {stats['avg_brightness_error']:.2f}")
            print(f"平均最终BER: {stats['avg_ber']:.6f}")
            print(f"平均综合评分: {stats['avg_score']:.1f}/100")
            print(f"\n详细结果请查看: {output_dir}")
    else:
        all_results = system.process_folder(INPUT_PATH, use_damping=True, extensions=EXTENSIONS)
        if all_results:
            summary_path = output_dir / f"batch_summary_{timestamp}.txt"
            stats = system.generate_summary_report(all_results, str(summary_path))
            csv_path = output_dir / f"batch_results_{timestamp}.csv"
            system.save_results_to_csv(all_results, str(csv_path))
            if not NO_VISUALIZATION:
                for img_path, result in all_results.items():
                    if 'error' not in result:
                        img_name = Path(img_path).stem
                        vis_path = output_dir / f"batch_visualization_{img_name}_{timestamp}.png"
                        system.visualize_results(
                            result, str(vis_path),
                            f"Auto-Exposure Result - {img_name}"
                        )
            print(f"\n{'=' * 60}")
            print("批量处理完成!")
            print(f"{'=' * 60}")
            print(f"处理图像总数: {stats['total_images']}")
            print(f"收敛图像数量: {stats['converged_images']} ({stats['convergence_rate']:.1f}%)")
            print(f"平均迭代次数: {stats['avg_iterations']:.1f}")
            print(f"平均亮度误差: {stats['avg_brightness_error']:.2f}")
            print(f"平均最终BER: {stats['avg_ber']:.6f}")
            print(f"平均综合评分: {stats['avg_score']:.1f}/100")
            print(f"\n详细结果请查看: {output_dir}")


# ========================= 主程序 =========================
def main() -> None:
    """主函数，根据配置运行自动曝光处理"""
    exposure_min_val = parse_fraction(EXPOSURE_MIN)
    exposure_max_val = parse_fraction(EXPOSURE_MAX)
    initial_exposure_val = parse_fraction(INITIAL_EXPOSURE) if INITIAL_EXPOSURE is not None else None

    config = {
        'target_brightness': TARGET_BRIGHTNESS,
        'brightness_tolerance': BRIGHTNESS_TOLERANCE,
        'exposure_min': exposure_min_val,
        'exposure_max': exposure_max_val,
        'gain_min': GAIN_MIN,
        'gain_max': GAIN_MAX,
        'exposure_strategy': EXPOSURE_STRATEGY,
        'debug_mode': DEBUG,
        'roi': ROI,
        'max_iterations': MAX_ITERATIONS,
        'min_iterations': MIN_ITERATIONS,
        'initial_exposure': initial_exposure_val,
        'initial_iso': INITIAL_ISO
    }

    system = AutoExposureSystem(config)
    system.algorithm.max_iters = MAX_ITERATIONS
    system.algorithm.min_iters = MIN_ITERATIONS

    input_path = Path(INPUT_PATH)
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {INPUT_PATH}")
        sys.exit(1)

    is_file = input_path.is_file()
    is_dir = input_path.is_dir()

    if BATCH_MODE:
        mode = 'batch'
    elif is_file:
        mode = 'single'
    elif is_dir:
        mode = 'batch'
    else:
        print(f"错误: 无法确定输入类型: {INPUT_PATH}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print("自动曝光系统配置（误码率最小化）")
    print(f"{'=' * 60}")
    print(f"输入路径: {INPUT_PATH}")
    print(f"处理模式: {'批量处理' if mode == 'batch' else '单图像处理'}")
    print(f"目标亮度: {TARGET_BRIGHTNESS} ± {BRIGHTNESS_TOLERANCE}")
    print(f"曝光时间范围: {system.algorithm.format_exposure_time(exposure_min_val)} "
          f"到 {system.algorithm.format_exposure_time(exposure_max_val)}")
    print(f"增益范围: {GAIN_MIN:.1f} 到 {GAIN_MAX:.1f}")
    if initial_exposure_val:
        print(f"初始曝光时间: {system.algorithm.format_exposure_time(initial_exposure_val)}")
    if INITIAL_ISO:
        print(f"初始感光度: {INITIAL_ISO:.0f}")
    print(f"曝光策略: {EXPOSURE_STRATEGY}")
    print(f"最大迭代次数: {MAX_ITERATIONS}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'=' * 60}")

    if mode == 'single':
        _run_single_mode(system, timestamp, output_dir)
    else:  # batch mode
        _run_batch_mode(system, timestamp, output_dir)


if __name__ == "__main__":
    main()