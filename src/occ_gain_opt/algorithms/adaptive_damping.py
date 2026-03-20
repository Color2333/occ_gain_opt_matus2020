"""
自适应阻尼算法 (完整 5 态状态机)

移植为 AlgorithmBase 接口，完整保留原 5 态逻辑：
  I   初始化规整
  II  比例法快速单向收敛
  III 局部线性拟合收敛
  IV  增益单向收敛
  V   增益夹逼收敛

与 gain_advisor.py 中的简化单步版本不同，本实现维护完整状态机跨调用。
每次 compute_next_params() 对应处理一帧（一轮实验），
状态跨调用持久（reset() 重置）。
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import CameraParams
from .base import AlgorithmBase


class AdaptiveDampingAlgorithm(AlgorithmBase):
    """
    自适应阻尼自动曝光算法（以最小化误码率为目标）。
    可同时调整曝光时间和 ISO（增益）。
    """

    name = "adaptive_damping"
    description = "Ma自适应阻尼: 5态状态机，可调曝光+ISO，以BER最小化为目标"

    def __init__(
        self,
        target_brightness: float = 125.0,
        brightness_tolerance: float = 5.0,
        exposure_min_us: float = 1.0,       # µs
        exposure_max_us: float = 1000.0,    # µs
        iso_min: float = 30.0,
        iso_max: float = 10000.0,
        strategy: str = "exposure_priority",
        use_damping: bool = True,
        debug: bool = False,
    ) -> None:
        """
        Args:
            target_brightness:  目标亮度（0–255 灰度均值）
            brightness_tolerance: 收敛容差
            exposure_min_us:    最小曝光时间（µs）
            exposure_max_us:    最大曝光时间（µs）
            iso_min:            最小 ISO
            iso_max:            最大 ISO
            strategy:           "exposure_priority" 或 "gain_priority"
            use_damping:        是否使用自适应阻尼
            debug:              是否打印调试信息
        """
        self.target_br = target_brightness
        self.tolerance = brightness_tolerance
        self.exp_min = exposure_min_us * 1e-6   # 内部使用秒
        self.exp_max = exposure_max_us * 1e-6
        self.gain_min = iso_min / 100.0         # 内部使用线性增益
        self.gain_max = iso_max / 100.0
        self.strategy = strategy
        self.use_damping = use_damping
        self.debug = debug
        self.iso_base = 100.0

        # 当前状态
        self._current_exp: Optional[float] = None
        self._current_gain: Optional[float] = None
        self._current_l: Optional[float] = None

        # 状态机
        self._state = "I"
        self._refs: Dict[str, Dict[str, Optional[float]]] = {
            s: {"L": None, "E": None, "G": None}
            for s in ("I", "II", "III", "IV", "V")
        }
        self._state_ii_cnt = 0
        self._state_iii_cnt = 0
        self._state_iv_cnt = 0
        self._converged = False

        # 历史
        self._brightness_hist: List[float] = []
        self._exposure_hist: List[float] = []
        self._gain_hist: List[float] = []
        self._damping_hist: List[float] = []
        self._state_hist: List[str] = []
        self._iteration = 0

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    def _iso_to_gain(self, iso: float) -> float:
        return iso / self.iso_base

    def _gain_to_iso(self, gain: float) -> float:
        return self.iso_base * gain

    def _need_adjust(self, brightness: float) -> bool:
        return abs(brightness - self.target_br) > self.tolerance

    # ── 算法1：比例法 ─────────────────────────────────────────────────────────

    def _proportional(
        self,
        l_curr: float,
        e_curr: Optional[float] = None,
        g_curr: Optional[float] = None,
    ) -> Tuple[float, float]:
        if l_curr < 0.1:
            l_curr = 0.1
        r = float(np.clip(self.target_br / l_curr, 0.1, 10.0))
        if e_curr is not None:
            e_new = float(np.clip(r * e_curr, self.exp_min, self.exp_max))
            return e_new, g_curr if g_curr is not None else self._current_gain
        if g_curr is not None:
            g_new = float(np.clip(r * g_curr, self.gain_min, self.gain_max))
            return (e_curr if e_curr is not None else self._current_exp), g_new
        return self._current_exp, self._current_gain

    # ── 算法2：局部线性拟合 ───────────────────────────────────────────────────

    def _linear_fitting(
        self,
        l1: float, e1: float, g1: float,
        l2: float, e2: float, g2: float,
    ) -> Tuple[float, float]:
        if abs(l2 - l1) < 1e-6:
            return self._proportional((l1 + l2) / 2, (e1 + e2) / 2, (g1 + g2) / 2)
        if abs(e2 - e1) > 1e-10 and abs(g2 - g1) < 1e-6:
            e_new = e1 + (e2 - e1) * (self.target_br - l1) / (l2 - l1)
            e_new = float(np.clip(e_new, self.exp_min, self.exp_max))
            return e_new, g1
        if abs(g2 - g1) > 1e-6 and abs(e2 - e1) < 1e-10:
            g_new = g1 + (g2 - g1) * (self.target_br - l1) / (l2 - l1)
            g_new = float(np.clip(g_new, self.gain_min, self.gain_max))
            return e1, g_new
        return self._proportional((l1 + l2) / 2, (e1 + e2) / 2, (g1 + g2) / 2)

    # ── 自适应阻尼 ────────────────────────────────────────────────────────────

    def _calc_damping(self, l_prev: float, l_curr: float) -> float:
        if l_prev < 1.0 or l_curr < 1.0:
            return 0.3
        lam = min(l_prev / l_curr, l_curr / l_prev)
        d_base = 1.0 - lam
        err = abs(l_curr - self.target_br)
        d_factor = 0.5 if err > 50 else (0.7 if err > 20 else (0.9 if err > 10 else 1.0))
        return float(np.clip(d_base * d_factor, 0.0, 1.0))

    def _apply_damping(
        self,
        e_old: float, g_old: float,
        e_new: float, g_new: float,
        d: float,
    ) -> Tuple[float, float]:
        if abs(e_new - e_old) / max(e_old, 1e-10) < 0.001:
            e_new = e_old * 1.001 if e_new >= e_old else e_old * 0.999
        if abs(g_new - g_old) / max(g_old, 1e-10) < 0.001:
            g_new = g_old * 1.001 if g_new >= g_old else g_old * 0.999
        e_act = float(np.clip(d * e_old + (1 - d) * e_new, self.exp_min, self.exp_max))
        g_act = float(np.clip(d * g_old + (1 - d) * g_new, self.gain_min, self.gain_max))
        return e_act, g_act

    # ── 5 态执行函数 ──────────────────────────────────────────────────────────

    def _state_i(self) -> Tuple[float, float]:
        """状态I：初始化规整"""
        if self.strategy in ("exposure_priority", "exposure_only"):
            g_new = self.gain_min
            if self._current_gain > self.gain_min:
                e_new = (self._current_gain / self.gain_min) * self._current_exp
            else:
                e_new = self._current_exp * 1.5
        else:
            e_new = self.exp_min
            if self._current_exp > self.exp_min:
                g_new = (self._current_exp / self.exp_min) * self._current_gain
            else:
                g_new = self._current_gain * 1.5
        e_new = float(np.clip(e_new, self.exp_min, self.exp_max))
        g_new = float(np.clip(g_new, self.gain_min, self.gain_max))
        self._refs["I"] = {"L": self._current_l, "E": self._current_exp, "G": self._current_gain}
        return e_new, g_new

    def _state_ii(self) -> Tuple[float, float]:
        """状态II：比例法单向收敛"""
        self._state_ii_cnt += 1
        if self.strategy in ("exposure_priority", "exposure_only"):
            e_new, _ = self._proportional(self._current_l, self._current_exp, None)
            g_new = self._current_gain
        else:
            _, g_new = self._proportional(self._current_l, None, self._current_gain)
            e_new = self._current_exp
        if abs(e_new - self._current_exp) / max(self._current_exp, 1e-10) < 0.01:
            e_new = self._current_exp * 1.05 if e_new >= self._current_exp else self._current_exp * 0.95
        if abs(g_new - self._current_gain) / max(self._current_gain, 1e-10) < 0.01:
            g_new = self._current_gain * 1.05 if g_new >= self._current_gain else self._current_gain * 0.95
        return e_new, g_new

    def _state_iii(self) -> Tuple[float, float]:
        """状态III：局部线性拟合收敛"""
        self._state_iii_cnt += 1
        r = self._refs
        if None in (r["I"]["L"], r["I"]["E"], r["I"]["G"], r["II"]["L"], r["II"]["E"], r["II"]["G"]):
            return self._state_ii()
        l1, e1, g1 = r["I"]["L"], r["I"]["E"], r["I"]["G"]
        l2, e2, g2 = r["II"]["L"], r["II"]["E"], r["II"]["G"]
        if (l1 - self.target_br) * (l2 - self.target_br) > 0:
            return self._state_ii()
        return self._linear_fitting(l1, e1, g1, l2, e2, g2)

    def _state_iv(self) -> Tuple[float, float]:
        """状态IV：增益单向收敛"""
        self._state_iv_cnt += 1
        e_new = self._current_exp
        _, g_new = self._proportional(self._current_l, None, self._current_gain)
        if abs(g_new - self._current_gain) / max(self._current_gain, 1e-10) < 0.01:
            g_new = self._current_gain * 1.05 if g_new >= self._current_gain else self._current_gain * 0.95
        return e_new, g_new

    def _state_v(self) -> Tuple[float, float]:
        """状态V：增益夹逼收敛"""
        e_new = self._current_exp
        r = self._refs
        if None in (r["III"]["L"], r["III"]["E"], r["III"]["G"], r["IV"]["L"], r["IV"]["E"], r["IV"]["G"]):
            _, g_new = self._proportional(self._current_l, None, self._current_gain)
            return e_new, g_new
        l1, e1, g1 = r["III"]["L"], r["III"]["E"], r["III"]["G"]
        l2, e2, g2 = r["IV"]["L"], r["IV"]["E"], r["IV"]["G"]
        if (l1 - self.target_br) * (l2 - self.target_br) > 0:
            _, g_new = self._proportional(self._current_l, None, self._current_gain)
            return e_new, g_new
        _, g_new = self._linear_fitting(l1, e1, g1, l2, e2, g2)
        return e_new, g_new

    # ── 状态转移 ──────────────────────────────────────────────────────────────

    def _update_state(self) -> None:
        if self._state == "I":
            self._state = "II"
        elif self._state == "II":
            l_ref = self._refs["I"]["L"]
            l_curr = self._current_l
            if l_ref is not None and l_curr is not None:
                if (l_ref < self.target_br < l_curr) or (l_curr < self.target_br < l_ref):
                    self._refs["II"] = {"L": l_curr, "E": self._current_exp, "G": self._current_gain}
                    self._state = "III"
                else:
                    self._refs["II"] = {"L": l_curr, "E": self._current_exp, "G": self._current_gain}
                    if self._state_ii_cnt >= 8:
                        self._state = "III"
        elif self._state == "III":
            if self._state_iii_cnt >= 5:
                self._refs["III"] = {"L": self._current_l, "E": self._current_exp, "G": self._current_gain}
                self._state = "IV"
            else:
                self._refs["III"] = {"L": self._current_l, "E": self._current_exp, "G": self._current_gain}
        elif self._state == "IV":
            l_ref = self._refs["III"]["L"]
            l_curr = self._current_l
            if l_ref is not None and l_curr is not None:
                if (l_ref < self.target_br < l_curr) or (l_curr < self.target_br < l_ref):
                    self._refs["IV"] = {"L": l_curr, "E": self._current_exp, "G": self._current_gain}
                    self._state = "V"
                else:
                    self._refs["IV"] = {"L": l_curr, "E": self._current_exp, "G": self._current_gain}
                    if self._state_iv_cnt >= 8:
                        self._state = "V"
        elif self._state == "V":
            self._refs["V"] = {"L": self._current_l, "E": self._current_exp, "G": self._current_gain}

    # ── 主接口 ────────────────────────────────────────────────────────────────

    def compute_next_params(
        self,
        current_params: CameraParams,
        roi_brightness: float,
        ber: Optional[float] = None,
    ) -> CameraParams:
        # 初始化内部状态（首次调用）
        if self._current_exp is None:
            self._current_exp = current_params.exposure_s
            self._current_gain = self._iso_to_gain(current_params.iso)
            self._current_l = roi_brightness
            self._brightness_hist.append(roi_brightness)
            self._exposure_hist.append(self._current_exp)
            self._gain_hist.append(self._current_gain)
            self._damping_hist.append(0.0)
            self._state_hist.append(self._state)
        else:
            self._current_l = roi_brightness

        if self.debug:
            iso_curr = self._gain_to_iso(self._current_gain)
            print(f"  [Ma iter {self._iteration}] L={roi_brightness:.1f} "
                  f"state={self._state} ISO={iso_curr:.0f} "
                  f"exp={self._current_exp * 1e6:.2f}µs")

        # 检查收敛（只在迭代 ≥ 1 次后才有效）
        if self._iteration > 0 and not self._need_adjust(roi_brightness):
            self._converged = True
            return CameraParams(
                iso=self._gain_to_iso(self._current_gain),
                exposure_us=self._current_exp * 1e6,
            )

        # 执行当前状态
        dispatch = {"I": self._state_i, "II": self._state_ii,
                    "III": self._state_iii, "IV": self._state_iv, "V": self._state_v}
        e_new, g_new = dispatch[self._state]()
        e_new = float(np.clip(e_new, self.exp_min, self.exp_max))
        g_new = float(np.clip(g_new, self.gain_min, self.gain_max))

        # 阻尼
        if self.use_damping and self._brightness_hist:
            d = self._calc_damping(self._brightness_hist[-1], roi_brightness)
            e_act, g_act = self._apply_damping(
                self._current_exp, self._current_gain, e_new, g_new, d
            )
        else:
            d = 0.0
            e_act, g_act = e_new, g_new

        # 防止参数停止变化
        e_ch = abs(e_act - self._current_exp) / max(self._current_exp, 1e-10)
        g_ch = abs(g_act - self._current_gain) / max(self._current_gain, 1e-10)
        if e_ch < 0.001 and g_ch < 0.001 and self._iteration > 5:
            if e_act <= self._current_exp:
                e_act = self._current_exp * 1.05
            else:
                e_act = self._current_exp * 0.95
            e_act = float(np.clip(e_act, self.exp_min, self.exp_max))

        # 更新内部状态
        self._current_exp = e_act
        self._current_gain = g_act
        self._update_state()

        self._brightness_hist.append(roi_brightness)
        self._exposure_hist.append(e_act)
        self._gain_hist.append(g_act)
        self._damping_hist.append(d)
        self._state_hist.append(self._state)
        self._iteration += 1

        return CameraParams(
            iso=self._gain_to_iso(g_act),
            exposure_us=e_act * 1e6,
        )

    def reset(self) -> None:
        self._current_exp = None
        self._current_gain = None
        self._current_l = None
        self._state = "I"
        self._refs = {s: {"L": None, "E": None, "G": None} for s in ("I", "II", "III", "IV", "V")}
        self._state_ii_cnt = 0
        self._state_iii_cnt = 0
        self._state_iv_cnt = 0
        self._converged = False
        self._brightness_hist.clear()
        self._exposure_hist.clear()
        self._gain_hist.clear()
        self._damping_hist.clear()
        self._state_hist.clear()
        self._iteration = 0

    def is_converged(self) -> bool:
        return self._converged

    def get_history(self) -> Dict[str, Any]:
        return {
            "brightness": list(self._brightness_hist),
            "exposure_us": [e * 1e6 for e in self._exposure_hist],
            "iso": [self._gain_to_iso(g) for g in self._gain_hist],
            "damping": list(self._damping_hist),
            "state": list(self._state_hist),
            "iterations": self._iteration,
        }
