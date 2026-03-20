"""
单次公式增益优化算法 (Paper Eq. 7)

G_opt(dB) = G_curr(dB) + 20·log10(Y_target / Y_curr)

单步收敛，无需迭代状态。
"""

from typing import Dict, List, Optional

import numpy as np

from ..config import CameraParams
from .base import AlgorithmBase


class SingleShotAlgorithm(AlgorithmBase):
    """
    单次公式增益优化算法。
    仅调整 ISO（增益），曝光时间保持不变。
    单步收敛，无需迭代状态。
    """

    name = "single_shot"
    description = "单次公式: G_opt = G_curr + 20·log10(Y_target/Y_curr)"

    def __init__(
        self,
        target_gray: float = 242.25,
        gain_db_min: float = -10.46,   # ISO 30 ≈ -10.46 dB
        gain_db_max: float = 40.0,     # ISO 10000 ≈ +40 dB
        step_max_db: Optional[float] = None,
    ) -> None:
        """
        Args:
            target_gray:  目标灰度值，默认 255 × 0.95 = 242.25
            gain_db_min:  增益下限 (dB)，对应相机最小 ISO
            gain_db_max:  增益上限 (dB)，对应相机最大 ISO
            step_max_db:  单步最大变化量 (dB)；None 表示不限
        """
        self.target_gray = target_gray
        self.gain_db_min = gain_db_min
        self.gain_db_max = gain_db_max
        self.step_max_db = step_max_db
        self._history: List[Dict] = []

    def compute_next_params(
        self,
        current_params: CameraParams,
        roi_brightness: float,
        ber: Optional[float] = None,
    ) -> CameraParams:
        g_db = current_params.gain_db
        if roi_brightness <= 0:
            return current_params

        delta_db = 20.0 * np.log10(self.target_gray / roi_brightness)
        if self.step_max_db is not None:
            delta_db = float(np.clip(delta_db, -self.step_max_db, self.step_max_db))
        g_next_db = float(np.clip(g_db + delta_db, self.gain_db_min, self.gain_db_max))
        iso_next = 100.0 * 10.0 ** (g_next_db / 20.0)

        self._history.append({
            "iso_prev": current_params.iso,
            "gain_db_prev": g_db,
            "roi_brightness": roi_brightness,
            "delta_db": delta_db,
            "gain_db_next": g_next_db,
            "iso_next": iso_next,
        })
        return CameraParams(iso=iso_next, exposure_us=current_params.exposure_us)

    def reset(self) -> None:
        self._history.clear()

    def is_converged(self) -> bool:
        """当最近一步亮度误差 < 5 时视为收敛"""
        if not self._history:
            return False
        last = self._history[-1]
        return abs(last["roi_brightness"] - self.target_gray) < 5.0

    def get_history(self) -> Dict:
        return {"steps": list(self._history)}
