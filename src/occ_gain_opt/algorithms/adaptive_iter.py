"""
自适应迭代增益优化算法 (带学习率 α)

G_{k+1} = G_k + α × 20·log10(Y_target / Y_k)

学习率 α 推荐范围: 0.3–0.7，默认 0.5。
单步变化量限制为 ±step_max_db (dB)，默认 5 dB。
"""

from typing import Dict, List, Optional

import numpy as np

from ..config import CameraParams
from .base import AlgorithmBase


class AdaptiveIterAlgorithm(AlgorithmBase):
    """
    带学习率的迭代增益优化算法。
    仅调整 ISO（增益），曝光时间保持不变。
    """

    name = "adaptive_iter"
    description = "自适应迭代: G_next = G_curr + α×20·log10(Y_target/Y_curr)"

    def __init__(
        self,
        alpha: float = 0.5,
        target_gray: float = 242.25,
        step_max_db: float = 5.0,
        gain_db_min: float = -10.46,
        gain_db_max: float = 40.0,
    ) -> None:
        """
        Args:
            alpha:        学习率 (0.3–0.7 推荐)
            target_gray:  目标灰度值
            step_max_db:  单步最大变化量 (dB)
            gain_db_min:  增益下限
            gain_db_max:  增益上限
        """
        self.alpha = alpha
        self.target_gray = target_gray
        self.step_max_db = step_max_db
        self.gain_db_min = gain_db_min
        self.gain_db_max = gain_db_max
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
        delta_db = float(np.clip(delta_db, -self.step_max_db, self.step_max_db))
        g_next_db = float(np.clip(g_db + self.alpha * delta_db,
                                  self.gain_db_min, self.gain_db_max))
        iso_next = 100.0 * 10.0 ** (g_next_db / 20.0)

        self._history.append({
            "iso_prev": current_params.iso,
            "gain_db_prev": g_db,
            "roi_brightness": roi_brightness,
            "delta_db": delta_db,
            "alpha": self.alpha,
            "gain_db_next": g_next_db,
            "iso_next": iso_next,
        })
        return CameraParams(iso=iso_next, exposure_us=current_params.exposure_us)

    def reset(self) -> None:
        self._history.clear()

    def is_converged(self) -> bool:
        if not self._history:
            return False
        last = self._history[-1]
        return abs(last["roi_brightness"] - self.target_gray) < 5.0

    def get_history(self) -> Dict:
        return {"alpha": self.alpha, "steps": list(self._history)}
