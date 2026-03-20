"""
BER 驱动探索算法

以当前 BER 为探索半径，在 ISO/曝光时间空间随机游走，
寻找最低 BER 对应的相机参数。

核心逻辑:
  exploration_radius = clip(ber * 0.9, r_min, r_max)
  若连续 N 步未改善 → 半径按指数扩张
  每步随机选择调整对象 (ISO / 曝光 / 两者) 和方向 (增大 / 减小)
  只有新 BER 更优时才更新最优参数

BER 为 None 时退化为固定步长单次调整。
"""

import random
from typing import Dict, List, Optional

import numpy as np

from ..config import CameraParams
from .base import AlgorithmBase


class BerExploreAlgorithm(AlgorithmBase):
    """
    BER 驱动随机探索算法。
    同时调整 ISO 和曝光时间，以最小化 BER 为目标。
    """

    name = "ber_explore"
    description = "BER驱动探索: 以当前BER为半径随机游走，连续无改善则扩张搜索范围"

    # 调整目标类型常量
    _ADJ_ISO = "iso"
    _ADJ_EXP = "exposure"
    _ADJ_BOTH = "both"

    def __init__(
        self,
        radius_min: float = 0.05,
        radius_max: float = 0.5,
        radius_expand_rate: float = 1.5,
        no_improve_thresh: int = 10,
        iso_min: float = 30.0,
        iso_max: float = 10000.0,
        exp_min_us: float = 1.0,
        exp_max_us: float = 500.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            radius_min:          最小探索半径（比例，0.05 = 5%）
            radius_max:          最大探索半径（比例，0.5 = 50%）
            radius_expand_rate:  无改善时半径扩张系数
            no_improve_thresh:   触发扩张所需的连续无改善步数
            iso_min / iso_max:   ISO 范围
            exp_min_us / exp_max_us: 曝光时间范围 (µs)
            seed:                随机种子；None 表示不固定
        """
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.radius_expand_rate = radius_expand_rate
        self.no_improve_thresh = no_improve_thresh
        self.iso_min = iso_min
        self.iso_max = iso_max
        self.exp_min_us = exp_min_us
        self.exp_max_us = exp_max_us
        self.seed = seed

        self._best_ber: Optional[float] = None
        self._best_params: Optional[CameraParams] = None
        self._no_improve_count: int = 0
        self._history: List[Dict] = []

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ── 主接口 ────────────────────────────────────────────────────────────────

    def compute_next_params(
        self,
        current_params: CameraParams,
        roi_brightness: float,
        ber: Optional[float] = None,
    ) -> CameraParams:
        """
        根据当前 BER 计算探索步长并随机调整参数。
        若 ber 为 None，使用固定最小半径单步调整。
        """
        # ── 更新最优状态 ──
        if ber is not None:
            if self._best_ber is None or ber < self._best_ber:
                self._best_ber = ber
                self._best_params = current_params
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1

        # ── 计算探索半径 ──
        if ber is not None:
            base_radius = float(np.clip(ber * 0.9, self.radius_min, self.radius_max))
        else:
            base_radius = self.radius_min

        # 连续无改善时指数扩张
        if self._no_improve_count >= self.no_improve_thresh:
            exponent = self._no_improve_count // self.no_improve_thresh
            radius = min(
                self.radius_max,
                base_radius * (self.radius_expand_rate ** exponent),
            )
        else:
            radius = base_radius

        # ── 随机选择调整目标和方向 ──
        adj_type = random.choice([self._ADJ_ISO, self._ADJ_EXP, self._ADJ_BOTH])
        direction = random.choice([1.0, -1.0])

        new_iso = current_params.iso
        new_exp = current_params.exposure_us

        if adj_type in (self._ADJ_ISO, self._ADJ_BOTH):
            new_iso = float(np.clip(
                current_params.iso * (1.0 + direction * radius),
                self.iso_min,
                self.iso_max,
            ))

        if adj_type in (self._ADJ_EXP, self._ADJ_BOTH):
            exp_direction = direction if adj_type == self._ADJ_EXP else random.choice([1.0, -1.0])
            new_exp = float(np.clip(
                current_params.exposure_us * (1.0 + exp_direction * radius),
                self.exp_min_us,
                self.exp_max_us,
            ))

        next_params = CameraParams(iso=new_iso, exposure_us=new_exp)

        self._history.append({
            "iso_prev": current_params.iso,
            "exp_prev": current_params.exposure_us,
            "ber": ber,
            "radius": radius,
            "adj_type": adj_type,
            "direction": direction,
            "iso_next": new_iso,
            "exp_next": new_exp,
            "no_improve_count": self._no_improve_count,
        })

        return next_params

    # ── 辅助接口 ──────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._best_ber = None
        self._best_params = None
        self._no_improve_count = 0
        self._history.clear()
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def is_converged(self) -> bool:
        """BER < 0.01 且连续 5 步无改善视为收敛"""
        return (
            self._best_ber is not None
            and self._best_ber < 0.01
            and self._no_improve_count >= 5
        )

    def get_history(self) -> Dict:
        return {
            "best_ber": self._best_ber,
            "best_params": {
                "iso": self._best_params.iso,
                "exposure_us": self._best_params.exposure_us,
            } if self._best_params else None,
            "steps": list(self._history),
        }

    @property
    def best_params(self) -> Optional[CameraParams]:
        """返回历史最优参数（BER 最低时对应的参数）"""
        return self._best_params
