"""
算法基类 — AlgorithmBase ABC
所有增益优化算法的统一接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from ..config import CameraParams


class AlgorithmBase(ABC):
    """增益优化算法抽象基类"""

    name: str = ""           # 算法注册名（子类必须设置）
    description: str = ""   # 算法描述

    @abstractmethod
    def compute_next_params(
        self,
        current_params: CameraParams,
        roi_brightness: float,
        ber: Optional[float] = None,
    ) -> CameraParams:
        """
        根据当前相机参数和 ROI 亮度计算下一步参数。

        Args:
            current_params: 当前相机参数（ISO + 曝光时间）
            roi_brightness: 当前帧 ROI 平均灰度值（0–255）
            ber: 当前帧误码率（可选；部分算法不使用）

        Returns:
            建议的下一步相机参数
        """
        ...

    def reset(self) -> None:
        """重置算法内部状态（迭代历史等）"""

    def is_converged(self) -> bool:
        """判断算法是否已收敛"""
        return False

    def get_history(self) -> Dict:
        """返回迭代历史（格式由子类定义）"""
        return {}
