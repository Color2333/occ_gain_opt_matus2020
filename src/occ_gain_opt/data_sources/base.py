"""
数据源基类 — DataSource ABC
统一仿真、数据集、RTSP 相机的帧获取接口
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..config import CameraParams


class DataSource(ABC):
    """数据源抽象基类"""

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """
        获取一帧图像。

        Returns:
            BGR 或灰度 numpy 数组（uint8）
        """
        ...

    def get_n_frames(self, n: int) -> List[np.ndarray]:
        """
        连续获取 n 帧（默认逐帧调用 get_frame）。
        子类可重写以实现批量采集。
        """
        return [self.get_frame() for _ in range(n)]

    @property
    @abstractmethod
    def current_params(self) -> CameraParams:
        """当前相机参数（ISO + 曝光时间）"""
        ...

    def set_params(self, params: CameraParams) -> None:
        """
        设置相机参数。
        仿真源 / 数据集源默认无操作（或自动映射最近档位）；
        真实相机源覆写此方法发送 ISAPI / V4L2 命令。
        """
