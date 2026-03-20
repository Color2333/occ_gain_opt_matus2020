"""
数据采集模块（兼容层）
使用统一架构，委托给 data_sources/ 层

注意: 此模块为向后兼容保留，新项目请直接使用 data_sources/
"""

from typing import Tuple, Optional

import numpy as np

from .config import CameraConfig, ROIStrategy, ExperimentConfig
from .data_sources.roi import (
    create_center_roi_mask,
    create_auto_roi_mask,
    create_sync_based_roi_mask,
    compute_roi_stats,
)


class DataAcquisition:
    """数据采集类 - 兼容层，使用统一架构"""

    def __init__(self, width: int = CameraConfig.IMAGE_WIDTH,
                 height: int = CameraConfig.IMAGE_HEIGHT):
        """
        初始化数据采集模块

        Args:
            width: 图像宽度
            height: 图像高度
        """
        self.width = width
        self.height = height
        self.roi_mask = None

    def capture_image(self, led_intensity: float, gain: float,
                      background_light: float = 50,
                      noise_std: float = ExperimentConfig.NOISE_STD) -> np.ndarray:
        """
        模拟捕获图像

        Args:
            led_intensity: LED强度 (0-255)
            gain: 相机增益 (dB)
            background_light: 背景光强 (0-255)
            noise_std: 噪声标准差

        Returns:
            捕获的图像 (灰度图)
        """
        import cv2

        image = np.full((self.height, self.width), background_light, dtype=np.float32)

        # LED区域 (中心圆形区域)
        center_x, center_y = self.width // 2, self.height // 2
        radius = 50
        y, x = np.ogrid[:self.height, :self.width]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

        # 增益从dB到线性尺度
        gain_linear = 10 ** (gain / 20.0)

        # LED信号: 保持线性关系,避免立即饱和
        led_base_signal = (led_intensity / 255.0) * 40.0
        led_signal = led_base_signal * gain_linear

        image[mask] = background_light + led_signal

        # 添加高斯噪声
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, image.shape)
            image = image + noise

        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)

    def select_roi(self, strategy: str = ROIStrategy.CENTER,
                   manual_coords: Optional[Tuple[int, int, int, int]] = None,
                   image: Optional[np.ndarray] = None,
                   roi_size: int = 300) -> np.ndarray:
        """
        选择ROI区域 - 委托给 data_sources.roi

        Args:
            strategy: ROI选择策略
            manual_coords: 手动坐标 (x, y, w, h)
            image: 用于自动选择的图像
            roi_size: ROI 边长 (仅 CENTER / AUTO_BRIGHTNESS 使用)

        Returns:
            ROI掩码
        """
        import cv2

        if strategy == ROIStrategy.CENTER:
            placeholder = np.zeros((self.height, self.width), dtype=np.uint8)
            self.roi_mask = create_center_roi_mask(placeholder, roi_size=roi_size)

        elif strategy == ROIStrategy.MANUAL and manual_coords:
            x, y, w, h = manual_coords
            self.roi_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            self.roi_mask[y:y + h, x:x + w] = 1

        elif strategy == ROIStrategy.AUTO_BRIGHTNESS and image is not None:
            self.roi_mask = create_auto_roi_mask(image, roi_size=roi_size)

        elif strategy == ROIStrategy.SYNC_BASED and image is not None:
            self.roi_mask = create_sync_based_roi_mask(image)

        else:
            return self.select_roi(ROIStrategy.CENTER, roi_size=roi_size)

        return self.roi_mask

    def extract_roi_gray_values(self, image: np.ndarray,
                                roi_mask: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """
        提取ROI区域的灰度值

        Args:
            image: 输入图像
            roi_mask: ROI掩码

        Returns:
            (平均灰度值, ROI区域的像素值)
        """
        if roi_mask is None:
            roi_mask = self.roi_mask

        if roi_mask is None:
            gray_values = image.flatten()
        else:
            gray_values = image[roi_mask == 1]

        mean_gray = np.mean(gray_values)
        return mean_gray, gray_values

    def get_roi_statistics(self, image: np.ndarray,
                           roi_mask: Optional[np.ndarray] = None) -> dict:
        """
        获取ROI区域的统计信息 - 委托给 data_sources.roi

        Args:
            image: 输入图像
            roi_mask: ROI掩码

        Returns:
            包含统计信息的字典
        """
        if roi_mask is None:
            # 无掩码时使用整个图像
            roi_mask = np.ones(image.shape[:2], dtype=np.uint8)

        return compute_roi_stats(image, roi_mask)

    def simulate_capture_sequence(self, led_duty_cycle: float,
                                  gains: np.ndarray,
                                  background_light: float = 50,
                                  noise_std: float = ExperimentConfig.NOISE_STD,
                                  roi_strategy: str = ROIStrategy.CENTER) -> list:
        """
        模拟一系列不同增益下的图像捕获

        Args:
            led_duty_cycle: LED占空比 (0-100)
            gains: 增益值数组 (dB)
            background_light: 背景光强
            noise_std: 噪声标准差
            roi_strategy: ROI选择策略

        Returns:
            图像和统计信息列表
        """
        led_intensity = (led_duty_cycle / 100.0) * 255

        results = []
        for gain in gains:
            image = self.capture_image(
                led_intensity, gain, background_light, noise_std=noise_std
            )
            roi_mask = self.select_roi(strategy=roi_strategy, image=image)
            stats = self.get_roi_statistics(image, roi_mask)

            results.append({
                'gain': gain,
                'image': image,
                'stats': stats
            })

        return results
