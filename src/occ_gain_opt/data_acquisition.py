"""
数据采集模块
模拟相机图像采集和ROI灰度值提取
"""

from typing import Tuple, Optional

import cv2
import numpy as np

from .config import CameraConfig, LEDConfig, ROIStrategy, ExperimentConfig


# ---- 公共 ROI mask 工具函数 (可被任意模块复用) ----

def create_center_roi_mask(image: np.ndarray, roi_size: int = 300) -> np.ndarray:
    """
    在图像中心创建 roi_size × roi_size 的 ROI 掩码

    Args:
        image: 输入图像 (2D 或 3D numpy 数组)
        roi_size: ROI 边长 (像素)

    Returns:
        与 image 同尺寸的 uint8 掩码 (ROI=1, 其余=0)
    """
    height, width = image.shape[:2]
    roi_w = min(roi_size, width)
    roi_h = min(roi_size, height)
    x = (width - roi_w) // 2
    y = (height - roi_h) // 2
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y:y + roi_h, x:x + roi_w] = 1
    return mask


def create_auto_roi_mask(image: np.ndarray, roi_size: int = 300) -> np.ndarray:
    """
    自动找最亮区域作为 ROI (优先取最亮连通域, 否则取最亮滑窗)

    阈值自适应: 使用图像最大灰度值的 70% 作为阈值, 至少为 10。
    检测到的区域会被扩展到至少 roi_size × roi_size。

    Args:
        image: 输入图像 (灰度 2D 或彩色 3D numpy 数组)
        roi_size: ROI 最小边长 (像素)

    Returns:
        与 image 同尺寸的 uint8 掩码 (ROI=1, 其余=0)
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 自适应阈值: 取最大灰度值的 70%, 至少为 10
    adaptive_thresh = max(int(float(gray.max()) * 0.7), 10)
    _, thresh_img = cv2.threshold(gray, adaptive_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        # 将检测到的区域扩展到至少 roi_size 大小
        cx, cy = x + w // 2, y + h // 2
        new_w = max(w, roi_size)
        new_h = max(h, roi_size)
        x = max(0, cx - new_w // 2)
        y = max(0, cy - new_h // 2)
        new_w = min(gray.shape[1] - x, new_w)
        new_h = min(gray.shape[0] - y, new_h)
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[y:y + new_h, x:x + new_w] = 1
        return mask

    # 无明显亮区时退化为滑窗搜索最亮区域
    h, w = gray.shape
    roi_w = min(roi_size, w)
    roi_h = min(roi_size, h)
    max_mean = -1
    best_x = 0
    best_y = 0
    step = max(roi_size // 4, 10)
    for sy in range(0, h - roi_h + 1, step):
        for sx in range(0, w - roi_w + 1, step):
            window = gray[sy:sy + roi_h, sx:sx + roi_w]
            mean_val = float(np.mean(window))
            if mean_val > max_mean:
                max_mean = mean_val
                best_x = sx
                best_y = sy
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[best_y:best_y + roi_h, best_x:best_x + roi_w] = 1
    return mask


class DataAcquisition:
    """数据采集类 - 模拟相机图像采集"""

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
        选择ROI区域

        Args:
            strategy: ROI选择策略
            manual_coords: 手动坐标 (x, y, w, h)
            image: 用于自动选择的图像
            roi_size: ROI 边长 (仅 CENTER / AUTO_BRIGHTNESS 使用)

        Returns:
            ROI掩码
        """
        if strategy == ROIStrategy.CENTER:
            # 使用一个临时图像占位 (仅需尺寸信息)
            placeholder = np.zeros((self.height, self.width), dtype=np.uint8)
            self.roi_mask = create_center_roi_mask(placeholder, roi_size=roi_size)

        elif strategy == ROIStrategy.MANUAL and manual_coords:
            x, y, w, h = manual_coords
            self.roi_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            self.roi_mask[y:y + h, x:x + w] = 1

        elif strategy == ROIStrategy.AUTO_BRIGHTNESS and image is not None:
            self.roi_mask = create_auto_roi_mask(image, roi_size=roi_size)

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
        获取ROI区域的统计信息

        Args:
            image: 输入图像
            roi_mask: ROI掩码

        Returns:
            包含统计信息的字典
        """
        mean_gray, gray_values = self.extract_roi_gray_values(image, roi_mask)

        stats = {
            'mean': mean_gray,
            'std': np.std(gray_values),
            'min': np.min(gray_values),
            'max': np.max(gray_values),
            'median': np.median(gray_values),
            'percentile_25': np.percentile(gray_values, 25),
            'percentile_75': np.percentile(gray_values, 75),
            'num_pixels': len(gray_values),
            'saturated_ratio': np.sum(gray_values >= 255) / len(gray_values)
        }

        return stats

    def simulate_capture_sequence(self, led_duty_cycle: float,
                                  gains: np.ndarray,
                                  background_light: float = 50,
                                  noise_std: float = ExperimentConfig.NOISE_STD,
                                  roi_strategy: str = ROIStrategy.CENTER) -> list:
        """
        模拟一系列不同增益下的图像捕获

        Args:
            led_duty_cycle: LED占空比 (0-100)
            gains: 增益值数组
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
