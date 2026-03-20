"""
ROI 工具函数
从 data_acquisition.py 分离，可被任意模块复用。

提供三种 ROI 掩码生成策略：
  CENTER          — 图像中心固定矩形
  AUTO_BRIGHTNESS — 最亮区域自动检测
  SYNC_BASED      — 基于 OOK 同步头精确定位数据包区域
"""

import numpy as np


def create_center_roi_mask(image: np.ndarray, roi_size: int = 300) -> np.ndarray:
    """
    在图像中心创建 roi_size × roi_size 的 ROI 掩码。

    Args:
        image:    输入图像（2D 或 3D numpy 数组）
        roi_size: ROI 边长（像素）

    Returns:
        与 image 同尺寸的 uint8 掩码（ROI=1，其余=0）
    """
    import cv2  # lazy import，不影响无 cv2 环境的导入
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
    自动找最亮区域作为 ROI。

    自适应阈值：取图像最大灰度值的 70% 作为阈值（至少为 10）。
    检测到的区域会被扩展到至少 roi_size × roi_size。

    Args:
        image:    输入图像（灰度 2D 或彩色 3D numpy 数组）
        roi_size: ROI 最小边长

    Returns:
        与 image 同尺寸的 uint8 掩码
    """
    import cv2

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = max(int(float(gray.max()) * 0.7), 10)
    _, thresh_img = cv2.threshold(gray, adaptive_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
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

    # 无明显亮区时退化为滑窗搜索
    h, w = gray.shape
    roi_w = min(roi_size, w)
    roi_h = min(roi_size, h)
    max_mean, best_x, best_y = -1, 0, 0
    step = max(roi_size // 4, 10)
    for sy in range(0, h - roi_h + 1, step):
        for sx in range(0, w - roi_w + 1, step):
            m = float(np.mean(gray[sy:sy + roi_h, sx:sx + roi_w]))
            if m > max_mean:
                max_mean, best_x, best_y = m, sx, sy
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[best_y:best_y + roi_h, best_x:best_x + roi_w] = 1
    return mask


def create_sync_based_roi_mask(image: np.ndarray, demod_config: dict = None) -> np.ndarray:
    """
    基于 OOK 同步头检测创建精确的数据包 ROI 掩码。

    Args:
        image:        输入图像（BGR 或灰度）
        demod_config: 可选解调配置字典

    Returns:
        与 image 同尺寸的 uint8 掩码
    """
    from ..demodulation import OOKDemodulator
    demodulator = OOKDemodulator(config=demod_config)
    return demodulator.get_packet_roi_mask(image)


def compute_roi_stats(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    给定 ROI 掩码，计算灰度统计量。

    Args:
        image: 输入图像（灰度或彩色）
        mask:  uint8 掩码（1=ROI，0=背景）

    Returns:
        包含 mean, std, min, max, median, saturated_ratio 的字典
    """
    import cv2
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = gray[mask == 1].astype(np.float64)
    if pixels.size == 0:
        pixels = gray.flatten().astype(np.float64)
    return {
        "mean": float(np.mean(pixels)),
        "std": float(np.std(pixels)),
        "min": float(np.min(pixels)),
        "max": float(np.max(pixels)),
        "median": float(np.median(pixels)),
        "num_pixels": len(pixels),
        "saturated_ratio": float(np.sum(pixels >= 255) / len(pixels)),
    }
