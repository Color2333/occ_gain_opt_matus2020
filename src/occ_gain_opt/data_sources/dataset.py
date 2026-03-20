"""
数据集数据源
包装 ExperimentLoader，从 ISO-Texp/ 真实图像数据集读取帧。
set_params() 自动选择参数最接近的图像文件。
"""

import os
from typing import Optional

import cv2
import numpy as np

from ..config import CameraParams
from ..experiment_loader import ExperimentLoader
from .base import DataSource


class DatasetDataSource(DataSource):
    """
    数据集数据源：从真实实验图像数据集（ISO-Texp/）中选取帧。

    set_params(params) 根据 params.iso 和 params.exposure_us
    选择数据集中参数最接近的图像；get_frame() 返回该图像。
    """

    def __init__(
        self,
        dataset_dir: str = "ISO-Texp",
        condition: Optional[str] = None,
        image_type: Optional[str] = None,
    ) -> None:
        """
        Args:
            dataset_dir: 数据集根目录（默认 "ISO-Texp"）
            condition:   条件过滤，如 "bubble_1_2_2"（None = 不过滤）
            image_type:  "ISO" 或 "Texp"（None = 不过滤）
        """
        self._loader = ExperimentLoader(dataset_dir)
        self._images = self._loader.load_all_images(
            condition=condition,
            image_type=image_type,
        )
        if not self._images:
            raise RuntimeError(
                f"数据集 '{dataset_dir}' 中未找到图像（condition={condition}, type={image_type}）"
            )
        # 按 ISO 排序，方便二分查找
        self._images_sorted = sorted(self._images, key=lambda img: img.iso)
        self._current_image = self._images_sorted[0]
        self._current_params = CameraParams(
            iso=self._current_image.iso,
            exposure_us=self._current_image.exposure_time * 1e6,
        )

    @property
    def current_params(self) -> CameraParams:
        return self._current_params

    def set_params(self, params: CameraParams) -> None:
        """选择参数最接近的图像（以 ISO 为主排序指标）"""
        target_iso = params.iso
        best = min(self._images_sorted, key=lambda img: abs(img.iso - target_iso))
        self._current_image = best
        self._current_params = CameraParams(
            iso=best.iso,
            exposure_us=best.exposure_time * 1e6,
        )

    def get_frame(self) -> np.ndarray:
        """加载并返回当前选中图像（BGR uint8）"""
        if not self._current_image.load():
            raise RuntimeError(f"无法加载图像: {self._current_image.filepath}")
        return self._current_image.image

    def list_available(self):
        """返回所有可用图像的 (ISO, exposure_us, filepath) 列表"""
        return [
            (img.iso, img.exposure_time * 1e6, img.filepath)
            for img in self._images_sorted
        ]
