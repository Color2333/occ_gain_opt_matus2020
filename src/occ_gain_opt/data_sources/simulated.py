"""
仿真数据源
基于 DataAcquisition 的合成图像生成，用于仿真实验。
"""

import numpy as np

from ..config import CameraConfig, CameraParams, ExperimentConfig
from .base import DataSource


class SimulatedDataSource(DataSource):
    """
    仿真数据源：根据当前相机参数和场景参数生成合成图像。
    set_params() 立即更新内部参数，下一次 get_frame() 反映新参数。
    """

    def __init__(
        self,
        led_intensity: float = 127.5,   # LED 强度（0–255）
        background_light: float = 50.0,
        noise_std: float = ExperimentConfig.NOISE_STD,
        initial_params: CameraParams = None,
        width: int = CameraConfig.IMAGE_WIDTH,
        height: int = CameraConfig.IMAGE_HEIGHT,
    ) -> None:
        self.led_intensity = led_intensity
        self.background_light = background_light
        self.noise_std = noise_std
        self._params = initial_params or CameraParams(iso=100.0, exposure_us=27.9)
        self._width = width
        self._height = height

    @property
    def current_params(self) -> CameraParams:
        return self._params

    def set_params(self, params: CameraParams) -> None:
        self._params = params

    def get_frame(self) -> np.ndarray:
        """生成一帧合成图像（灰度）"""
        gain_linear = 10 ** (self._params.gain_db / 20.0)
        image = np.full((self._height, self._width), self.background_light, dtype=np.float32)

        cx, cy = self._width // 2, self._height // 2
        radius = 50
        yy, xx = np.ogrid[:self._height, :self._width]
        led_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

        led_base = (self.led_intensity / 255.0) * 40.0
        image[led_mask] = self.background_light + led_base * gain_linear

        if self.noise_std > 0:
            image += np.random.normal(0, self.noise_std, image.shape).astype(np.float32)

        return np.clip(image, 0, 255).astype(np.uint8)
