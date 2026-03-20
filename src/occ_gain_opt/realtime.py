"""
实时增益优化接口
用于真实实验装置：根据当前帧返回可设置的相机参数，供外部循环「采集 → 设置参数 → 再采集」使用。

新接口（推荐）:
    from occ_gain_opt.algorithms import get as algo_get
    from occ_gain_opt.config import CameraParams
    algo = algo_get("single_shot")()
    next_params = algo.compute_next_params(current_params, roi_brightness)

旧接口（向后兼容，保留）:
    compute_next_gain(), RealtimeGainController
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import CameraConfig, CameraParams, OptimizationConfig, ROIStrategy
from .data_acquisition import DataAcquisition
from .gain_optimization import AdaptiveGainOptimizer, GainOptimizer


def compute_next_gain(
    current_gain_db: float,
    image: np.ndarray,
    *,
    roi_strategy: str = ROIStrategy.SYNC_BASED,
    roi_manual_coords: Optional[Tuple[int, int, int, int]] = None,
    use_adaptive: bool = True,
    learning_rate: float = 0.5,
    target_gray: float = OptimizationConfig.TARGET_GRAY * OptimizationConfig.SAFETY_FACTOR,
    tolerance_gray: float = 5.0,
    safety_factor: float = OptimizationConfig.SAFETY_FACTOR,
) -> Dict[str, Any]:
    """
    单步计算：根据当前增益和当前帧，返回下一帧应设置的增益（及是否收敛）。

    在真实装置下的典型用法：
        1. 用当前增益拍一帧，得到 image。
        2. 调用本函数： params = compute_next_gain(current_gain_db, image, ...)。
        3. 将相机增益设为 params["gain_db"]（若支持曝光可再设 params.get("exposure_us")）。
        4. 若 params["converged"] 为 True 可停止；否则回到步骤 1。

    Args:
        current_gain_db: 当前相机增益 (dB)。
        image: 当前帧 (BGR 或灰度，由 ROI 逻辑内部处理)。
        roi_strategy: ROI 策略，如 "sync_based" / "auto" / "center"。
        roi_manual_coords: 仅当 roi_strategy=="manual" 时使用 (x, y, w, h)。
        use_adaptive: True 时使用带学习率的迭代步进，False 时使用单次公式。
        learning_rate: 自适应步长 (0.3–0.7 常用)，仅 use_adaptive 时有效。
        target_gray: 目标灰度 (默认 255*0.95≈242.25)。
        tolerance_gray: 与目标灰度差小于此值即视为收敛。
        safety_factor: 目标灰度相对饱和的比例 (默认 0.95)。

    Returns:
        字典，包含：
        - gain_db: 建议设置的增益 (dB)，已限制在 CameraConfig 范围内。
        - converged: 是否已收敛（当前灰度与目标差 < tolerance_gray）。
        - mean_gray: 当前 ROI 平均灰度。
        - target_gray: 使用的目标灰度。
        - error: 当前灰度与目标之差 (mean_gray - target_gray)。
        - roi_mask: ROI 二值掩码 (与 image 同尺寸)，可选用于可视化。
    """
    h, w = image.shape[:2]
    data_acq = DataAcquisition(width=w, height=h)

    if roi_strategy == ROIStrategy.AUTO_BRIGHTNESS:
        roi_mask = data_acq.select_roi(strategy=roi_strategy, image=image)
    elif roi_strategy == ROIStrategy.SYNC_BASED:
        roi_mask = data_acq.select_roi(strategy=roi_strategy, image=image)
    elif roi_strategy == ROIStrategy.MANUAL and roi_manual_coords is not None:
        roi_mask = data_acq.select_roi(
            strategy=roi_strategy, manual_coords=roi_manual_coords
        )
    else:
        roi_mask = data_acq.select_roi(strategy=ROIStrategy.CENTER)

    stats = data_acq.get_roi_statistics(image, roi_mask)
    mean_gray = float(stats["mean"])
    effective_target = target_gray * safety_factor

    if mean_gray <= 0:
        next_gain_db = float(CameraConfig.GAIN_MIN)
    else:
        if use_adaptive:
            opt = AdaptiveGainOptimizer(
                data_acq,
                learning_rate=learning_rate,
                target_gray=OptimizationConfig.TARGET_GRAY,
                safety_factor=safety_factor,
            )
            next_gain_db = opt.calculate_optimal_gain(current_gain_db, mean_gray)
        else:
            opt = GainOptimizer(
                data_acq,
                target_gray=OptimizationConfig.TARGET_GRAY,
                safety_factor=safety_factor,
            )
            next_gain_db = opt.calculate_optimal_gain(current_gain_db, mean_gray)
        next_gain_db = float(
            np.clip(next_gain_db, CameraConfig.GAIN_MIN, CameraConfig.GAIN_MAX)
        )

    error = mean_gray - effective_target
    converged = abs(error) < tolerance_gray

    return {
        "gain_db": next_gain_db,
        "converged": converged,
        "mean_gray": mean_gray,
        "target_gray": effective_target,
        "error": error,
        "roi_mask": roi_mask,
    }


class RealtimeGainController:
    """
    持有一个「当前增益」状态的实时控制器，便于在循环中连续调用。
    每次传入新帧，返回本步建议的增益并更新内部状态（用于下一轮可选地做平滑或限幅）。
    """

    def __init__(
        self,
        initial_gain_db: float = 0.0,
        roi_strategy: str = ROIStrategy.SYNC_BASED,
        use_adaptive: bool = True,
        learning_rate: float = 0.5,
        tolerance_gray: float = 5.0,
        max_iterations: int = 10,
        **kwargs: Any,
    ):
        self.current_gain_db = initial_gain_db
        self.roi_strategy = roi_strategy
        self.use_adaptive = use_adaptive
        self.learning_rate = learning_rate
        self.tolerance_gray = tolerance_gray
        self.max_iterations = max_iterations
        self.extra_kwargs = kwargs
        self.iteration = 0
        self.last_result: Optional[Dict[str, Any]] = None

    def step(self, image: np.ndarray) -> Dict[str, Any]:
        """
        执行一步：根据当前帧计算下一增益，并更新内部 current_gain_db。

        Returns:
            与 compute_next_gain() 相同的字典；若已达 max_iterations 会设 converged=True 以停止。
        """
        if self.iteration >= self.max_iterations:
            return {
                "gain_db": self.current_gain_db,
                "converged": True,
                "mean_gray": self.last_result.get("mean_gray", 0.0)
                if self.last_result
                else 0.0,
                "target_gray": float(
                    OptimizationConfig.TARGET_GRAY * OptimizationConfig.SAFETY_FACTOR
                ),
                "error": 0.0,
                "roi_mask": None,
            }

        result = compute_next_gain(
            self.current_gain_db,
            image,
            roi_strategy=self.roi_strategy,
            use_adaptive=self.use_adaptive,
            learning_rate=self.learning_rate,
            tolerance_gray=self.tolerance_gray,
            **self.extra_kwargs,
        )
        self.current_gain_db = result["gain_db"]
        self.last_result = result
        self.iteration += 1
        return result

    def reset(self, initial_gain_db: Optional[float] = None) -> None:
        """重置迭代次数与当前增益（便于新一次运行）。"""
        self.iteration = 0
        if initial_gain_db is not None:
            self.current_gain_db = initial_gain_db
        self.last_result = None
