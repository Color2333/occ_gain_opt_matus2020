"""
增益优化算法模块
实现论文中的模拟增益优化算法
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import OptimizationConfig, CameraConfig, ROIStrategy
from .data_acquisition import DataAcquisition


class GainOptimizer:
    """增益优化器 - 实现论文公式7和优化算法"""

    def __init__(self, data_acquisition: DataAcquisition,
                 target_gray: float = OptimizationConfig.TARGET_GRAY,
                 safety_factor: float = OptimizationConfig.SAFETY_FACTOR):
        """
        初始化增益优化器

        Args:
            data_acquisition: 数据采集模块
            target_gray: 目标灰度值 (默认255)
            safety_factor: 安全因子,防止过饱和
        """
        self.data_acq = data_acquisition
        self.target_gray = target_gray
        self.safety_factor = safety_factor
        self.optimization_history = []

    def calculate_optimal_gain(self, current_gain: float,
                               current_gray: float) -> float:
        """
        计算最优增益 (论文公式7)

        G_opt = G_curr × (Y_target / Y_curr)

        Args:
            current_gain: 当前增益 (dB)
            current_gray: 当前灰度值

        Returns:
            最优增益 (dB)
        """
        if current_gray <= 0:
            return CameraConfig.GAIN_MIN

        effective_target = self.target_gray * self.safety_factor
        gain_linear = 10 ** (current_gain / 20.0)
        optimal_gain_linear = gain_linear * (effective_target / current_gray)
        optimal_gain_db = 20 * np.log10(optimal_gain_linear)

        return float(optimal_gain_db)

    def clamp_gain(self, gain: float) -> float:
        """将增益限制在合法范围内"""
        return float(np.clip(gain, CameraConfig.GAIN_MIN, CameraConfig.GAIN_MAX))

    def optimize_gain(self, led_duty_cycle: float,
                      initial_gain: float = 0.0,
                      background_light: float = 50,
                      max_iterations: int = OptimizationConfig.MAX_ITERATIONS,
                      tolerance: float = OptimizationConfig.TOLERANCE,
                      noise_std: float = 0.0,
                      roi_strategy: str = ROIStrategy.CENTER,
                      roi_manual_coords: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """
        执行增益优化算法

        Args:
            led_duty_cycle: LED占空比
            initial_gain: 初始增益
            background_light: 背景光强
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            noise_std: 噪声标准差 (优化闭环中的噪声)
            roi_strategy: ROI选择策略
            roi_manual_coords: 手动ROI坐标

        Returns:
            优化结果字典
        """
        current_gain = initial_gain
        iteration = 0
        converged = False
        history = []
        roi_mask = None

        led_intensity = (led_duty_cycle / 100.0) * 255

        while iteration < max_iterations and not converged:
            image = self.data_acq.capture_image(
                led_intensity, current_gain, background_light, noise_std=noise_std
            )

            if roi_strategy == ROIStrategy.AUTO_BRIGHTNESS:
                roi_mask = self.data_acq.select_roi(strategy=roi_strategy, image=image)
            elif roi_mask is None:
                roi_mask = self.data_acq.select_roi(
                    strategy=roi_strategy, manual_coords=roi_manual_coords
                )

            stats = self.data_acq.get_roi_statistics(image, roi_mask)
            mean_gray = stats['mean']

            state = {
                'iteration': iteration,
                'gain': current_gain,
                'mean_gray': mean_gray,
                'std_gray': stats['std'],
                'min_gray': stats['min'],
                'max_gray': stats['max'],
                'saturated_ratio': stats['saturated_ratio']
            }
            history.append(state)

            error = abs(mean_gray - self.target_gray * self.safety_factor)
            if error < tolerance:
                converged = True
                break

            optimal_gain = self.calculate_optimal_gain(current_gain, mean_gray)
            optimal_gain = self.clamp_gain(optimal_gain)

            if abs(optimal_gain - current_gain) < tolerance:
                converged = True
                break

            current_gain = optimal_gain
            iteration += 1

        final_image = self.data_acq.capture_image(
            led_intensity, current_gain, background_light, noise_std=noise_std
        )
        if roi_strategy == ROIStrategy.AUTO_BRIGHTNESS:
            roi_mask = self.data_acq.select_roi(strategy=roi_strategy, image=final_image)
        final_stats = self.data_acq.get_roi_statistics(final_image, roi_mask)
        reference_image = self.data_acq.capture_image(
            led_intensity, current_gain, background_light, noise_std=0.0
        )

        result = {
            'optimal_gain': current_gain,
            'final_gray': final_stats['mean'],
            'iterations': iteration + 1,
            'converged': converged,
            'history': history,
            'final_image': final_image,
            'reference_image': reference_image,
            'roi_mask': roi_mask,
            'final_stats': final_stats
        }

        self.optimization_history.append(result)
        return result

    def batch_optimize(self, led_duty_cycles: List[float],
                       background_lights: List[float],
                       initial_gain: float = 0.0,
                       noise_std: float = 0.0,
                       roi_strategy: str = ROIStrategy.CENTER) -> List[Dict]:
        """
        批量优化不同条件下的增益
        """
        results = []
        for led_dc in led_duty_cycles:
            for bg_light in background_lights:
                result = self.optimize_gain(
                    led_duty_cycle=led_dc,
                    initial_gain=initial_gain,
                    background_light=bg_light,
                    noise_std=noise_std,
                    roi_strategy=roi_strategy
                )
                result['led_duty_cycle'] = led_dc
                result['background_light'] = bg_light
                results.append(result)

        return results

    def analyze_optimization_curve(self, result: Dict) -> Dict:
        """分析优化曲线特性"""
        history = result['history']
        gains = [state['gain'] for state in history]
        gray_values = [state['mean_gray'] for state in history]

        if len(gains) > 1:
            gain_changes = [abs(gains[i + 1] - gains[i]) for i in range(len(gains) - 1)]
            avg_change = float(np.mean(gain_changes))
            final_change = float(gain_changes[-1] if gain_changes else 0)
        else:
            avg_change = 0.0
            final_change = 0.0

        gray_std = float(np.std(gray_values[-min(5, len(gray_values)):]))

        analysis = {
            'gain_range': (max(gains) - min(gains)) if gains else 0,
            'average_gain_change': avg_change,
            'final_gain_change': final_change,
            'gray_value_stability': gray_std,
            'convergence_rate': len(history) / result['iterations'] if result['iterations'] > 0 else 0
        }

        return analysis


class AdaptiveGainOptimizer(GainOptimizer):
    """自适应增益优化器 - 改进版本"""

    def __init__(self, data_acquisition: DataAcquisition,
                 learning_rate: float = 0.5,
                 ema_alpha: float = 0.6,
                 min_learning_rate: float = 0.1,
                 **kwargs):
        super().__init__(data_acquisition, **kwargs)
        self.learning_rate = learning_rate
        self.ema_alpha = ema_alpha
        self.min_learning_rate = min_learning_rate

    def calculate_optimal_gain(self, current_gain: float,
                               current_gray: float,
                               momentum: float = 0.0,
                               learning_rate: Optional[float] = None) -> float:
        base_optimal = super().calculate_optimal_gain(current_gain, current_gray)
        delta = base_optimal - current_gain
        # 限制单步更新幅度，提升稳定性
        delta = float(np.clip(delta, -OptimizationConfig.STEP_SIZE_MAX, OptimizationConfig.STEP_SIZE_MAX))
        lr = self.learning_rate if learning_rate is None else learning_rate
        lr = float(np.clip(lr, self.min_learning_rate, self.learning_rate))
        adaptive_optimal = current_gain + lr * delta + momentum
        return float(adaptive_optimal)

    def optimize_gain(self, led_duty_cycle: float,
                      initial_gain: float = 0.0,
                      background_light: float = 50,
                      **kwargs) -> Dict:
        current_gain = initial_gain
        momentum = 0.0
        momentum_decay = 0.9

        iteration = 0
        max_iterations = kwargs.get('max_iterations', OptimizationConfig.MAX_ITERATIONS)
        tolerance = kwargs.get('tolerance', OptimizationConfig.TOLERANCE)
        noise_std = kwargs.get('noise_std', 0.0)
        roi_strategy = kwargs.get('roi_strategy', ROIStrategy.CENTER)
        roi_manual_coords = kwargs.get('roi_manual_coords', None)

        converged = False
        history = []
        roi_mask = None
        ema_gray = None

        led_intensity = (led_duty_cycle / 100.0) * 255

        while iteration < max_iterations and not converged:
            image = self.data_acq.capture_image(
                led_intensity, current_gain, background_light, noise_std=noise_std
            )
            if roi_strategy == ROIStrategy.AUTO_BRIGHTNESS:
                roi_mask = self.data_acq.select_roi(strategy=roi_strategy, image=image)
            elif roi_mask is None:
                roi_mask = self.data_acq.select_roi(
                    strategy=roi_strategy, manual_coords=roi_manual_coords
                )

            stats = self.data_acq.get_roi_statistics(image, roi_mask)
            mean_gray = stats['mean']
            if ema_gray is None:
                ema_gray = mean_gray
            else:
                ema_gray = self.ema_alpha * mean_gray + (1 - self.ema_alpha) * ema_gray

            state = {
                'iteration': iteration,
                'gain': current_gain,
                'mean_gray': mean_gray,
                'ema_gray': ema_gray,
                'momentum': momentum
            }
            history.append(state)

            effective_target = self.target_gray * self.safety_factor
            error = abs(ema_gray - effective_target)
            if error < tolerance:
                converged = True
                break

            old_gain = current_gain
            adaptive_lr = self.learning_rate * (error / effective_target) if effective_target > 0 else self.learning_rate
            adaptive_lr = float(np.clip(adaptive_lr, self.min_learning_rate, self.learning_rate))
            optimal_gain = self.calculate_optimal_gain(
                current_gain, ema_gray, momentum, learning_rate=adaptive_lr
            )
            optimal_gain = self.clamp_gain(optimal_gain)

            momentum = momentum_decay * momentum + \
                      (1 - momentum_decay) * (optimal_gain - old_gain)

            if abs(optimal_gain - current_gain) < tolerance:
                converged = True
                break

            current_gain = optimal_gain
            iteration += 1

        final_image = self.data_acq.capture_image(
            led_intensity, current_gain, background_light, noise_std=noise_std
        )
        if roi_strategy == ROIStrategy.AUTO_BRIGHTNESS:
            roi_mask = self.data_acq.select_roi(strategy=roi_strategy, image=final_image)
        final_stats = self.data_acq.get_roi_statistics(final_image, roi_mask)
        reference_image = self.data_acq.capture_image(
            led_intensity, current_gain, background_light, noise_std=0.0
        )

        result = {
            'optimal_gain': current_gain,
            'final_gray': final_stats['mean'],
            'iterations': iteration + 1,
            'converged': converged,
            'history': history,
            'final_image': final_image,
            'reference_image': reference_image,
            'roi_mask': roi_mask,
            'final_stats': final_stats
        }

        return result
