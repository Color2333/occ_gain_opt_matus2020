"""
改进版本的测试 - 更真实的相机模型
"""

import numpy as np

from ..data_acquisition import DataAcquisition
from ..gain_optimization import GainOptimizer
from ..performance_evaluation import PerformanceEvaluator
from ..config import ROIStrategy, ExperimentConfig


class ImprovedDataAcquisition(DataAcquisition):
    """改进的数据采集类 - 更真实的相机模型"""

    def capture_image(self, led_intensity: float, gain: float,
                      background_light: float = 50,
                      noise_std: float = ExperimentConfig.NOISE_STD) -> np.ndarray:
        """改进的图像捕获 - 更真实的相机响应模型"""
        image = np.full((self.height, self.width), background_light, dtype=np.float32)

        center_x, center_y = self.width // 2, self.height // 2
        radius = 50
        y, x = np.ogrid[:self.height, :self.width]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

        if gain <= 10:
            gain_linear = 10 ** (gain / 20.0)
            gain_factor = gain_linear
        else:
            analog_gain = 10 ** (10 / 20.0)
            digital_gain = 10 ** ((gain - 10) / 20.0)
            gain_factor = analog_gain * digital_gain

        led_efficiency = 2.5
        base_led_signal = led_intensity * led_efficiency
        distance_factor = 0.9
        quantum_efficiency = 0.8
        led_signal = base_led_signal * distance_factor * quantum_efficiency * gain_factor

        image[mask] = background_light + led_signal
        image[~mask] = background_light

        shot_noise = np.random.poisson(image.astype(int) * 0.1).astype(float) * 0.1
        read_noise = np.random.normal(0, noise_std, image.shape)
        dark_current = np.random.normal(0, 0.5, image.shape)
        image = image + shot_noise + read_noise + dark_current

        gamma = 2.2
        image = 255 * np.power(image / 255.0, 1.0 / gamma)
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)


def test_improved_version():
    print("\n" + "=" * 70)
    print("改进版本测试 - 更真实的相机模型")
    print("=" * 70)

    data_acq = ImprovedDataAcquisition()
    optimizer = GainOptimizer(data_acq)
    evaluator = PerformanceEvaluator()

    led_duty_cycle = 50
    background_light = 50

    print(f"\n测试条件:")
    print(f"  LED占空比: {led_duty_cycle}%")
    print(f"  背景光强: {background_light}")
    print(f"  使用改进的相机模型")

    print("\n运行增益优化...")
    result = optimizer.optimize_gain(
        led_duty_cycle=led_duty_cycle,
        initial_gain=0.0,
        background_light=background_light,
        noise_std=ExperimentConfig.NOISE_STD,
        roi_strategy=ROIStrategy.CENTER
    )

    evaluation = evaluator.evaluate_optimization_result(
        result,
        reference_image=result.get('reference_image'),
        roi_mask=result.get('roi_mask')
    )

    print("\n" + "=" * 70)
    print("优化结果:")
    print("=" * 70)
    print(f"  最优增益: {result['optimal_gain']:.2f} dB")
    print(f"  最终灰度值: {result['final_gray']:.2f}")
    print(f"  目标灰度值: 255")
    print(f"  灰度误差: {evaluation['gray_error']:.2f}")
    print(f"  误差百分比: {evaluation['gray_error_percent']:.2f}%")
    print(f"  迭代次数: {result['iterations']}")
    print(f"  是否收敛: {'是' if result['converged'] else '否'}")
    print(f"  优化得分: {evaluation['optimization_score']:.2f}/100")

    print(f"\n信号质量:")
    print(f"  均值: {evaluation['signal_quality']['mean']:.2f}")
    print(f"  标准差: {evaluation['signal_quality']['std']:.2f}")
    print(f"  最小值: {evaluation['signal_quality']['min']:.2f}")
    print(f"  最大值: {evaluation['signal_quality']['max']:.2f}")
    print(f"  动态范围: {evaluation['signal_quality']['dynamic_range']:.2f}")
    print(f"  饱和比例: {evaluation['saturated_ratio'] * 100:.2f}%")

    print("\n" + "=" * 70)
    print("优化过程:")
    print("=" * 70)
    print("  迭代 |   增益(dB)  |  灰度值  | 标准差  | 饱和比例")
    print("-" * 70)
    for state in result['history']:
        saturated = (state['max_gray'] >= 254)
        sat_ratio = (state['max_gray'] >= 254) / 255 * 100 if saturated else 0
        print(f"   {state['iteration']:2d}   |  {state['gain']:6.2f}   | "
              f"{state['mean_gray']:6.2f} | {state['std_gray']:6.2f} | "
              f"{sat_ratio:5.1f}%")

    print("\n" + "=" * 70)
    print("效果分析:")
    print("=" * 70)

    if evaluation['gray_error'] < 20:
        print("  ✅ 优秀! 灰度值非常接近目标")
    elif evaluation['gray_error'] < 50:
        print("  ✓ 良好! 灰度值接近目标")
    elif evaluation['gray_error'] < 100:
        print("  ⚠ 一般,灰度值有一定偏差")
    else:
        print("  ❌ 效果不理想,需要进一步调整参数")

    if evaluation['optimization_score'] > 80:
        print("  ✅ 优化得分优秀 (>80)")
    elif evaluation['optimization_score'] > 60:
        print("  ✓ 优化得分良好 (>60)")
    else:
        print("  ⚠ 优化得分一般 (<60)")

    print("\n改进要点:")
    print("  1. 分模拟增益和数字增益两个阶段")
    print("  2. 增强LED强度 (效率系数2.5)")
    print("  3. 考虑距离衰减和量子效率")
    print("  4. 更真实的噪声模型(散粒噪声+读出噪声)")
    print("  5. 添加伽马校正")

    return result


def main():
    test_improved_version()


if __name__ == "__main__":
    main()
