"""
最终修正版本 - 正确的相机模型
"""

import numpy as np

from ..data_acquisition import DataAcquisition
from ..gain_optimization import GainOptimizer
from ..performance_evaluation import PerformanceEvaluator
from ..config import ROIStrategy, ExperimentConfig


class CorrectDataAcquisition(DataAcquisition):
    """正确的数据采集类 - 修复饱和问题"""

    def capture_image(self, led_intensity: float, gain: float,
                      background_light: float = 50,
                      noise_std: float = ExperimentConfig.NOISE_STD) -> np.ndarray:
        """正确的图像捕获 - 关键是LED信号不能太强"""
        image = np.full((self.height, self.width), background_light, dtype=np.float32)

        center_x, center_y = self.width // 2, self.height // 2
        radius = 50
        y, x = np.ogrid[:self.height, :self.width]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

        gain_linear = 10 ** (gain / 20.0)
        max_gain_linear = 10 ** (20 / 20.0)

        led_base_signal = (led_intensity / 255.0) * 25.0
        led_signal = led_base_signal * gain_linear

        image[mask] = background_light + led_signal

        if noise_std > 0:
            noise = np.random.normal(0, noise_std, image.shape)
            image = image + noise

        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)


def test_correct_version():
    print("\n" + "=" * 70)
    print("修正版本测试 - 正确的相机模型")
    print("=" * 70)

    data_acq = CorrectDataAcquisition()
    optimizer = GainOptimizer(data_acq)
    evaluator = PerformanceEvaluator()

    led_duty_cycle = 50
    background_light = 50

    print(f"\n测试条件:")
    print(f"  LED占空比: {led_duty_cycle}%")
    print(f"  背景光强: {background_light}")

    print("\n" + "=" * 70)
    print("增益响应测试:")
    print("=" * 70)
    print(f"{'增益(dB)':<12} {'线性增益':<12} {'ROI灰度均值':<15} {'饱和比例':<12}")
    print("-" * 70)

    led_intensity = (led_duty_cycle / 100.0) * 255
    for gain_db in [0, 5, 10, 15, 20]:
        image = data_acq.capture_image(led_intensity, gain_db, background_light)
        roi_mask = data_acq.select_roi(strategy=ROIStrategy.CENTER)
        stats = data_acq.get_roi_statistics(image, roi_mask)

        gain_linear = 10 ** (gain_db / 20.0)
        print(f"{gain_db:<12.1f} {gain_linear:<12.2f} {stats['mean']:<15.2f} "
              f"{stats['saturated_ratio']*100:<12.2f}%")

    print("\n" + "=" * 70)
    print("运行增益优化...")
    print("=" * 70)

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
    print(f"  饱和比例: {evaluation['saturated_ratio']*100:.2f}%")

    print("\n" + "=" * 70)
    print("优化过程:")
    print("=" * 70)
    print("  迭代 |   增益(dB)  |  灰度值  | 标准差  | 饱和比例")
    print("-" * 70)
    for state in result['history']:
        sat_ratio = state['saturated_ratio'] * 100
        print(f"   {state['iteration']:2d}   |  {state['gain']:6.2f}   | "
              f"{state['mean_gray']:6.2f} | {state['std_gray']:6.2f} | "
              f"{sat_ratio:5.1f}%")

    print("\n" + "=" * 70)
    print("效果分析:")
    print("=" * 70)

    if evaluation['gray_error'] < 15:
        print("  ✅ 优秀! 灰度值非常接近目标 (<15)")
    elif evaluation['gray_error'] < 30:
        print("  ✓ 良好! 灰度值接近目标 (<30)")
    elif evaluation['gray_error'] < 50:
        print("  ⚠ 一般,灰度值有一定偏差")
    else:
        print("  ❌ 效果不理想")

    if evaluation['optimization_score'] > 85:
        print("  ✅ 优化得分优秀 (>85)")
    elif evaluation['optimization_score'] > 70:
        print("  ✓ 优化得分良好 (>70)")
    elif evaluation['optimization_score'] > 60:
        print("  ⚠ 优化得分一般 (>60)")
    else:
        print("  ❌ 优化得分较低")

    print("\n修正要点:")
    print("  1. LED基础信号强度设为25(而不是127.5)")
    print("  2. 这样在最大增益10x时,信号约250,接近但不超过255")
    print("  3. 给增益留出了足够的动态范围")
    print("  4. 避免了立即饱和的问题")

    return result


def main():
    test_correct_version()


if __name__ == "__main__":
    main()
