"""
使用示例和测试脚本
演示如何使用增益优化算法
"""

import numpy as np
import matplotlib.pyplot as plt

from .data_acquisition import DataAcquisition
from .gain_optimization import GainOptimizer, AdaptiveGainOptimizer
from .performance_evaluation import PerformanceEvaluator
from .config import ROIStrategy, ExperimentConfig


def example_1_basic_usage():
    """示例1: 基础使用"""
    print("\n" + "=" * 60)
    print("示例1: 基础使用 - 单次增益优化")
    print("=" * 60)

    data_acq = DataAcquisition()
    optimizer = GainOptimizer(data_acq)

    led_duty_cycle = 50
    background_light = 50
    initial_gain = 0.0

    print(f"\n实验条件:")
    print(f"  LED占空比: {led_duty_cycle}%")
    print(f"  背景光强: {background_light}")
    print(f"  初始增益: {initial_gain} dB")

    result = optimizer.optimize_gain(
        led_duty_cycle=led_duty_cycle,
        initial_gain=initial_gain,
        background_light=background_light,
        noise_std=ExperimentConfig.NOISE_STD,
        roi_strategy=ROIStrategy.CENTER
    )

    print(f"\n优化结果:")
    print(f"  最优增益: {result['optimal_gain']:.2f} dB")
    print(f"  最终灰度值: {result['final_gray']:.2f}")
    print(f"  迭代次数: {result['iterations']}")
    print(f"  是否收敛: {result['converged']}")

    return result


def example_2_gain_sweep():
    """示例2: 增益扫描"""
    print("\n" + "=" * 60)
    print("示例2: 增益扫描 - 研究增益对灰度值的影响")
    print("=" * 60)

    data_acq = DataAcquisition()
    led_duty_cycle = 50
    background_light = 50
    gains = np.linspace(0, 20, 21)

    print(f"\n扫描增益范围: {gains[0]:.1f} - {gains[-1]:.1f} dB")

    results = data_acq.simulate_capture_sequence(
        led_duty_cycle, gains, background_light,
        noise_std=ExperimentConfig.NOISE_STD,
        roi_strategy=ROIStrategy.CENTER
    )

    gray_values = [r['stats']['mean'] for r in results]
    max_idx = np.argmax(gray_values)
    optimal_gain = gains[max_idx]
    max_gray = gray_values[max_idx]

    print(f"\n最优工作点:")
    print(f"  增益: {optimal_gain:.2f} dB")
    print(f"  灰度值: {max_gray:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(gains, gray_values, 'b-o', linewidth=2, markersize=6)
    plt.axhline(y=255, color='r', linestyle='--', label='饱和值(255)')
    plt.plot(optimal_gain, max_gray, 'ro', markersize=12,
             label=f'最优点({optimal_gain:.2f}dB, {max_gray:.2f})')
    plt.xlabel('增益 (dB)', fontsize=12)
    plt.ylabel('ROI平均灰度值', fontsize=12)
    plt.title('增益-灰度响应曲线', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/example_2_gain_sweep.png', dpi=150)
    print(f"\n图表已保存: results/plots/example_2_gain_sweep.png")
    plt.close()

    return results


def example_3_algorithm_comparison():
    """示例3: 算法比较"""
    print("\n" + "=" * 60)
    print("示例3: 算法比较 - 基础算法 vs 自适应算法")
    print("=" * 60)

    data_acq = DataAcquisition()
    basic_opt = GainOptimizer(data_acq)
    adaptive_opt = AdaptiveGainOptimizer(data_acq, learning_rate=0.5)
    evaluator = PerformanceEvaluator()

    led_duty_cycle = 50
    background_light = 50

    print(f"\n测试条件:")
    print(f"  LED占空比: {led_duty_cycle}%")
    print(f"  背景光强: {background_light}")

    print("\n运行基础算法...")
    result_basic = basic_opt.optimize_gain(
        led_duty_cycle, 0.0, background_light,
        noise_std=ExperimentConfig.NOISE_STD,
        roi_strategy=ROIStrategy.CENTER
    )

    print("运行自适应算法...")
    result_adaptive = adaptive_opt.optimize_gain(
        led_duty_cycle, 0.0, background_light,
        noise_std=ExperimentConfig.NOISE_STD,
        roi_strategy=ROIStrategy.CENTER
    )

    eval_basic = evaluator.evaluate_optimization_result(
        result_basic,
        reference_image=result_basic.get('reference_image'),
        roi_mask=result_basic.get('roi_mask')
    )
    eval_adaptive = evaluator.evaluate_optimization_result(
        result_adaptive,
        reference_image=result_adaptive.get('reference_image'),
        roi_mask=result_adaptive.get('roi_mask')
    )

    print(f"\n结果比较:")
    print(f"{'指标':<20} {'基础算法':<15} {'自适应算法':<15}")
    print("-" * 50)
    print(f"{'最优增益 (dB)':<20} {result_basic['optimal_gain']:<15.2f} "
          f"{result_adaptive['optimal_gain']:<15.2f}")
    print(f"{'最终灰度值':<20} {result_basic['final_gray']:<15.2f} "
          f"{result_adaptive['final_gray']:<15.2f}")
    print(f"{'灰度误差':<20} {eval_basic['gray_error']:<15.2f} "
          f"{eval_adaptive['gray_error']:<15.2f}")
    print(f"{'迭代次数':<20} {result_basic['iterations']:<15} "
          f"{result_adaptive['iterations']:<15}")
    print(f"{'优化得分':<20} {eval_basic['optimization_score']:<15.2f} "
          f"{eval_adaptive['optimization_score']:<15.2f}")

    return result_basic, result_adaptive


def example_4_roi_selection():
    """示例4: ROI选择策略"""
    print("\n" + "=" * 60)
    print("示例4: ROI选择策略")
    print("=" * 60)

    data_acq = DataAcquisition()
    led_intensity = 128
    gain = 10.0
    background_light = 30

    image = data_acq.capture_image(led_intensity, gain, background_light)

    strategies = ['center', 'auto']

    for strategy in strategies:
        print(f"\n策略: {strategy}")

        if strategy == 'center':
            roi_mask = data_acq.select_roi(strategy='center')
        else:
            roi_mask = data_acq.select_roi(strategy='auto', image=image)

        stats = data_acq.get_roi_statistics(image, roi_mask)
        print(f"  ROI像素数: {stats['num_pixels']}")
        print(f"  平均灰度值: {stats['mean']:.2f}")
        print(f"  标准差: {stats['std']:.2f}")
        print(f"  最小值: {stats['min']:.2f}")
        print(f"  最大值: {stats['max']:.2f}")
        print(f"  饱和比例: {stats['saturated_ratio']*100:.2f}%")

    return image


def example_5_noise_robustness():
    """示例5: 噪声鲁棒性测试"""
    print("\n" + "=" * 60)
    print("示例5: 噪声鲁棒性测试")
    print("=" * 60)

    data_acq = DataAcquisition()
    optimizer = GainOptimizer(data_acq)
    evaluator = PerformanceEvaluator()

    led_duty_cycle = 50
    background_light = 50
    noise_levels = [0.5, 2.0, 5.0]

    print(f"\n测试噪声水平: {noise_levels}")
    print(f"\n{'噪声':<10} {'最优增益':<12} {'灰度误差':<12} {'迭代次数':<12} {'得分':<10}")
    print("-" * 60)

    results = []
    for noise_std in noise_levels:
        result = optimizer.optimize_gain(
            led_duty_cycle=led_duty_cycle,
            initial_gain=0.0,
            background_light=background_light,
            noise_std=noise_std,
            roi_strategy=ROIStrategy.CENTER
        )
        eval_result = evaluator.evaluate_optimization_result(
            result,
            reference_image=result.get('reference_image'),
            roi_mask=result.get('roi_mask')
        )

        print(f"{noise_std:<10.1f} {result['optimal_gain']:<12.2f} "
              f"{eval_result['gray_error']:<12.2f} {result['iterations']:<12} "
              f"{eval_result['optimization_score']:<10.2f}")

        results.append({
            'noise_std': noise_std,
            'result': result,
            'evaluation': eval_result
        })

    return results


def example_6_varying_conditions():
    """示例6: 不同条件下的测试"""
    print("\n" + "=" * 60)
    print("示例6: 不同光照条件下的测试")
    print("=" * 60)

    data_acq = DataAcquisition()
    optimizer = GainOptimizer(data_acq)

    led_duty_cycles = [30, 50, 70]
    background_lights = [30, 80, 130]

    print(f"\nLED占空比: {led_duty_cycles}")
    print(f"背景光强: {background_lights}")
    print(f"\n{'LED':<8} {'背景光':<10} {'最优增益':<12} {'灰度值':<10} {'迭代':<8}")
    print("-" * 60)

    results = []
    for led_dc in led_duty_cycles:
        for bg_light in background_lights:
            result = optimizer.optimize_gain(
                led_duty_cycle=led_dc,
                initial_gain=0.0,
                background_light=bg_light,
                noise_std=ExperimentConfig.NOISE_STD,
                roi_strategy=ROIStrategy.CENTER
            )

            print(f"{led_dc:<8} {bg_light:<10} {result['optimal_gain']:<12.2f} "
                  f"{result['final_gray']:<10.2f} {result['iterations']:<8}")

            results.append({
                'led_duty_cycle': led_dc,
                'background_light': bg_light,
                'result': result
            })

    return results


def run_all_examples():
    """运行所有示例"""
    print("\n" + "=" * 80)
    print("运行所有示例")
    print("=" * 80)

    examples = [
        ("示例1: 基础使用", example_1_basic_usage),
        ("示例2: 增益扫描", example_2_gain_sweep),
        ("示例3: 算法比较", example_3_algorithm_comparison),
        ("示例4: ROI选择", example_4_roi_selection),
        ("示例5: 噪声鲁棒性", example_5_noise_robustness),
        ("示例6: 不同条件", example_6_varying_conditions)
    ]

    for name, example_func in examples:
        try:
            print(f"\n\n{'=' * 80}")
            print(f"正在运行: {name}")
            print('=' * 80)
            example_func()
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "=" * 80)
    print("所有示例运行完成!")
    print("=" * 80)


def main():
    import sys

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        examples = {
            '1': example_1_basic_usage,
            '2': example_2_gain_sweep,
            '3': example_3_algorithm_comparison,
            '4': example_4_roi_selection,
            '5': example_5_noise_robustness,
            '6': example_6_varying_conditions
        }

        if example_num in examples:
            examples[example_num]()
        else:
            print(f"未找到示例 {example_num}")
            print("可用示例: 1, 2, 3, 4, 5, 6")
    else:
        run_all_examples()


if __name__ == "__main__":
    main()
