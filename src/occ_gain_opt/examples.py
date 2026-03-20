"""
使用示例和测试脚本
演示如何使用统一架构的增益优化算法
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_sources import SimulatedDataSource, create_center_roi_mask, compute_roi_stats
from .algorithms import get as algo_get
from .performance_evaluation import PerformanceEvaluator
from .config import CameraParams, ROIStrategy, ExperimentConfig


def example_1_basic_usage():
    """示例1: 基础使用 - 单次公式算法"""
    print("\n" + "=" * 60)
    print("示例1: 基础使用 - 单次公式算法")
    print("=" * 60)

    # 创建仿真数据源
    data_source = SimulatedDataSource(
        width=640, height=480,
        led_intensity=128, background_light=50, noise_std=2.0
    )

    # 使用单次公式算法
    algo = algo_get("single_shot")()

    # 初始参数
    current_params = CameraParams(iso=35, exposure_us=27.9)
    data_source.set_params(current_params)

    print(f"\n实验条件:")
    print(f"  LED强度: {data_source.led_intensity}")
    print(f"  背景光: {data_source.background_light}")
    print(f"  初始ISO: {current_params.iso}")

    # 采集当前帧
    image = data_source.get_frame()
    roi_mask = create_center_roi_mask(image, roi_size=300)
    stats = compute_roi_stats(image, roi_mask)
    brightness = stats['mean']

    print(f"  当前亮度: {brightness:.1f}")

    # 计算下一步参数
    next_params = algo.compute_next_params(current_params, brightness)

    print(f"\n优化结果:")
    print(f"  建议ISO: {next_params.iso:.1f}")
    print(f"  建议增益: {next_params.gain_db:+.2f} dB")
    print(f"  曝光时间: {next_params.exposure_us:.1f} µs")

    return next_params


def example_2_gain_sweep():
    """示例2: 增益扫描 - 研究增益对灰度值的影响"""
    print("\n" + "=" * 60)
    print("示例2: 增益扫描 - 研究增益对灰度值的影响")
    print("=" * 60)

    # 创建数据源
    data_source = SimulatedDataSource(
        width=640, height=480,
        led_intensity=128, background_light=50, noise_std=2.0
    )

    # 扫描增益
    iso_values = np.linspace(30, 1000, 50)
    brightness_values = []

    print(f"\n扫描 {len(iso_values)} 个ISO档位...")

    for iso in iso_values:
        params = CameraParams(iso=iso, exposure_us=27.9)
        data_source.set_params(params)
        image = data_source.get_frame()
        roi_mask = create_center_roi_mask(image, roi_size=300)
        stats = compute_roi_stats(image, roi_mask)
        brightness_values.append(stats['mean'])

    # 找到最优
    max_idx = np.argmax(brightness_values)
    optimal_iso = iso_values[max_idx]
    max_brightness = brightness_values[max_idx]

    print(f"\n最优工作点:")
    print(f"  ISO: {optimal_iso:.1f}")
    print(f"  亮度: {max_brightness:.2f}")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(iso_values, brightness_values, 'b-', linewidth=2)
    plt.axhline(y=255, color='r', linestyle='--', label='饱和值(255)')
    plt.plot(optimal_iso, max_brightness, 'ro', markersize=10,
             label=f'最优点(ISO={optimal_iso:.0f}, 亮度={max_brightness:.1f})')
    plt.xlabel('ISO', fontsize=12)
    plt.ylabel('ROI平均灰度值', fontsize=12)
    plt.title('ISO-灰度响应曲线', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    import os
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/example_2_gain_sweep.png', dpi=150)
    print(f"\n图表已保存: results/plots/example_2_gain_sweep.png")
    plt.close()

    return iso_values, brightness_values


def example_3_algorithm_comparison():
    """示例3: 算法比较 - 单次公式 vs 自适应迭代 vs 自适应阻尼"""
    print("\n" + "=" * 60)
    print("示例3: 算法比较 - 单次公式 vs 自适应迭代 vs 自适应阻尼")
    print("=" * 60)

    # 测试条件
    led_intensity = 128
    background_light = 50

    print(f"\n测试条件:")
    print(f"  LED强度: {led_intensity}")
    print(f"  背景光: {background_light}")

    algorithms = {
        "single_shot": "单次公式",
        "adaptive_iter": "自适应迭代",
        "adaptive_damping": "自适应阻尼"
    }

    results = {}

    for algo_name, algo_desc in algorithms.items():
        print(f"\n运行 {algo_desc} 算法...")

        data_source = SimulatedDataSource(
            width=640, height=480,
            led_intensity=led_intensity,
            background_light=background_light,
            noise_std=2.0
        )

        algo = algo_get(algo_name)()
        current_params = CameraParams(iso=35, exposure_us=27.9)
        data_source.set_params(current_params)

        history = []
        max_iterations = 10

        for i in range(max_iterations):
            image = data_source.get_frame()
            roi_mask = create_center_roi_mask(image, roi_size=300)
            stats = compute_roi_stats(image, roi_mask)
            brightness = stats['mean']

            history.append({
                'iteration': i,
                'iso': current_params.iso,
                'gain_db': current_params.gain_db,
                'brightness': brightness
            })

            if algo.is_converged():
                break

            next_params = algo.compute_next_params(current_params, brightness)
            current_params = next_params
            data_source.set_params(current_params)

        results[algo_name] = history

        print(f"  迭代次数: {len(history)}")
        print(f"  最终ISO: {history[-1]['iso']:.1f}")
        print(f"  最终亮度: {history[-1]['brightness']:.1f}")

    # 绘制对比图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for algo_name, algo_desc in algorithms.items():
        hist = results[algo_name]
        iterations = [h['iteration'] for h in hist]
        gains = [h['gain_db'] for h in hist]
        plt.plot(iterations, gains, 'o-', label=algo_desc, linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('增益 (dB)')
    plt.title('增益收敛曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for algo_name, algo_desc in algorithms.items():
        hist = results[algo_name]
        iterations = [h['iteration'] for h in hist]
        brightness = [h['brightness'] for h in hist]
        plt.plot(iterations, brightness, 'o-', label=algo_desc, linewidth=2)
    plt.axhline(y=242.25, color='r', linestyle='--', label='目标亮度')
    plt.xlabel('迭代次数')
    plt.ylabel('ROI亮度')
    plt.title('亮度收敛曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/plots/example_3_comparison.png', dpi=150)
    print(f"\n对比图已保存: results/plots/example_3_comparison.png")
    plt.close()

    return results


def example_4_roi_strategies():
    """示例4: ROI策略对比"""
    print("\n" + "=" * 60)
    print("示例4: ROI策略对比")
    print("=" * 60)

    from .data_sources.roi import create_auto_roi_mask, create_sync_based_roi_mask

    # 创建数据源
    data_source = SimulatedDataSource(
        width=640, height=480,
        led_intensity=128, background_light=50, noise_std=2.0
    )

    params = CameraParams(iso=200, exposure_us=27.9)
    data_source.set_params(params)
    image = data_source.get_frame()

    print("\n测试不同ROI策略:")

    # 中心ROI
    roi_center = create_center_roi_mask(image, roi_size=300)
    stats_center = compute_roi_stats(image, roi_center)
    print(f"  中心ROI: 亮度={stats_center['mean']:.1f}, 像素数={stats_center['num_pixels']}")

    # 自动亮度ROI
    roi_auto = create_auto_roi_mask(image, roi_size=300)
    stats_auto = compute_roi_stats(image, roi_auto)
    print(f"  自动ROI: 亮度={stats_auto['mean']:.1f}, 像素数={stats_auto['num_pixels']}")

    # 可视化
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    overlay_center = image.copy()
    overlay_center[roi_center == 0] = overlay_center[roi_center == 0] // 2
    plt.imshow(overlay_center, cmap='gray')
    plt.title(f'中心ROI (亮度={stats_center["mean"]:.1f})')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    overlay_auto = image.copy()
    overlay_auto[roi_auto == 0] = overlay_auto[roi_auto == 0] // 2
    plt.imshow(overlay_auto, cmap='gray')
    plt.title(f'自动ROI (亮度={stats_auto["mean"]:.1f})')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/plots/example_4_roi_strategies.png', dpi=150)
    print(f"\nROI对比图已保存: results/plots/example_4_roi_strategies.png")
    plt.close()

    return {
        'center': stats_center,
        'auto': stats_auto
    }


def example_5_simulated_vs_real():
    """示例5: 仿真数据源 vs 数据集数据源"""
    print("\n" + "=" * 60)
    print("示例5: 数据源对比 - 仿真 vs 数据集")
    print("=" * 60)

    from .data_sources import DatasetDataSource

    # 仿真数据源
    sim_source = SimulatedDataSource(
        width=640, height=480,
        led_intensity=128, background_light=50, noise_std=2.0
    )

    params = CameraParams(iso=200, exposure_us=27.9)
    sim_source.set_params(params)
    sim_frame = sim_source.get_frame()

    print(f"\n仿真数据源:")
    print(f"  帧 shape: {sim_frame.shape}")
    print(f"  亮度范围: [{sim_frame.min()}, {sim_frame.max()}]")

    # 尝试加载真实数据集
    try:
        dataset_source = DatasetDataSource(
            condition="tap water",
            image_type="ISO"
        )
        # 设置接近的参数
        dataset_source.set_params(params)
        real_frame = dataset_source.get_frame()

        print(f"\n真实数据集:")
        print(f"  帧 shape: {real_frame.shape}")
        print(f"  亮度范围: [{real_frame.min()}, {real_frame.max()}]")

        # 可视化对比
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(sim_frame, cmap='gray', vmin=0, vmax=255)
        plt.title('仿真图像')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(real_frame, cmap='gray', vmin=0, vmax=255)
        plt.title('真实图像')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('results/plots/example_5_sim_vs_real.png', dpi=150)
        print(f"\n对比图已保存: results/plots/example_5_sim_vs_real.png")
        plt.close()

        return {'sim': sim_frame, 'real': real_frame}

    except Exception as e:
        print(f"\n真实数据集不可用: {e}")
        print("  (需要 ISO-Texp/ 数据集)")
        return {'sim': sim_frame}


def example_6_ber_explore():
    """示例6: BER探索算法"""
    print("\n" + "=" * 60)
    print("示例6: BER探索算法")
    print("=" * 60)

    # BER探索算法需要BER作为输入
    algo = algo_get("ber_explore")()

    data_source = SimulatedDataSource(
        width=640, height=480,
        led_intensity=128, background_light=50, noise_std=2.0
    )

    current_params = CameraParams(iso=35, exposure_us=27.9)
    data_source.set_params(current_params)

    print("\n模拟BER探索 (随机BER):")
    import random

    for i in range(5):
        image = data_source.get_frame()
        roi_mask = create_center_roi_mask(image, roi_size=300)
        stats = compute_roi_stats(image, roi_mask)
        brightness = stats['mean']

        # 模拟BER (随机)
        ber = random.uniform(0.001, 0.1) if random.random() > 0.3 else None

        next_params = algo.compute_next_params(current_params, brightness, ber)

        print(f"  迭代 {i}: ISO={current_params.iso:.0f}, "
              f"亮度={brightness:.1f}, BER={ber if ber else 'N/A'}, "
              f"-> ISO={next_params.iso:.0f}")

        current_params = next_params
        data_source.set_params(current_params)

    return True


def main():
    """运行所有示例"""
    import sys
    import os

    os.makedirs('results/plots', exist_ok=True)

    # 获取命令行参数
    if len(sys.argv) > 1:
        example_id = sys.argv[1]
        examples = {
            '1': example_1_basic_usage,
            '2': example_2_gain_sweep,
            '3': example_3_algorithm_comparison,
            '4': example_4_roi_strategies,
            '5': example_5_simulated_vs_real,
            '6': example_6_ber_explore,
        }
        if example_id in examples:
            examples[example_id]()
        else:
            print(f"未知示例编号: {example_id}")
            print(f"可用示例: {', '.join(examples.keys())}")
    else:
        # 运行所有示例
        print("\n" + "=" * 70)
        print("运行所有示例")
        print("=" * 70)

        example_1_basic_usage()
        example_2_gain_sweep()
        example_3_algorithm_comparison()
        example_4_roi_strategies()
        example_5_simulated_vs_real()
        example_6_ber_explore()

        print("\n" + "=" * 70)
        print("所有示例运行完成!")
        print("=" * 70)


if __name__ == "__main__":
    main()
