"""
增益优化算法在实际图片上的验证
测试算法预测的最优增益与真实实验数据的对比
"""

import sys
import os
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
import cv2

# 配置中文字体
plt.rcParams['font.sans-serif'] = [
    'Hiragino Sans GB', 'Arial Unicode MS', 'PingFang SC',
    'Heiti TC', 'SimHei', 'DejaVu Sans',
]
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

from occ_gain_opt.experiment_loader import ExperimentLoader
from occ_gain_opt.data_acquisition import DataAcquisition
from occ_gain_opt.gain_optimization import GainOptimizer
from occ_gain_opt.config import ROIStrategy


def select_roi_with_fallback(data_acq, image_color, image_gray):
    """
    优先使用 SYNC_BASED ROI，若检测失败 (空掩码) 则退回 AUTO_BRIGHTNESS。

    Args:
        data_acq: DataAcquisition 实例
        image_color: BGR 彩色图像 (供 SYNC_BASED 使用)
        image_gray: 灰度图像 (供 AUTO_BRIGHTNESS 回退使用)

    Returns:
        (roi_mask, strategy_used)
    """
    try:
        roi_mask = data_acq.select_roi(strategy=ROIStrategy.SYNC_BASED, image=image_color)
        if np.sum(roi_mask) > 0:
            return roi_mask, "sync_based"
    except Exception:
        pass
    roi_mask = data_acq.select_roi(strategy=ROIStrategy.AUTO_BRIGHTNESS, image=image_gray)
    return roi_mask, "auto_brightness"


def get_real_gain_response(images, experiment_type, image_type):
    """
    从真实图片中获取增益-灰度响应曲线

    Returns:
        (gains, gray_values, 等效增益列表)
    """
    print(f"\n分析 {experiment_type}/{image_type} 的真实增益响应...")

    results = []
    for img in images:
        if not img.load():
            continue

        color = img.image        # BGR 彩色图
        gray = img.gray_image    # 灰度图
        h, w = gray.shape

        # 优先使用 sync-based ROI，检测失败则退回自动亮度
        data_acq = DataAcquisition(width=w, height=h)
        roi_mask, roi_strategy = select_roi_with_fallback(data_acq, color, gray)
        stats = data_acq.get_roi_statistics(gray, roi_mask)

        # 计算相对于基准的增益
        base_exposure = 1/27000  # 根据说明文档
        base_iso = 35
        equiv_gain = img.calculate_equivalent_gain(base_exposure, base_iso)

        results.append({
            'gain': equiv_gain,
            'gray': stats['mean'],
            'std': stats['std'],
            'saturated': stats['saturated_ratio'],
            'image': img
        })

    # 按增益排序
    results.sort(key=lambda x: x['gain'])

    gains = [r['gain'] for r in results]
    grays = [r['gray'] for r in results]
    stds = [r['std'] for r in results]
    saturated = [r['saturated'] for r in results]

    return gains, grays, stds, saturated, results


def test_algorithm_prediction(images, experiment_type, image_type):
    """
    测试算法对最优增益的预测

    从最低增益图片开始，模拟算法优化过程
    """
    print(f"\n{'='*60}")
    print(f"测试算法预测 - {experiment_type}/{image_type}")
    print(f"{'='*60}")

    # 获取真实响应
    gains, grays, stds, saturated, results = get_real_gain_response(images, experiment_type, image_type)

    if len(gains) == 0:
        print("没有可用的图片数据")
        return None

    # 找到实际的最优增益（灰度最接近255且不过饱和）
    target_gray = 255 * 0.95
    best_idx = None
    best_score = float('inf')

    for i, (gain, gray, sat) in enumerate(zip(gains, grays, saturated)):
        if sat < 0.1:  # 饱和度小于10%
            score = abs(gray - target_gray)
            if score < best_score:
                best_score = score
                best_idx = i

    if best_idx is None:
        print("警告：没有找到未饱和的图片")
        best_idx = 0

    actual_optimal_gain = gains[best_idx]
    actual_optimal_gray = grays[best_idx]

    print(f"\n实际最优设置:")
    print(f"  最优增益: {actual_optimal_gain:.2f} dB")
    print(f"  灰度值: {actual_optimal_gray:.2f}")
    print(f"  饱和度: {saturated[best_idx]*100:.2f}%")

    # 模拟算法优化过程
    # 从最低增益开始
    initial_gain = gains[0]
    initial_gray = grays[0]

    print(f"\n算法优化模拟:")
    print(f"  初始增益: {initial_gain:.2f} dB")
    print(f"  初始灰度: {initial_gray:.2f}")

    # 使用增益优化公式计算最优增益
    # G_opt = G_curr × (Y_target / Y_curr)
    if initial_gray > 0:
        predicted_gain_db = initial_gain + 20 * np.log10(target_gray / initial_gray)
        print(f"  预测增益: {predicted_gain_db:.2f} dB")

        # 找到最接近预测增益的实际图片
        gain_errors = [abs(g - predicted_gain_db) for g in gains]
        closest_idx = np.argmin(gain_errors)
        closest_gain = gains[closest_idx]
        closest_gray = grays[closest_idx]

        print(f"  最接近的实际增益: {closest_gain:.2f} dB")
        print(f"  对应灰度值: {closest_gray:.2f}")

        # 计算预测误差
        gain_error = abs(predicted_gain_db - actual_optimal_gain)
        gray_error = abs(closest_gray - target_gray)

        print(f"\n预测误差:")
        print(f"  增益误差: {gain_error:.2f} dB")
        print(f"  灰度误差: {gray_error:.2f}")

        return {
            'experiment_type': experiment_type,
            'image_type': image_type,
            'gains': gains,
            'grays': grays,
            'stds': stds,
            'saturated': saturated,
            'initial_gain': initial_gain,
            'initial_gray': initial_gray,
            'predicted_gain': predicted_gain_db,
            'actual_optimal_gain': actual_optimal_gain,
            'actual_optimal_gray': actual_optimal_gray,
            'closest_gain': closest_gain,
            'closest_gray': closest_gray,
            'gain_error': gain_error,
            'gray_error': gray_error
        }

    return None


def plot_validation_results(all_results):
    """绘制算法验证结果"""
    print("\n生成验证图表...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'v', 'D', 'p']

    # 1. 增益-灰度响应曲线
    ax = axes[0, 0]
    for i, result in enumerate(all_results):
        gains = result['gains']
        grays = result['grays']
        label = f"{result['experiment_type']}/{result['image_type']}"
        ax.plot(gains, grays, marker=markers[i % len(markers)],
                color=colors[i % len(colors)], label=label, alpha=0.7)

    ax.axhline(y=255*0.95, color='r', linestyle='--', linewidth=2, label='目标灰度 (95%)')
    ax.set_xlabel('增益 (dB)', fontsize=12)
    ax.set_ylabel('ROI平均灰度值', fontsize=12)
    ax.set_title('增益-灰度响应曲线', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. 预测误差对比
    ax = axes[0, 1]
    experiments = [f"{r['experiment_type']}/{r['image_type']}" for r in all_results]
    gain_errors = [r['gain_error'] for r in all_results]
    gray_errors = [r['gray_error'] for r in all_results]

    x = np.arange(len(experiments))
    width = 0.35

    bars1 = ax.bar(x - width/2, gain_errors, width, label='增益误差 (dB)', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, gray_errors, width, label='灰度误差', color='#ff7f0e', alpha=0.8)

    ax.set_xlabel('实验类型', fontsize=12)
    ax.set_ylabel('误差', fontsize=12)
    ax.set_title('算法预测误差', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. 预测vs实际增益对比
    ax = axes[1, 0]
    predicted_gains = [r['predicted_gain'] for r in all_results]
    actual_gains = [r['actual_optimal_gain'] for r in all_results]

    ax.scatter(actual_gains, predicted_gains, s=100, alpha=0.7, color='#2ca02c', edgecolors='black')

    # 添加对角线（完美预测）
    min_gain = min(min(actual_gains), min(predicted_gains))
    max_gain = max(max(actual_gains), max(predicted_gains))
    ax.plot([min_gain, max_gain], [min_gain, max_gain], 'r--', linewidth=2, label='完美预测')

    # 标注每个点
    for i, exp in enumerate(experiments):
        ax.annotate(exp.split('/')[0], (actual_gains[i], predicted_gains[i]),
                   fontsize=8, alpha=0.7)

    ax.set_xlabel('实际最优增益 (dB)', fontsize=12)
    ax.set_ylabel('预测增益 (dB)', fontsize=12)
    ax.set_title('预测vs实际增益对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. 初始vs优化后灰度值对比
    ax = axes[1, 1]
    initial_grays = [r['initial_gray'] for r in all_results]
    final_grays = [r['closest_gray'] for r in all_results]

    x = np.arange(len(experiments))
    width = 0.35

    bars1 = ax.bar(x - width/2, initial_grays, width, label='初始灰度', color='#d62728', alpha=0.8)
    bars2 = ax.bar(x + width/2, final_grays, width, label='优化后灰度', color='#2ca02c', alpha=0.8)

    ax.axhline(y=255*0.95, color='r', linestyle='--', linewidth=2, label='目标灰度 (95%)')

    ax.set_xlabel('实验类型', fontsize=12)
    ax.set_ylabel('灰度值', fontsize=12)
    ax.set_title('算法优化效果', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图表
    results_dir = Path(__file__).parent.parent / "results" / "algorithm_validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "algorithm_validation_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")

    return output_path


def main():
    """主函数"""
    print("\n" + "="*60)
    print("增益优化算法在实际数据上的验证")
    print("="*60)

    loader = ExperimentLoader()

    # 测试所有实验类型
    experiments = [
        ("bubble", "ISO"),
        ("bubble", "Texp"),
        ("tap water", "ISO"),
        ("tap water", "Texp"),
        ("turbidity", "ISO"),
        ("turbidity", "Texp"),
    ]

    all_results = []

    for exp_type, img_type in experiments:
        images = loader.load_experiment(exp_type, img_type)

        if len(images) == 0:
            print(f"\n跳过 {exp_type}/{img_type}（没有图片）")
            continue

        result = test_algorithm_prediction(images, exp_type, img_type)

        if result:
            all_results.append(result)

    # 生成汇总报告
    if len(all_results) > 0:
        print(f"\n{'='*60}")
        print("验证汇总")
        print(f"{'='*60}")

        avg_gain_error = np.mean([r['gain_error'] for r in all_results])
        avg_gray_error = np.mean([r['gray_error'] for r in all_results])

        print(f"\n平均预测误差:")
        print(f"  增益误差: {avg_gain_error:.2f} dB")
        print(f"  灰度误差: {avg_gray_error:.2f}")

        print(f"\n详细结果:")
        for r in all_results:
            print(f"\n  {r['experiment_type']}/{r['image_type']}:")
            print(f"    增益误差: {r['gain_error']:.2f} dB")
            print(f"    灰度误差: {r['gray_error']:.2f}")
            print(f"    改善: {r['closest_gray'] - r['initial_gray']:.2f} 灰度值")

        # 绘制结果
        plot_validation_results(all_results)

        # 保存报告
        results_dir = Path(__file__).parent.parent / "results" / "algorithm_validation"
        report_path = results_dir / "validation_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("增益优化算法验证报告\n")
            f.write("="*60 + "\n\n")

            f.write(f"测试数据集数量: {len(all_results)}\n\n")
            f.write(f"平均预测误差:\n")
            f.write(f"  增益误差: {avg_gain_error:.2f} dB\n")
            f.write(f"  灰度误差: {avg_gray_error:.2f}\n\n")

            f.write("详细结果:\n")
            f.write("-"*60 + "\n")
            for r in all_results:
                f.write(f"\n{r['experiment_type']}/{r['image_type']}:\n")
                f.write(f"  初始增益: {r['initial_gain']:.2f} dB\n")
                f.write(f"  初始灰度: {r['initial_gray']:.2f}\n")
                f.write(f"  预测增益: {r['predicted_gain']:.2f} dB\n")
                f.write(f"  实际最优增益: {r['actual_optimal_gain']:.2f} dB\n")
                f.write(f"  增益误差: {r['gain_error']:.2f} dB\n")
                f.write(f"  灰度误差: {r['gray_error']:.2f}\n")
                f.write(f"  改善: {r['closest_gray'] - r['initial_gray']:.2f} 灰度值\n")

        print(f"\n报告已保存到: {report_path}")

    print(f"\n{'='*60}")
    print("验证完成！")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
