"""
使用迭代优化验证增益优化算法
对比单次计算和迭代优化的效果，并生成详细的对比图
"""

import sys
import os
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.gridspec import GridSpec

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

from occ_gain_opt.experiment_loader import ExperimentLoader, ExperimentImage
from occ_gain_opt.data_acquisition import DataAcquisition
from occ_gain_opt.config import ROIStrategy


def find_closest_image(images, target_gain):
    """从图片列表中找到最接近目标增益的图片"""
    if not images:
        return None

    gains = [img.calculate_equivalent_gain() for img in images]
    errors = [abs(g - target_gain) for g in gains]
    idx = np.argmin(errors)

    return images[idx]


def measure_gray_from_image(image_path):
    """从图片中测量ROI灰度值"""
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    data_acq = DataAcquisition(width=w, height=h)
    roi_mask = data_acq.select_roi(strategy=ROIStrategy.AUTO_BRIGHTNESS, image=gray)
    stats = data_acq.get_roi_statistics(gray, roi_mask)

    return stats['mean'], stats['std'], stats['saturated_ratio'], img, gray, roi_mask


def single_step_optimization(initial_gray, initial_gain, target_gray=255*0.95):
    """单次优化"""
    if initial_gray <= 0:
        return initial_gain

    gain_db = initial_gain + 20 * np.log10(target_gray / initial_gray)
    return gain_db


def iterative_optimization(images, initial_image, max_iterations=10, target_gray=255*0.95,
                          tolerance=5.0, alpha=0.5):
    """
    迭代优化算法

    Args:
        images: 可用的实验图片列表
        initial_image: 初始图片
        max_iterations: 最大迭代次数
        target_gray: 目标灰度
        tolerance: 收敛容忍度
        alpha: 学习率（0-1）

    Returns:
        优化历史和最终结果
    """
    history = []

    # 初始状态
    current_image = initial_image
    current_gain = current_image.calculate_equivalent_gain()

    # 测量初始灰度
    result = measure_gray_from_image(current_image.filepath)
    if result is None:
        return history

    mean_gray, std_gray, sat_ratio, color_img, gray_img, roi_mask = result

    history.append({
        'iteration': 0,
        'gain': current_gain,
        'gray': mean_gray,
        'std': std_gray,
        'saturated': sat_ratio,
        'image': current_image,
        'color_img': color_img,
        'gray_img': gray_img,
        'roi_mask': roi_mask
    })

    print(f"\n迭代优化过程:")
    print(f"  迭代 0: 增益={current_gain:.2f} dB, 灰度={mean_gray:.2f}, 饱和度={sat_ratio*100:.2f}%")

    # 迭代优化
    for i in range(1, max_iterations + 1):
        current_gray = history[-1]['gray']
        current_gain_val = history[-1]['gain']

        # 检查是否收敛
        if abs(current_gray - target_gray) < tolerance:
            print(f"  ✓ 收敛于迭代 {i-1}: 灰度={current_gray:.2f} (目标={target_gray:.2f})")
            break

        # 计算最优增益
        if current_gray > 0:
            # 使用学习率避免过度调整
            optimal_gain = current_gain_val + alpha * 20 * np.log10(target_gray / current_gray)
        else:
            optimal_gain = current_gain_val + 3  # 保守增加

        # 找到最接近的实验图片
        closest_image = find_closest_image(images, optimal_gain)

        if closest_image is None:
            print(f"  ✗ 无法找到合适的图片")
            break

        # 测量新图片
        result = measure_gray_from_image(closest_image.filepath)
        if result is None:
            break

        new_gain = closest_image.calculate_equivalent_gain()
        new_gray, new_std, new_sat, new_color, new_gray_img, new_roi = result

        history.append({
            'iteration': i,
            'gain': new_gain,
            'gray': new_gray,
            'std': new_std,
            'saturated': new_sat,
            'image': closest_image,
            'color_img': new_color,
            'gray_img': new_gray_img,
            'roi_mask': new_roi
        })

        print(f"  迭代 {i}: 增益={new_gain:.2f} dB, 灰度={new_gray:.2f}, 饱和度={new_sat*100:.2f}%")

        # 检查增益变化
        if abs(new_gain - current_gain_val) < 0.5:
            print(f"  ✓ 增益变化小于0.5 dB，停止迭代")
            break

    return history


def plot_comparison(images, initial_image, exp_name, target_gray=255*0.95, max_iterations=5, alpha=0.5):
    """绘制对比图"""
    # 迭代优化
    iterative_history = iterative_optimization(
        images, initial_image, max_iterations, target_gray, alpha=alpha
    )

    if len(iterative_history) == 0:
        print(f"  ✗ 迭代优化失败")
        return None, None, None, None

    # 单次优化
    initial_gray = iterative_history[0]['gray']
    initial_gain = iterative_history[0]['gain']
    single_gain = single_step_optimization(initial_gray, initial_gain, target_gray)
    single_image = find_closest_image(images, single_gain)

    if single_image:
        result = measure_gray_from_image(single_image.filepath)
        if result:
            single_gray, single_std, single_sat, single_color, single_gray_img, single_roi = result
        else:
            single_gray, single_sat, single_color, single_gray_img, single_roi = None, None, None, None, None, None
    else:
        single_gray, single_sat, single_color, single_gray_img, single_roi = None, None, None, None, None

    # 创建大图
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # 获取数据
    init_color = iterative_history[0]['color_img']
    init_gray = iterative_history[0]['gray_img']
    init_roi = iterative_history[0]['roi_mask']

    final_iter = iterative_history[-1]
    final_color = final_iter['color_img']
    final_gray = final_iter['gray_img']
    final_roi = final_iter['roi_mask']

    # 1. 初始图像
    ax1 = fig.add_subplot(gs[0, 0])
    init_display = init_color.copy()
    mask_3ch = cv2.cvtColor((init_roi * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    init_display = cv2.addWeighted(init_display, 0.7, mask_3ch, 0.3, 0)

    ax1.imshow(cv2.cvtColor(init_display, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'初始图像\n增益={initial_gain:.2f} dB, 灰度={initial_gray:.2f}',
                  fontsize=11, fontweight='bold')
    ax1.axis('off')

    # 2. 迭代优化最终图像
    ax2 = fig.add_subplot(gs[0, 1])
    final_display = final_color.copy()
    mask_3ch = cv2.cvtColor((final_roi * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    final_display = cv2.addWeighted(final_display, 0.7, mask_3ch, 0.3, 0)

    ax2.imshow(cv2.cvtColor(final_display, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'迭代优化结果 ({len(iterative_history)-1}次迭代)\n增益={final_iter["gain"]:.2f} dB, 灰度={final_iter["gray"]:.2f}',
                  fontsize=11, fontweight='bold', color='green')
    ax2.axis('off')

    # 3. 单次优化图像
    ax3 = fig.add_subplot(gs[0, 2])
    if single_image and single_gray is not None:
        single_display = single_color.copy()
        mask_3ch = cv2.cvtColor((single_roi * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        single_display = cv2.addWeighted(single_display, 0.7, mask_3ch, 0.3, 0)

        ax3.imshow(cv2.cvtColor(single_display, cv2.COLOR_BGR2RGB))
        ax3.set_title(f'单次优化结果\n增益={single_gain:.2f} dB, 灰度={single_gray:.2f}',
                      fontsize=11, fontweight='bold', color='orange')
    else:
        ax3.text(0.5, 0.5, '单次优化\n失败', ha='center', va='center', fontsize=12)
        ax3.set_title('单次优化结果', fontsize=11)
    ax3.axis('off')

    # 4. ROI直方图对比
    ax4 = fig.add_subplot(gs[0, 3])

    # 绘制直方图
    init_roi_vals = init_gray[init_roi == 1]
    final_roi_vals = final_gray[final_roi == 1]

    ax4.hist(init_roi_vals, bins=30, alpha=0.5, label='初始', color='red', density=True)
    ax4.hist(final_roi_vals, bins=30, alpha=0.5, label='迭代优化', color='green', density=True)

    if single_image and single_gray is not None and single_gray_img is not None:
        single_roi_vals = single_gray_img[single_roi == 1]
        ax4.hist(single_roi_vals, bins=30, alpha=0.5, label='单次优化', color='orange', density=True)

    ax4.axvline(x=target_gray, color='blue', linestyle='--', linewidth=2, label='目标灰度')
    ax4.set_xlabel('灰度值', fontsize=10)
    ax4.set_ylabel('密度', fontsize=10)
    ax4.set_title('ROI灰度分布对比', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. 迭代收敛曲线
    ax5 = fig.add_subplot(gs[1, :2])

    iterations_val = [h['iteration'] for h in iterative_history]
    gains_val = [h['gain'] for h in iterative_history]
    grays_val = [h['gray'] for h in iterative_history]

    ax5_twin = ax5.twinx()

    line1 = ax5.plot(iterations_val, gains_val, 'o-', linewidth=2, markersize=8, color='blue', label='增益 (dB)')
    line2 = ax5_twin.plot(iterations_val, grays_val, 's-', linewidth=2, markersize=8, color='green', label='灰度值')

    ax5_twin.axhline(y=target_gray, color='red', linestyle='--', linewidth=2, label='目标灰度')

    ax5.set_xlabel('迭代次数', fontsize=11)
    ax5.set_ylabel('增益 (dB)', fontsize=11, color='blue')
    ax5_twin.set_ylabel('灰度值', fontsize=11, color='green')
    ax5.set_title('迭代优化收敛曲线', fontsize=12, fontweight='bold')
    ax5.tick_params(axis='y', labelcolor='blue')
    ax5_twin.tick_params(axis='y', labelcolor='green')
    ax5.grid(True, alpha=0.3)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper left', fontsize=10)

    # 6. 方法对比柱状图
    ax6 = fig.add_subplot(gs[1, 2:])

    methods = ['初始', '迭代优化', '单次优化', '目标']
    gray_values_list = [
        iterative_history[0]['gray'],
        iterative_history[-1]['gray'],
        single_gray if single_gray else 0,
        target_gray
    ]

    x_val = np.arange(len(methods))
    width = 0.35

    bars1 = ax6.bar(x_val - width/2, gray_values_list, width, label='灰度值', color='green', alpha=0.8)
    ax6.axhline(y=target_gray, color='red', linestyle='--', linewidth=2, label='目标灰度')

    # 在柱子上标注数值
    for i, (bar, val) in enumerate(zip(bars1, gray_values_list)):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    ax6.set_xlabel('优化方法', fontsize=11)
    ax6.set_ylabel('灰度值', fontsize=11)
    ax6.set_title('不同方法结果对比', fontsize=12, fontweight='bold')
    ax6.set_xticks(x_val)
    ax6.set_xticklabels(methods, fontsize=10)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. 误差分析
    ax7 = fig.add_subplot(gs[1, 3])

    errors_list = [
        abs(iterative_history[0]['gray'] - target_gray),
        abs(iterative_history[-1]['gray'] - target_gray),
        abs(single_gray - target_gray) if single_gray else 0,
        0
    ]

    colors_bars = ['red', 'green', 'orange', 'blue']
    bars = ax7.bar(methods, errors_list, color=colors_bars, alpha=0.8, edgecolor='black')

    for i, (bar, val) in enumerate(zip(bars, errors_list)):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    ax7.set_ylabel('与目标的误差', fontsize=11)
    ax7.set_title('灰度误差对比', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. 详细统计信息
    ax8 = fig.add_subplot(gs[2, :])
    ax8.axis('off')

    # 创建文本统计
    info_text = f"实验: {exp_name}\n"
    info_text += f"目标灰度: {target_gray:.2f}\n\n"
    info_text += f"初始状态:\n"
    info_text += f"  增益: {iterative_history[0]['gain']:.2f} dB\n"
    info_text += f"  灰度: {iterative_history[0]['gray']:.2f}\n"
    info_text += f"  误差: {abs(iterative_history[0]['gray'] - target_gray):.2f}\n\n"

    info_text += f"迭代优化 ({len(iterative_history)-1}次迭代):\n"
    info_text += f"  最终增益: {iterative_history[-1]['gain']:.2f} dB\n"
    info_text += f"  最终灰度: {iterative_history[-1]['gray']:.2f}\n"
    info_text += f"  误差: {abs(iterative_history[-1]['gray'] - target_gray):.2f}\n"
    init_error = abs(iterative_history[0]['gray'] - target_gray)
    iter_error = abs(iterative_history[-1]['gray'] - target_gray)
    improvement = (init_error - iter_error)
    improvement_rate = (improvement / init_error * 100) if init_error > 0 else 0
    info_text += f"  改善: {improvement:.2f} ({improvement_rate:.1f}%)\n\n"

    if single_gray is not None:
        info_text += f"单次优化:\n"
        info_text += f"  预测增益: {single_gain:.2f} dB\n"
        info_text += f"  实际灰度: {single_gray:.2f}\n"
        info_text += f"  误差: {abs(single_gray - target_gray):.2f}\n"
        single_improvement = (init_error - abs(single_gray - target_gray))
        single_improvement_rate = (single_improvement / init_error * 100) if init_error > 0 else 0
        info_text += f"  改善: {single_improvement:.2f} ({single_improvement_rate:.1f}%)\n\n"

        # 方法对比
        info_text += f"方法对比:\n"
        if iter_error < abs(single_gray - target_gray):
            better = "迭代优化"
            diff = abs(single_gray - target_gray) - iter_error
        else:
            better = "单次优化"
            diff = iter_error - abs(single_gray - target_gray)
        info_text += f"  更优方法: {better}\n"
        info_text += f"  误差差异: {diff:.2f}\n"

    ax8.text(0.1, 0.5, info_text, transform=ax8.transAxes, fontsize=11,
            verticalalignment='center', family='sans-serif',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'增益优化算法对比：迭代优化 vs 单次优化 - {exp_name}',
                 fontsize=16, fontweight='bold', y=0.995)

    return fig, iterative_history, single_gray, single_gain


def validate_all_experiments():
    """在所有实验上验证迭代优化"""
    print("\n" + "="*70)
    print("迭代优化算法验证（对比单次优化）")
    print("="*70)

    loader = ExperimentLoader()

    experiments = [
        ("bubble", "ISO"),
        ("bubble", "Texp"),
        ("tap water", "ISO"),
        ("tap water", "Texp"),
        ("turbidity", "ISO"),
        ("turbidity", "Texp"),
    ]

    all_results = []
    results_dir = Path(__file__).parent.parent / "results" / "iterative_validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    for exp_type, img_type in experiments:
        print(f"\n{'='*70}")
        print(f"实验: {exp_type}/{img_type}")
        print(f"{'='*70}")

        images = loader.load_experiment(exp_type, img_type)

        if len(images) == 0:
            print(f"跳过 {exp_type}/{img_type}（没有图片）")
            continue

        # 使用最低增益的图片作为初始点
        images_by_gain = sorted(images, key=lambda x: x.calculate_equivalent_gain())
        initial_image = images_by_gain[0]

        print(f"初始图片: {initial_image.filepath}")
        print(f"初始增益: {initial_image.calculate_equivalent_gain():.2f} dB")

        # 生成对比图
        try:
            exp_name = f"{exp_type}/{img_type}"
            fig, history, single_gray, single_gain_val = plot_comparison(images, initial_image, exp_name)

            if fig is None:
                continue

            # 保存图表
            safe_name = exp_name.replace('/', '_').replace(' ', '_')
            output_path = results_dir / f"{safe_name}_comparison.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ 图表已保存: {output_path}")

            plt.close(fig)

            # 记录结果
            target_gray = 255 * 0.95
            result = {
                'experiment': exp_name,
                'initial_gray': history[0]['gray'],
                'initial_gain': history[0]['gain'],
                'iterative_gray': history[-1]['gray'],
                'iterative_gain': history[-1]['gain'],
                'iterations': len(history) - 1,
                'single_gray': single_gray,
                'single_gain': single_gain_val,
                'target_gray': target_gray,
                'iterative_error': abs(history[-1]['gray'] - target_gray),
                'single_error': abs(single_gray - target_gray) if single_gray else None
            }
            all_results.append(result)

        except Exception as e:
            print(f"\n✗ 错误: {e}")
            import traceback
            traceback.print_exc()

    # 生成汇总报告
    if len(all_results) > 0:
        generate_summary_report(all_results, results_dir)

    print(f"\n{'='*70}")
    print("验证完成！")
    print(f"结果保存在: {results_dir}")
    print(f"{'='*70}\n")


def generate_summary_report(results, results_dir):
    """生成汇总报告"""
    print("\n生成汇总报告...")

    # 创建汇总图表
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. 误差对比（迭代 vs 单次）
    ax1 = fig.add_subplot(gs[0, 0])

    experiments_list = [r['experiment'] for r in results]
    iter_errors = [r['iterative_error'] for r in results]
    single_errors = [r['single_error'] if r['single_error'] else 0 for r in results]

    x = np.arange(len(experiments_list))
    width = 0.35

    bars1 = ax1.bar(x - width/2, iter_errors, width, label='迭代优化', color='green', alpha=0.8)
    bars2 = ax1.bar(x + width/2, single_errors, width, label='单次优化', color='orange', alpha=0.8)

    ax1.set_xlabel('实验类型', fontsize=11)
    ax1.set_ylabel('灰度误差', fontsize=11)
    ax1.set_title('优化方法误差对比', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(experiments_list, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. 改善率对比
    ax2 = fig.add_subplot(gs[0, 1])

    iter_improvements = []
    single_improvements = []

    for r in results:
        init_error = abs(r['initial_gray'] - r['target_gray'])
        iter_improvement = (init_error - r['iterative_error']) / init_error * 100 if init_error > 0 else 0
        single_improvement = (init_error - r['single_error']) / init_error * 100 if r['single_error'] and init_error > 0 else 0

        iter_improvements.append(iter_improvement)
        single_improvements.append(single_improvement)

    bars1 = ax2.bar(x - width/2, iter_improvements, width, label='迭代优化 (%)', color='green', alpha=0.8)
    bars2 = ax2.bar(x + width/2, single_improvements, width, label='单次优化 (%)', color='orange', alpha=0.8)

    ax2.set_xlabel('实验类型', fontsize=11)
    ax2.set_ylabel('改善率 (%)', fontsize=11)
    ax2.set_title('优化改善率对比', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(experiments_list, rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

    # 3. 迭代次数分布
    ax3 = fig.add_subplot(gs[1, 0])

    iterations = [r['iterations'] for r in results]
    ax3.bar(experiments_list, iterations, color='steelblue', alpha=0.8, edgecolor='black')

    for i, v in enumerate(iterations):
        ax3.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax3.set_xlabel('实验类型', fontsize=11)
    ax3.set_ylabel('迭代次数', fontsize=11)
    ax3.set_title('收敛所需的迭代次数', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(experiments_list, rotation=45, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 最终灰度值对比
    ax4 = fig.add_subplot(gs[1, 1])

    initial_grays = [r['initial_gray'] for r in results]
    iterative_grays = [r['iterative_gray'] for r in results]
    single_grays = [r['single_gray'] if r['single_gray'] else 0 for r in results]
    target = 255 * 0.95

    width_val = 0.25
    x_val = np.arange(len(experiments_list))

    bars1 = ax4.bar(x_val - width_val, initial_grays, width_val, label='初始', color='red', alpha=0.8)
    bars2 = ax4.bar(x_val, iterative_grays, width_val, label='迭代优化', color='green', alpha=0.8)
    bars3 = ax4.bar(x_val + width_val, single_grays, width_val, label='单次优化', color='orange', alpha=0.8)

    ax4.axhline(y=target, color='blue', linestyle='--', linewidth=2, label='目标灰度')

    ax4.set_xlabel('实验类型', fontsize=11)
    ax4.set_ylabel('灰度值', fontsize=11)
    ax4.set_title('各方法最终灰度值对比', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_val)
    ax4.set_xticklabels(experiments_list, rotation=45, ha='right', fontsize=9)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('迭代优化 vs 单次优化 - 综合对比', fontsize=14, fontweight='bold')

    summary_path = results_dir / "iterative_vs_single_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"✓ 汇总图表已保存: {summary_path}")

    plt.close(fig)

    # 保存文本报告
    report_path = results_dir / "iterative_validation_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("迭代优化 vs 单次优化 - 验证报告\n")
        f.write("="*70 + "\n\n")

        for r in results:
            f.write(f"\n实验: {r['experiment']}\n")
            f.write("-"*70 + "\n")
            f.write(f"初始状态:\n")
            f.write(f"  增益: {r['initial_gain']:.2f} dB\n")
            f.write(f"  灰度: {r['initial_gray']:.2f}\n")
            f.write(f"  误差: {abs(r['initial_gray'] - r['target_gray']):.2f}\n\n")

            f.write(f"迭代优化 ({r['iterations']}次迭代):\n")
            f.write(f"  最终增益: {r['iterative_gain']:.2f} dB\n")
            f.write(f"  最终灰度: {r['iterative_gray']:.2f}\n")
            f.write(f"  误差: {r['iterative_error']:.2f}\n")

            init_error = abs(r['initial_gray'] - r['target_gray'])
            improvement = (init_error - r['iterative_error'])
            improvement_rate = (improvement / init_error * 100) if init_error > 0 else 0
            f.write(f"  改善: {improvement:.2f} ({improvement_rate:.1f}%)\n\n")

            if r['single_gray']:
                f.write(f"单次优化:\n")
                f.write(f"  预测增益: {r['single_gain']:.2f} dB\n")
                f.write(f"  实际灰度: {r['single_gray']:.2f}\n")
                f.write(f"  误差: {r['single_error']:.2f}\n")

                single_improvement = (init_error - r['single_error'])
                single_improvement_rate = (single_improvement / init_error * 100) if init_error > 0 else 0
                f.write(f"  改善: {single_improvement:.2f} ({single_improvement_rate:.1f}%)\n\n")

                # 方法对比
                f.write(f"方法对比:\n")
                if r['iterative_error'] < r['single_error']:
                    better = "迭代优化"
                    diff = r['single_error'] - r['iterative_error']
                else:
                    better = "单次优化"
                    diff = r['iterative_error'] - r['single_error']
                f.write(f"  更优方法: {better}\n")
                f.write(f"  误差差异: {diff:.2f}\n")

    print(f"✓ 文本报告已保存: {report_path}")


if __name__ == "__main__":
    validate_all_experiments()
