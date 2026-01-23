"""
实验数据验证脚本
使用真实实验图片验证增益优化算法
"""

import sys
import os
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from occ_gain_opt.experiment_loader import ExperimentLoader, print_experiment_summary
from occ_gain_opt.data_acquisition import DataAcquisition
from occ_gain_opt.gain_optimization import GainOptimizer
from occ_gain_opt.performance_evaluation import PerformanceEvaluator
from occ_gain_opt.config import ROIStrategy


def analyze_roi_on_real_images(images, loader, roi_strategy=ROIStrategy.AUTO_BRIGHTNESS):
    """
    在真实图片上分析ROI

    Args:
        images: ExperimentImage列表
        roi_strategy: ROI选择策略

    Returns:
        分析结果列表
    """
    results = []

    print(f"\n分析 {len(images)} 张图片的ROI...")

    for i, exp_img in enumerate(images):
        if not exp_img.load():
            print(f"无法加载图片: {exp_img.filepath}")
            continue

        gray = exp_img.gray_image
        h, w = gray.shape

        # 为每张图片创建适当尺寸的 DataAcquisition
        data_acq = DataAcquisition(width=w, height=h)
        roi_mask = data_acq.select_roi(strategy=roi_strategy, image=gray)
        stats = data_acq.get_roi_statistics(gray, roi_mask)

        result = {
            'image': exp_img,
            'stats': stats,
            'roi_mask': roi_mask,
            'equivalent_gain': exp_img.calculate_equivalent_gain()
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{len(images)} 张图片")

    return results


def find_best_gain_from_real_images(images, loader, target_gray=255 * 0.95):
    """
    从真实图片中寻找最佳增益设置

    Args:
        images: ExperimentImage列表
        target_gray: 目标灰度值（考虑安全因子）

    Returns:
        最佳增益和对应的图片信息
    """
    print("\n从真实图片中寻找最佳增益...")

    # 分析所有图片的ROI
    results = analyze_roi_on_real_images(images, loader)

    # 找到最接近目标灰度值且不过饱和的图片
    best_result = None
    best_score = float('inf')

    for result in results:
        stats = result['stats']
        # 评分函数：考虑与目标的距离和饱和度
        distance = abs(stats['mean'] - target_gray)
        saturation_penalty = stats['saturated_ratio'] * 1000  # 严重惩罚饱和
        score = distance + saturation_penalty

        # 只考虑饱和度低于10%的图片
        if stats['saturated_ratio'] < 0.1 and score < best_score:
            best_score = score
            best_result = result

    if best_result:
        img = best_result['image']
        stats = best_result['stats']
        gain = best_result['equivalent_gain']

        print(f"\n最佳增益设置:")
        print(f"  等效增益: {gain:.2f} dB")
        print(f"  曝光时间: {img.exposure_time:.6f} 秒 (1/{int(1/img.exposure_time)})")
        print(f"  ISO: {img.iso:.0f}")
        print(f"  平均灰度: {stats['mean']:.2f}")
        print(f"  饱和度: {stats['saturated_ratio']*100:.2f}%")
        print(f"  条件: {img.condition}")

    return best_result


def verify_gain_optimization_on_real_data(experiment_type="bubble", image_type="ISO"):
    """
    在真实数据上验证增益优化算法

    Args:
        experiment_type: 实验类型
        image_type: 图片类型
    """
    print(f"\n{'='*60}")
    print(f"验证增益优化算法 - {experiment_type}/{image_type}")
    print(f"{'='*60}")

    # 加载数据
    loader = ExperimentLoader()
    images = loader.load_experiment(experiment_type, image_type)

    if not images:
        print(f"没有找到 {experiment_type}/{image_type} 的图片")
        return

    print(f"\n加载了 {len(images)} 张图片")

    # 找到最佳增益
    best_result = find_best_gain_from_real_images(images, loader)

    if not best_result:
        print("未找到合适的图片")
        return

    # 分析增益-灰度关系
    print("\n分析增益-灰度关系...")
    results = analyze_roi_on_real_images(images, loader)

    gains = [r['equivalent_gain'] for r in results]
    gray_means = [r['stats']['mean'] for r in results]
    saturated_ratios = [r['stats']['saturated_ratio'] for r in results]

    # 绘制关系图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 增益 vs 平均灰度
    axes[0, 0].scatter(gains, gray_means, alpha=0.6)
    axes[0, 0].axhline(y=255 * 0.95, color='r', linestyle='--', label='目标灰度 (95%)')
    axes[0, 0].set_xlabel('等效增益 (dB)')
    axes[0, 0].set_ylabel('ROI平均灰度值')
    axes[0, 0].set_title('增益 vs 灰度值关系')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 增益 vs 饱和度
    axes[0, 1].scatter(gains, saturated_ratios, alpha=0.6, color='orange')
    axes[0, 1].axhline(y=0.1, color='r', linestyle='--', label='10%饱和阈值')
    axes[0, 1].set_xlabel('等效增益 (dB)')
    axes[0, 1].set_ylabel('饱和比例')
    axes[0, 1].set_title('增益 vs 饱和度')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 灰度值分布直方图
    axes[1, 0].hist(gray_means, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=255 * 0.95, color='r', linestyle='--', label='目标灰度 (95%)')
    axes[1, 0].axvline(x=best_result['stats']['mean'], color='g',
                      linestyle='-', linewidth=2, label='最佳图片灰度')
    axes[1, 0].set_xlabel('ROI平均灰度值')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title('灰度值分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 增益分布直方图
    axes[1, 1].hist(gains, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=best_result['equivalent_gain'], color='g',
                      linestyle='-', linewidth=2, label='最佳增益')
    axes[1, 1].set_xlabel('等效增益 (dB)')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].set_title('增益分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存结果
    results_dir = Path(__file__).parent.parent / "results" / "experiment_verification"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"{experiment_type}_{image_type}_verification.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {output_path}")

    # 保存分析报告
    report_path = results_dir / f"{experiment_type}_{image_type}_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"增益优化算法验证报告 - {experiment_type}/{image_type}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"数据集规模: {len(images)} 张图片\n\n")

        f.write(f"最佳增益设置:\n")
        f.write(f"  等效增益: {best_result['equivalent_gain']:.2f} dB\n")
        f.write(f"  曝光时间: {best_result['image'].exposure_time:.6f} 秒\n")
        f.write(f"  ISO: {best_result['image'].iso:.0f}\n")
        f.write(f"  平均灰度: {best_result['stats']['mean']:.2f}\n")
        f.write(f"  饱和度: {best_result['stats']['saturated_ratio']*100:.2f}%\n")
        f.write(f"  条件: {best_result['image'].condition}\n\n")

        f.write(f"统计信息:\n")
        f.write(f"  增益范围: {min(gains):.2f} - {max(gains):.2f} dB\n")
        f.write(f"  灰度范围: {min(gray_means):.2f} - {max(gray_means):.2f}\n")
        f.write(f"  平均饱和度: {np.mean(saturated_ratios)*100:.2f}%\n")

    print(f"报告已保存到: {report_path}")

    return best_result, results


def compare_iso_vs_texp(experiment_type="bubble"):
    """
    比较ISO和Texp两种类型的结果
    """
    print(f"\n{'='*60}")
    print(f"比较 ISO vs Texp - {experiment_type}")
    print(f"{'='*60}")

    loader = ExperimentLoader()

    # 分别加载ISO和Texp数据
    iso_images = loader.load_experiment(experiment_type, "ISO")
    texp_images = loader.load_experiment(experiment_type, "Texp")

    print(f"\nISO图片: {len(iso_images)} 张")
    print(f"Texp图片: {len(texp_images)} 张")

    # 找到各自的最佳设置
    best_iso = find_best_gain_from_real_images(iso_images, loader)
    best_texp = find_best_gain_from_real_images(texp_images, loader)

    if best_iso and best_texp:
        print(f"\n{'='*60}")
        print("比较结果:")
        print(f"{'='*60}")
        print(f"\nISO最佳设置:")
        print(f"  增益: {best_iso['equivalent_gain']:.2f} dB")
        print(f"  灰度: {best_iso['stats']['mean']:.2f}")
        print(f"  饱和度: {best_iso['stats']['saturated_ratio']*100:.2f}%")

        print(f"\nTexp最佳设置:")
        print(f"  增益: {best_texp['equivalent_gain']:.2f} dB")
        print(f"  灰度: {best_texp['stats']['mean']:.2f}")
        print(f"  饱和度: {best_texp['stats']['saturated_ratio']*100:.2f}%")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("实验数据验证程序")
    print("="*60)

    # 打印实验摘要
    print_experiment_summary("bubble")
    print_experiment_summary("tap water")
    print_experiment_summary("turbidity")

    # 验证每个实验
    experiments = ["bubble", "tap water", "turbidity"]

    for exp_type in experiments:
        for img_type in ["ISO", "Texp"]:
            try:
                verify_gain_optimization_on_real_data(exp_type, img_type)
            except Exception as e:
                print(f"\n错误: {exp_type}/{img_type} - {e}")
                import traceback
                traceback.print_exc()

    # 比较ISO vs Texp
    for exp_type in experiments:
        try:
            compare_iso_vs_texp(exp_type)
        except Exception as e:
            print(f"\n错误: {exp_type} 比较 - {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("验证完成！")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
