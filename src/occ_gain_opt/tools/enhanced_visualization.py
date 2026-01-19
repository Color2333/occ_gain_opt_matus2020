"""
增强可视化模块 - 生成更直观的效果展示
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from ..config import ROIStrategy
from ..data_acquisition import DataAcquisition
from ..gain_optimization import GainOptimizer

matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_comprehensive_visualization():
    """创建综合可视化展示"""
    print("\n生成增强可视化...")

    data_acq = DataAcquisition()
    optimizer = GainOptimizer(data_acq)

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('光相机通信增益优化 - 完整效果展示\n'
                 'Optical Camera Communication: Gain Optimization Results',
                 fontsize=18, fontweight='bold')

    ax1 = fig.add_subplot(gs[0, :2])

    led_duty_cycle = 50
    background_light = 50
    led_intensity = (led_duty_cycle / 100.0) * 255

    gains = np.linspace(0, 20, 21)
    gray_values = []

    for gain in gains:
        image = data_acq.capture_image(led_intensity, gain, background_light)
        roi_mask = data_acq.select_roi(strategy=ROIStrategy.CENTER)
        stats = data_acq.get_roi_statistics(image, roi_mask)
        gray_values.append(stats['mean'])

    ax1.plot(gains, gray_values, 'b-o', linewidth=3, markersize=8,
             label='ROI灰度值', markerfacecolor='white', markeredgewidth=2)
    ax1.axhline(y=255, color='r', linestyle='--', linewidth=2,
                label='目标值(255)', alpha=0.7)

    max_idx = np.argmax(gray_values)
    ax1.plot(gains[max_idx], gray_values[max_idx], 'r*', markersize=20,
             label=f'最大值({gray_values[max_idx]:.1f} @ {gains[max_idx]:.1f}dB)',
             zorder=5)

    ax1.fill_between(gains, 0, gray_values, alpha=0.3, color='blue')

    ax1.set_xlabel('增益 (dB)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('ROI平均灰度值', fontsize=14, fontweight='bold')
    ax1.set_title(f'增益-灰度响应曲线\n(LED={led_duty_cycle}%, 背景={background_light})',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 280])

    table_data = [
        ['增益', '灰度值', '增幅'],
        ['0dB', f'{gray_values[0]:.1f}', '-'],
        ['10dB', f'{gray_values[10]:.1f}', f'+{((gray_values[10]-gray_values[0])/gray_values[0]*100):.1f}%'],
        ['20dB', f'{gray_values[20]:.1f}', f'+{((gray_values[20]-gray_values[0])/gray_values[0]*100):.1f}%']
    ]
    table = ax1.table(cellText=table_data, cellLoc='center',
                      bbox=[0.65, 0.55, 0.3, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    ax2 = fig.add_subplot(gs[0, 2:])
    result = optimizer.optimize_gain(
        led_duty_cycle=led_duty_cycle,
        initial_gain=0.0,
        background_light=background_light
    )

    history = result['history']
    iterations = [h['iteration'] for h in history]
    gains_history = [h['gain'] for h in history]
    gray_history = [h['mean_gray'] for h in history]

    ax2.plot(iterations, gains_history, 's-', linewidth=3, markersize=12,
             label='增益', color='green', markerfacecolor='white')
    ax2.set_xlabel('迭代次数', fontsize=13, fontweight='bold')
    ax2.set_ylabel('增益 (dB)', fontsize=13, fontweight='bold', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.grid(True, alpha=0.3)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(iterations, gray_history, 'o-', linewidth=3, markersize=12,
                  label='灰度值', color='blue', markerfacecolor='white')
    ax2_twin.axhline(y=255, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax2_twin.set_ylabel('灰度值', fontsize=13, fontweight='bold', color='blue')
    ax2_twin.tick_params(axis='y', labelcolor='blue')

    ax2.set_title(f'优化过程收敛\n(最终: {result["optimal_gain"]:.1f}dB, '
                  f'{result["final_gray"]:.1f}, {result["iterations"]}次迭代)',
                  fontsize=13, fontweight='bold')

    for it, g in zip(iterations, gains_history):
        ax2.annotate(f'{it}', (it, g), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')

    ax3a = fig.add_subplot(gs[1, 0])
    ax3b = fig.add_subplot(gs[1, 1])
    ax3c = fig.add_subplot(gs[1, 2])
    ax3d = fig.add_subplot(gs[1, 3])

    test_gains = [0, 7, 14, 20]
    axes = [ax3a, ax3b, ax3c, ax3d]

    for gain, ax in zip(test_gains, axes):
        image = data_acq.capture_image(led_intensity, gain, background_light)
        roi_mask = data_acq.select_roi(strategy=ROIStrategy.CENTER)

        ax.imshow(image, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'增益 {gain}dB\n灰度均值={np.mean(image[roi_mask==1]):.1f}',
                     fontsize=11, fontweight='bold')
        ax.axis('off')

        y, x = np.where(roi_mask == 1)
        if len(x) > 0:
            ax.plot([x.min(), x.max(), x.max(), x.min(), x.min()],
                    [y.min(), y.min(), y.max(), y.max(), y.min()],
                    'r-', linewidth=2)

    ax4 = fig.add_subplot(gs[2, :2])
    background_lights = [20, 50, 100, 150]
    led_duty_cycles = [30, 50, 70]

    X = np.arange(len(background_lights))
    width = 0.25

    for i, led_dc in enumerate(led_duty_cycles):
        max_gray_values = []
        for bg_light in background_lights:
            temp_data_acq = DataAcquisition()
            temp_optimizer = GainOptimizer(temp_data_acq)
            result = temp_optimizer.optimize_gain(led_dc, 0.0, bg_light)
            max_gray_values.append(result['final_gray'])

        offset = (i - 1) * width
        ax4.bar(X + offset, max_gray_values, width,
                label=f'LED {led_dc}%', alpha=0.8)

    ax4.axhline(y=255, color='r', linestyle='--', linewidth=2, label='目标值(255)')
    ax4.set_xlabel('背景光强', fontsize=13, fontweight='bold')
    ax4.set_ylabel('最终灰度值', fontsize=13, fontweight='bold')
    ax4.set_title('不同条件下的优化效果', fontsize=13, fontweight='bold')
    ax4.set_xticks(X)
    ax4.set_xticklabels([f'{bg}' for bg in background_lights])
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 280])

    ax5 = fig.add_subplot(gs[2, 2:])
    image_low = data_acq.capture_image(led_intensity, 0, background_light)
    image_high = data_acq.capture_image(led_intensity, 20, background_light)

    roi_mask = data_acq.select_roi(strategy=ROIStrategy.CENTER)
    gray_low = image_low[roi_mask == 1]
    gray_high = image_high[roi_mask == 1]

    ax5.hist(gray_low.flatten(), bins=30, alpha=0.5, label='0dB',
             color='blue', edgecolor='black')
    ax5.hist(gray_high.flatten(), bins=30, alpha=0.5, label='20dB',
             color='red', edgecolor='black')

    ax5.axvline(np.mean(gray_low), color='blue', linestyle='--',
                linewidth=2, label=f'0dB均值={np.mean(gray_low):.1f}')
    ax5.axvline(np.mean(gray_high), color='red', linestyle='--',
                linewidth=2, label=f'20dB均值={np.mean(gray_high):.1f}')

    ax5.set_xlabel('灰度值', fontsize=13, fontweight='bold')
    ax5.set_ylabel('像素数量', fontsize=13, fontweight='bold')
    ax5.set_title('ROI灰度值分布对比', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    output_path = 'results/plots/enhanced_comprehensive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 综合可视化已保存: {output_path}")
    plt.close()


def create_gain_comparison_plot():
    """创建增益对比图"""
    print("\n生成增益对比可视化...")
    data_acq = DataAcquisition()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('不同增益下的图像效果对比', fontsize=16, fontweight='bold')

    led_duty_cycle = 50
    background_light = 50
    led_intensity = (led_duty_cycle / 100.0) * 255

    gains = [0, 4, 8, 12, 16, 20]

    for gain, ax in zip(gains, axes.flat):
        image = data_acq.capture_image(led_intensity, gain, background_light)
        roi_mask = data_acq.select_roi(strategy=ROIStrategy.CENTER)

        im = ax.imshow(image, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'增益 {gain}dB', fontsize=14, fontweight='bold')

        stats = data_acq.get_roi_statistics(image, roi_mask)
        textstr = f'ROI统计:\n均值: {stats["mean"]:.1f}\n标准差: {stats["std"]:.1f}\n' \
                  f'最小: {stats["min"]:.1f}\n最大: {stats["max"]:.1f}\n' \
                  f'饱和比: {stats["saturated_ratio"]*100:.1f}%'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        y, x = np.where(roi_mask == 1)
        if len(x) > 0:
            ax.plot([x.min(), x.max(), x.max(), x.min(), x.min()],
                    [y.min(), y.min(), y.max(), y.max(), y.min()],
                    'r-', linewidth=2)

        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('灰度值', rotation=270, labelpad=15)

    plt.tight_layout()
    output_path = 'results/plots/gain_comparison_grid.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 增益对比图已保存: {output_path}")
    plt.close()


def create_optimization_animation():
    """创建优化过程的详细展示"""
    print("\n生成优化过程详细展示...")
    data_acq = DataAcquisition()
    optimizer = GainOptimizer(data_acq)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('增益优化过程详细分析', fontsize=16, fontweight='bold')

    result = optimizer.optimize_gain(50, 0.0, 50)
    history = result['history']

    ax1 = fig.add_subplot(gs[0, 0])
    iterations = [h['iteration'] for h in history]
    gains = [h['gain'] for h in history]
    grays = [h['mean_gray'] for h in history]

    ax1.plot(iterations, gains, marker='s', linestyle='-', linewidth=3, markersize=12,
             color='green', markerfacecolor='white', markeredgewidth=2)
    ax1.fill_between(iterations, 0, gains, alpha=0.3, color='green')

    for it, g in zip(iterations, gains):
        ax1.annotate(f'{g:.1f}', (it, g), textcoords="offset points",
                     xytext=(0, 15), ha='center', fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    ax1.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
    ax1.set_ylabel('增益 (dB)', fontsize=12, fontweight='bold')
    ax1.set_title('增益变化曲线', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 22])

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iterations, grays, marker='o', linestyle='-', linewidth=3, markersize=12,
             color='blue', markerfacecolor='white', markeredgewidth=2)
    ax2.axhline(y=255, color='r', linestyle='--', linewidth=2, label='目标255')
    ax2.fill_between(iterations, 0, grays, alpha=0.3, color='blue')

    for it, g in zip(iterations, grays):
        ax2.annotate(f'{g:.1f}', (it, g), textcoords="offset points",
                     xytext=(0, 15), ha='center', fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='cyan', alpha=0.5))

    ax2.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ROI灰度值', fontsize=12, fontweight='bold')
    ax2.set_title('灰度值变化曲线', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 280])

    ax3 = fig.add_subplot(gs[0, 2])
    scatter = ax3.scatter(gains, grays, c=iterations, s=300,
                          cmap='viridis', edgecolors='black', linewidth=2,
                          vmin=0, vmax=len(iterations))
    ax3.plot(gains, grays, 'k--', alpha=0.5, linewidth=2)
    ax3.axhline(y=255, color='r', linestyle='--', linewidth=2, alpha=0.5)

    for i, (g, gray) in enumerate(zip(gains, grays)):
        ax3.annotate(f'Iter{i}', (g, gray), textcoords="offset points",
                     xytext=(5, 5), ha='left', fontsize=10, fontweight='bold')

    ax3.set_xlabel('增益 (dB)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('ROI灰度值', fontsize=12, fontweight='bold')
    ax3.set_title('增益-灰度关系', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('迭代次数', fontsize=11)

    led_intensity = 127.5
    for i in range(min(3, len(history))):
        ax = fig.add_subplot(gs[1, i])
        gain = history[i]['gain']
        image = data_acq.capture_image(led_intensity, gain, 50)

        im = ax.imshow(image, cmap='gray')
        ax.set_title(f'迭代 {i}: 增益={gain:.1f}dB, 灰度={grays[i]:.1f}',
                     fontsize=12, fontweight='bold')

        roi_mask = data_acq.select_roi(strategy=ROIStrategy.CENTER)
        y, x = np.where(roi_mask == 1)
        if len(x) > 0:
            ax.plot([x.min(), x.max(), x.max(), x.min(), x.min()],
                    [y.min(), y.min(), y.max(), y.max(), y.min()],
                    'r-', linewidth=2)

        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = 'results/plots/optimization_process_detail.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 优化过程详情已保存: {output_path}")
    plt.close()


def main():
    print("\n" + "=" * 70)
    print("生成增强可视化图表")
    print("=" * 70)

    create_comprehensive_visualization()
    create_gain_comparison_plot()
    create_optimization_animation()

    print("\n" + "=" * 70)
    print("所有增强可视化生成完成!")
    print("=" * 70)
    print("\n生成的文件:")
    print("  1. enhanced_comprehensive.png - 综合效果展示")
    print("  2. gain_comparison_grid.png - 增益对比网格")
    print("  3. optimization_process_detail.png - 优化过程详情")
    print("\n位置: results/plots/")
    print("=" * 70)


if __name__ == "__main__":
    main()
