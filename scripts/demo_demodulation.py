"""
OOK 解调演示脚本
展示完整的同步头检测和数据包 ROI 定位流水线
"""

import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 确保能导入项目包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from occ_gain_opt.demodulation import OOKDemodulator

# 中文字体设置 (macOS 优先 Hiragino Sans GB, 其次 Arial Unicode MS)
plt.rcParams['font.sans-serif'] = [
    'Hiragino Sans GB', 'Arial Unicode MS', 'PingFang SC',
    'Heiti TC', 'SimHei', 'DejaVu Sans',
]
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ---- 配置 ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'ISO-Texp')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'demodulation')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 测试图片: 不同增益条件
TEST_IMAGES = [
    # (相对路径, 描述)
    ('bubble/ISO/52600_640_p32_bubble_1_4_1.jpg', 'bubble ISO=640 (中增益)'),
    ('bubble/ISO/52600_35_p32_bubble_1_4_1.jpg',  'bubble ISO=35 (低增益)'),
    ('bubble/ISO/52600_3200_p32_bubble_1_4_1.jpg', 'bubble ISO=3200 (高增益)'),
]


def visualize_demodulation(image_path: str, label: str, output_path: str):
    """对单张图片运行解调并生成可视化"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [跳过] 无法读取: {image_path}")
        return None

    print(f"  处理: {label}")
    print(f"    图像尺寸: {image.shape}")

    demod = OOKDemodulator()
    result = demod.demodulate(image)

    col_start, col_end = result.col_bounds
    print(f"    LED 列范围: [{col_start}, {col_end}]")
    print(f"    二值化阈值: {result.threshold:.1f}")
    print(f"    位周期: {result.bit_period:.1f} 行/bit")
    print(f"    比特序列长度: {len(result.bit_sequence)}")
    print(f"    同步头: {result.sync_pattern}")
    print(f"    同步头位置(bit): {result.sync_positions_bit}")
    print(f"    同步头位置(row): {result.sync_positions_row}")
    print(f"    数据包数: {len(result.packets)}")
    print(f"    置信度: {result.confidence:.3f}")
    if result.stats:
        print(f"    眼图开度: {result.stats.get('eye_opening', 0):.1f}")
        print(f"    SNR: {result.stats.get('snr_db', 0):.1f} dB")

    # ---- 6 面板可视化 ----
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'OOK 解调分析: {label}', fontsize=14, fontweight='bold')

    # Panel 1: 原图 + LED 列边界
    ax = axes[0, 0]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.axvline(col_start, color='r', linewidth=1.5, linestyle='--', label='LED 列边界')
    ax.axvline(col_end, color='r', linewidth=1.5, linestyle='--')
    ax.set_title('原图 + LED 列范围')
    ax.legend(fontsize=8)

    # Panel 2: 行均值曲线 + 阈值
    ax = axes[0, 1]
    rows = np.arange(len(result.row_profile))
    ax.plot(rows, result.row_profile, 'b-', linewidth=0.5, label='行均值')
    ax.axhline(result.threshold, color='r', linewidth=1, linestyle='--',
               label=f'阈值={result.threshold:.1f}')
    ax.set_xlabel('行号')
    ax.set_ylabel('灰度均值')
    ax.set_title('行均值曲线 + 二值化阈值')
    ax.legend(fontsize=8)

    # Panel 3: 二值化 + 采样点
    ax = axes[1, 0]
    ax.fill_between(rows, result.binary_profile, alpha=0.3, color='green', label='二值化')
    if len(result.sample_positions) > 0:
        sp = np.clip(np.round(result.sample_positions).astype(int),
                     0, len(result.binary_profile) - 1)
        ax.plot(result.sample_positions, result.binary_profile[sp],
                'r|', markersize=8, label=f'采样点 (T={result.bit_period:.1f})')
    ax.set_xlabel('行号')
    ax.set_ylabel('二值')
    ax.set_title(f'二值化信号 + 位采样 (周期={result.bit_period:.1f} 行)')
    ax.legend(fontsize=8)

    # Panel 4: 比特序列 + 同步头位置
    ax = axes[1, 1]
    if len(result.bit_sequence) > 0:
        bit_idx = np.arange(len(result.bit_sequence))
        colors = ['green' if b == 1 else 'black' for b in result.bit_sequence]
        ax.bar(bit_idx, result.bit_sequence, color=colors, width=0.8)
        for sp_bit in result.sync_positions_bit:
            sync_len = len(result.sync_pattern) if result.sync_pattern is not None else 0
            ax.axvspan(sp_bit, sp_bit + sync_len, alpha=0.3, color='red',
                       label='同步头' if sp_bit == result.sync_positions_bit[0] else '')
        ax.set_xlabel('比特索引')
        ax.set_ylabel('值')
        ax.set_title(f'比特序列 ({len(result.bit_sequence)} bits, '
                     f'同步头×{len(result.sync_positions_bit)})')
        if result.sync_positions_bit:
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, '无法提取比特序列', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('比特序列')

    # Panel 5: 原图 + Sync ROI 叠加
    ax = axes[2, 0]
    overlay = img_rgb.copy()
    roi_colored = np.zeros_like(overlay)
    roi_colored[:, :, 0] = 255  # 红色
    mask_3d = np.stack([result.roi_mask] * 3, axis=-1)
    overlay = np.where(mask_3d, cv2.addWeighted(overlay, 0.6, roi_colored, 0.4, 0),
                       overlay)
    ax.imshow(overlay)
    # 标注同步头行位置
    for i, sr in enumerate(result.sync_positions_row):
        ax.axhline(sr, color='yellow', linewidth=1, linestyle='-')
        ax.text(5, sr - 5, f'Sync #{i+1}', color='yellow', fontsize=8,
                fontweight='bold')
    ax.set_title(f'同步 ROI 叠加 (覆盖 {np.sum(result.roi_mask > 0)} 像素)')

    # Panel 6: 信号质量统计
    ax = axes[2, 1]
    ax.axis('off')
    stats_text = [
        f"图像尺寸: {image.shape[1]}×{image.shape[0]}",
        f"LED 列范围: [{col_start}, {col_end}] ({col_end - col_start} cols)",
        f"二值化阈值: {result.threshold:.2f}",
        f"位周期: {result.bit_period:.2f} 行/bit",
        f"总比特数: {len(result.bit_sequence)}",
        f"",
        f"同步头模式: {result.sync_pattern}",
        f"同步头位置(bit): {result.sync_positions_bit}",
        f"同步头位置(row): {result.sync_positions_row}",
        f"完整数据包: {len(result.packets)} 个",
        f"检测置信度: {result.confidence:.3f}",
        f"",
        f"信号质量:",
        f"  眼图开度: {result.stats.get('eye_opening', 0):.2f}",
        f"  亮条均值: {result.stats.get('bright_mean', 0):.2f}",
        f"  暗条均值: {result.stats.get('dark_mean', 0):.2f}",
        f"  SNR: {result.stats.get('snr_db', 0):.1f} dB",
    ]
    for i, pkt in enumerate(result.packets):
        bits_str = ''.join(str(b) for b in pkt)
        stats_text.append(f"  包{i+1}: {bits_str}")

    ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('解调统计信息')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    已保存: {output_path}")
    return result


def main():
    print("=" * 60)
    print("OOK 解调演示 — 同步头检测 & 数据包 ROI 定位")
    print("=" * 60)

    results = {}
    for rel_path, label in TEST_IMAGES:
        img_path = os.path.join(DATA_DIR, rel_path)
        if not os.path.exists(img_path):
            print(f"\n[跳过] 文件不存在: {img_path}")
            continue
        safe_name = rel_path.replace('/', '_').replace('.jpg', '')
        out_path = os.path.join(OUTPUT_DIR, f'demod_{safe_name}.png')
        print(f"\n--- {label} ---")
        result = visualize_demodulation(img_path, label, out_path)
        if result is not None:
            results[label] = result

    # ---- 汇总对比图 ----
    if len(results) >= 2:
        print("\n生成汇总对比图...")
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
        if len(results) == 1:
            axes = [axes]
        for ax, (label, res) in zip(axes, results.items()):
            ax.plot(res.row_profile, 'b-', linewidth=0.5)
            ax.axhline(res.threshold, color='r', linestyle='--', linewidth=0.8)
            for sr in res.sync_positions_row:
                ax.axvline(sr, color='orange', linestyle='-', linewidth=0.8, alpha=0.7)
            ax.set_title(f'{label}\nT={res.bit_period:.1f}, pkts={len(res.packets)}',
                         fontsize=10)
            ax.set_xlabel('行号')
            ax.set_ylabel('灰度均值')

        plt.suptitle('不同增益条件下的解调对比', fontsize=13, fontweight='bold')
        plt.tight_layout()
        summary_path = os.path.join(OUTPUT_DIR, 'demod_summary.png')
        plt.savefig(summary_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {summary_path}")

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")
    print("完成!")


if __name__ == '__main__':
    main()
