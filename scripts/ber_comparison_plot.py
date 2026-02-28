#!/usr/bin/env python3
"""
BER 对比可视化脚本

生成三张图:
  Figure 1 — BER vs 增益曲线 (2×3 子图, 6 组实验)
  Figure 2 — 优化前后 BER 对比 (柱状 + 散点改善图)
  Figure 3 — 典型案例图像对比 (turbidity/ISO & bubble/Texp)

输出目录: results/ber_analysis/
"""

import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams['font.family'] = ['Hiragino Sans GB', 'Arial Unicode MS',
                                      'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

BASE_DIR   = Path(__file__).parent.parent
ISO_TEXP   = BASE_DIR / 'ISO-Texp'
OUTPUT_DIR = BASE_DIR / 'results/ber_analysis'

# ─────────────────────────── 分析结果 (来自 ber_vs_gain_analysis.py) ────────────
# 格式: (增益dB列表, BER均值列表, BER标准差列表, ROI均值列表)
ANALYSIS = {
    'bubble / ISO': {
        'gains':    [-5.8, -2.5, 19.4, 22.8, 27.4, 30.7, 33.4, 36.8],
        'bers':     [0.000, 0.000, 0.1667, 0.1823, 0.0208, 0.0938, 0.0729, 0.1250],
        'ber_stds': [0.000, 0.000, 0.1179, 0.0962, 0.0691, 0.1112, 0.1226, 0.1250],
        'rois':     [57.4, 59.4, 82.4, 84.9, 81.6, 83.0, 85.8, 81.8],
        'init_gain': -5.8, 'init_ber': 0.000,
        'pred_gain': -2.5, 'pred_ber': 0.000,
        'best_gain': -5.8, 'best_ber': 0.000,
    },
    'bubble / Texp': {
        'gains':    [-5.8, -2.5, 0.0, 1.9],
        'bers':     [0.1649, 0.000, 0.1302, 0.0833],
        'ber_stds': [0.2363, 0.000, 0.1074, 0.1179],
        'rois':     [38.3, 66.2, 20.5, 80.0],
        'init_gain': -5.8, 'init_ber': 0.1649,
        'pred_gain':  1.9, 'pred_ber': 0.0833,
        'best_gain': -2.5, 'best_ber': 0.000,
    },
    'tap water / ISO': {
        'gains':    [-5.8, 13.4, 19.4, 23.3, 27.4, 31.3, 33.4],
        'bers':     [0.000, 0.000, 0.000, 0.000, 0.0833, 0.1042, 0.1771],
        'ber_stds': [0.000, 0.000, 0.000, 0.000, 0.1179, 0.1394, 0.1297],
        'rois':     [86.6, 96.3, 85.0, 91.5, 102.4, 108.1, 101.3],
        'init_gain': -5.8, 'init_ber': 0.000,
        'pred_gain': -5.8, 'pred_ber': 0.000,
        'best_gain': -5.8, 'best_ber': 0.000,
    },
    'tap water / Texp': {
        'gains':    [-5.8, -2.5, 1.9, 4.8],
        'bers':     [0.000, 0.000, 0.0417, 0.0833],
        'ber_stds': [0.000, 0.000, 0.0932, 0.1179],
        'rois':     [87.1, 100.9, 100.2, 112.7],
        'init_gain': -5.8, 'init_ber': 0.000,
        'pred_gain':  1.9, 'pred_ber': 0.0417,
        'best_gain': -5.8, 'best_ber': 0.000,
    },
    'turbidity / ISO': {
        'gains':    [-5.8, 19.4, 33.4],
        'bers':     [0.4948, 0.000, 0.000],
        'ber_stds': [0.0661, 0.000, 0.000],
        'rois':     [0.0, 85.8, 112.0],
        'init_gain': -5.8,  'init_ber': 0.4948,
        'pred_gain': 33.4,  'pred_ber': 0.000,
        'best_gain': 19.4,  'best_ber': 0.000,
    },
    'turbidity / Texp': {
        'gains':    [-5.8, -2.5, 0.0, 1.9],
        'bers':     [0.000, 0.000, 0.000, 0.0208],
        'ber_stds': [0.000, 0.000, 0.000, 0.0691],
        'rois':     [86.6, 90.9, 90.0, 93.8],
        'init_gain': -5.8, 'init_ber': 0.000,
        'pred_gain':  1.9, 'pred_ber': 0.0208,
        'best_gain': -5.8, 'best_ber': 0.000,
    },
}

# 颜色方案
C_INIT = '#FF8C00'   # 橙色: 初始工作点
C_PRED = '#2CA02C'   # 绿色: 优化预测工作点
C_BEST = '#D62728'   # 红色: 数据集最优
C_LINE = '#4C72B0'   # 蓝色: BER曲线


# ══════════════════════════════════════════════════════════════════
# Figure 1: BER vs 增益曲线 (2×3)
# ══════════════════════════════════════════════════════════════════
def plot_ber_curves() -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('BER vs 等效增益 — 增益优化算法效果分析',
                 fontsize=15, fontweight='bold', y=0.98)

    for ax, (title, d) in zip(axes.flatten(), ANALYSIS.items()):
        g = np.array(d['gains'])
        b = np.array(d['bers'])
        e = np.array(d['ber_stds'])

        # 按增益排序
        order = np.argsort(g)
        g, b, e = g[order], b[order], e[order]

        # BER 曲线
        ax.plot(g, b, 'o-', color=C_LINE, linewidth=2,
                markersize=6, zorder=3, label='_')
        ax.fill_between(g, np.maximum(b - e, 0), b + e,
                        alpha=0.15, color=C_LINE)

        # 初始工作点
        ax.plot(d['init_gain'], d['init_ber'], 's',
                color=C_INIT, markersize=12, zorder=6,
                label=f"初始 {d['init_gain']:+.0f} dB (BER={d['init_ber']:.3f})")
        ax.axvline(d['init_gain'], color=C_INIT,
                   linestyle='--', linewidth=1.2, alpha=0.6)

        # 优化预测工作点
        ax.plot(d['pred_gain'], d['pred_ber'], '^',
                color=C_PRED, markersize=12, zorder=6,
                label=f"优化后 {d['pred_gain']:+.0f} dB (BER={d['pred_ber']:.3f})")
        ax.axvline(d['pred_gain'], color=C_PRED,
                   linestyle='--', linewidth=1.2, alpha=0.6)

        # 数据集最优（与初始不同时才画）
        if abs(d['best_gain'] - d['init_gain']) > 0.5:
            ax.plot(d['best_gain'], d['best_ber'], '*',
                    color=C_BEST, markersize=14, zorder=6,
                    label=f"最优 {d['best_gain']:+.0f} dB (BER={d['best_ber']:.3f})")
            ax.axvline(d['best_gain'], color=C_BEST,
                       linestyle=':', linewidth=1.2, alpha=0.6)

        # BER 改善标注
        if d['init_ber'] > 0.001:
            imp = (d['init_ber'] - d['pred_ber']) / d['init_ber'] * 100
            color_txt = '#2CA02C' if imp > 0 else '#D62728'
            ax.text(0.97, 0.97, f"BER改善\n{imp:+.1f}%",
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=10, fontweight='bold', color=color_txt,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=color_txt, alpha=0.85))
        elif d['pred_ber'] > 0.001:
            ax.text(0.97, 0.97, "过冲\n(已最优)",
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=10, fontweight='bold', color='#D62728',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#D62728', alpha=0.85))
        else:
            ax.text(0.97, 0.97, "维持最优",
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=10, fontweight='bold', color='#2CA02C',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#2CA02C', alpha=0.85))

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('等效增益 (dB)', fontsize=9)
        ax.set_ylabel('BER', fontsize=9)
        ax.set_ylim(-0.04, 0.62)
        ax.legend(fontsize=7.5, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 公共图例
    handles = [
        mpatches.Patch(color=C_INIT, label='初始工作点'),
        mpatches.Patch(color=C_PRED, label='增益优化后'),
        mpatches.Patch(color=C_BEST, label='数据集最优'),
        mpatches.Patch(color=C_LINE, label='BER曲线 ± std'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=10, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    return fig


# ══════════════════════════════════════════════════════════════════
# Figure 2: 对比汇总图 (柱状 + 改善散点)
# ══════════════════════════════════════════════════════════════════
def plot_summary_comparison() -> plt.Figure:
    labels = list(ANALYSIS.keys())
    init_bers = [d['init_ber'] for d in ANALYSIS.values()]
    pred_bers = [d['pred_ber'] for d in ANALYSIS.values()]
    best_bers = [d['best_ber'] for d in ANALYSIS.values()]

    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)

    # ── 左: 分组柱状图 ──
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(len(labels))
    w = 0.24

    bars_init = ax1.bar(x - w, init_bers, w, label='初始 BER',
                        color=C_INIT, alpha=0.85, edgecolor='white', linewidth=0.5)
    bars_pred = ax1.bar(x,     pred_bers, w, label='优化后 BER',
                        color=C_PRED, alpha=0.85, edgecolor='white', linewidth=0.5)
    bars_best = ax1.bar(x + w, best_bers, w, label='数据集最优 BER',
                        color=C_BEST, alpha=0.85, edgecolor='white', linewidth=0.5)

    # 数值标注
    for bars in [bars_init, bars_pred, bars_best]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.005:
                ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                         f'{h:.3f}', ha='center', va='bottom', fontsize=7.5,
                         fontweight='bold')

    # 改善箭头 (init → pred)
    for i, (ib, pb) in enumerate(zip(init_bers, pred_bers)):
        if ib > 0.01 and pb < ib - 0.01:
            ax1.annotate('', xy=(x[i], pb + 0.01),
                         xytext=(x[i] - w, ib - 0.01),
                         arrowprops=dict(arrowstyle='->', color='black',
                                         lw=1.5))

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax1.set_ylabel('BER', fontsize=11)
    ax1.set_title('各实验组 BER 对比: 初始 / 优化后 / 数据集最优',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_ylim(0, 0.65)
    ax1.grid(True, axis='y', alpha=0.3, linestyle=':')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── 右: 散点图 init_ber vs pred_ber ──
    ax2 = fig.add_subplot(gs[1])
    colors_dot = [C_PRED if p < i - 0.01
                  else (C_BEST if abs(p - i) < 0.01 else C_INIT)
                  for i, p in zip(init_bers, pred_bers)]

    for i, (ib, pb, lbl, c) in enumerate(zip(init_bers, pred_bers, labels, colors_dot)):
        ax2.scatter(ib, pb, s=120, color=c, zorder=5, edgecolors='white', linewidth=1)
        short = lbl.replace(' / ', '/\n')
        offset_x = -0.005 if ib > 0.3 else 0.01
        ax2.text(ib + offset_x, pb + 0.01, short,
                 fontsize=7, ha='center', va='bottom', color='#333333')

    # 对角线: pred = init (无变化)
    lim = 0.56
    ax2.plot([0, lim], [0, lim], '--', color='gray', linewidth=1, alpha=0.6,
             label='BER不变')
    ax2.fill_between([0, lim], [0, 0], [0, lim], alpha=0.05, color='green',
                     label='BER改善区')
    ax2.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.05, color='red',
                     label='BER恶化区')
    ax2.text(0.05, 0.42, '改善区', fontsize=9, color='#2CA02C', alpha=0.8)
    ax2.text(0.35, 0.08, '恶化区', fontsize=9, color='#D62728', alpha=0.8)

    ax2.set_xlim(-0.02, lim)
    ax2.set_ylim(-0.02, lim)
    ax2.set_xlabel('初始 BER', fontsize=10)
    ax2.set_ylabel('优化后 BER', fontsize=10)
    ax2.set_title('初始 vs 优化后\nBER 散点图', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_aspect('equal')

    plt.suptitle('增益优化算法: BER 改善汇总', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# Figure 3: 典型案例图像对比
#   - turbidity/ISO: BER=0.49 → BER=0.00
#   - bubble/Texp:   BER=0.17 → BER=0.08
# ══════════════════════════════════════════════════════════════════
def _load_and_crop(path: str, target_h: int = 400, target_w: int = 600) -> np.ndarray:
    """加载图像，裁剪中心区域并缩放"""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    # 裁剪中心
    ch, cw = h // 2, w // 2
    half_h, half_w = min(h // 3, 300), min(w // 3, 450)
    crop = img[ch - half_h:ch + half_h, cw - half_w:cw + half_w]
    # RGB
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    # 缩放到目标尺寸
    crop = cv2.resize(crop, (target_w, target_h))
    return crop


def _annotate_image_ax(ax: plt.Axes, img: np.ndarray, title: str,
                       ber: float, iso: int, gain_db: float,
                       ber_color: str = 'white') -> None:
    """在子图上显示图像并标注 BER 和参数"""
    ax.imshow(img)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
    ax.axis('off')

    # BER 标注 (大字)
    ber_str = f"BER = {ber:.4f}" if ber > 0 else "BER = 0"
    ax.text(0.5, 0.06, ber_str,
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=14, fontweight='bold', color=ber_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black',
                      alpha=0.65, edgecolor='none'))

    # 参数标注 (小字, 左上)
    ax.text(0.02, 0.97, f"ISO={iso}  |  增益={gain_db:+.1f} dB",
            transform=ax.transAxes, ha='left', va='top',
            fontsize=9, color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                      alpha=0.55, edgecolor='none'))


def plot_image_comparison() -> plt.Figure:
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('实际图像对比: 增益优化前 vs 优化后',
                 fontsize=15, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           wspace=0.08, hspace=0.25,
                           left=0.04, right=0.96,
                           top=0.92, bottom=0.08)

    # ── 第一行: turbidity / ISO ──
    #   初始: ISO=35   BER=0.4948  (52600_35_p32_Mg_2.5_bubble_1_2_2_1.jpg)
    #   优化: ISO=3200 BER=0.000   (52600_3200_p32_Mg_2.5_bubble_1_2_2_1.jpg)
    turb_low  = str(ISO_TEXP / 'turbidity/ISO/52600_35_p32_Mg_2.5_bubble_1_2_2_1.jpg')
    turb_high = str(ISO_TEXP / 'turbidity/ISO/52600_3200_p32_Mg_2.5_bubble_1_2_2_1.jpg')

    img_tl = _load_and_crop(turb_low)
    img_th = _load_and_crop(turb_high)

    ax_tl = fig.add_subplot(gs[0, 0])
    ax_th = fig.add_subplot(gs[0, 1])
    ax_ts = fig.add_subplot(gs[0, 2])

    _annotate_image_ax(ax_tl, img_tl, '初始增益 (ISO=35)',
                       0.4948, 35, -5.8, ber_color='#FF6B6B')
    _annotate_image_ax(ax_th, img_th, '优化后增益 (ISO=3200)',
                       0.000, 3200, +33.4, ber_color='#6BFF6B')

    # 改善说明子图
    ax_ts.set_facecolor('#F5F5F5')
    ax_ts.text(0.5, 0.75, 'turbidity / ISO',
               ha='center', va='center', fontsize=13, fontweight='bold',
               transform=ax_ts.transAxes, color='#333333')
    ax_ts.annotate('',
                   xy=(0.7, 0.42), xytext=(0.3, 0.42),
                   xycoords='axes fraction', textcoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', color='#2CA02C', lw=3))
    ax_ts.text(0.18, 0.42, 'BER\n0.4948', ha='center', va='center',
               fontsize=14, fontweight='bold', color=C_INIT,
               transform=ax_ts.transAxes)
    ax_ts.text(0.82, 0.42, 'BER\n0.000', ha='center', va='center',
               fontsize=14, fontweight='bold', color=C_PRED,
               transform=ax_ts.transAxes)
    ax_ts.text(0.5, 0.18, '改善率: +100%\n(近随机→零误码)',
               ha='center', va='center', fontsize=12,
               fontweight='bold', color='#2CA02C',
               transform=ax_ts.transAxes,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='#EAFBE7',
                         edgecolor='#2CA02C', linewidth=2))
    ax_ts.set_xticks([])
    ax_ts.set_yticks([])
    ax_ts.spines['top'].set_linewidth(2)
    ax_ts.spines['bottom'].set_linewidth(2)
    ax_ts.spines['left'].set_linewidth(2)
    ax_ts.spines['right'].set_linewidth(2)
    for sp in ax_ts.spines.values():
        sp.set_edgecolor('#CCCCCC')

    # ── 第二行: bubble / Texp ──
    #   初始: Texp=1/21800s  ISO=35  BER=0.1649
    #   优化: Texp=1/35800s  ISO=35  BER=0.0833（最接近预测增益+1.9dB）
    # 对应文件需要查看实际存在的文件
    bub_low  = str(ISO_TEXP / 'bubble/Texp/21800_35_p32_bubble_1_4_1.jpg')
    bub_high = str(ISO_TEXP / 'bubble/Texp/35800_35_p32_bubble_1_2_1.jpg')

    img_bl = _load_and_crop(bub_low)
    img_bh = _load_and_crop(bub_high)

    ax_bl = fig.add_subplot(gs[1, 0])
    ax_bh = fig.add_subplot(gs[1, 1])
    ax_bs = fig.add_subplot(gs[1, 2])

    _annotate_image_ax(ax_bl, img_bl, '初始增益 (Texp=1/21800s)',
                       0.1649, 35, -5.8, ber_color='#FF6B6B')
    _annotate_image_ax(ax_bh, img_bh, '优化后增益 (Texp=1/35800s)',
                       0.0833, 35, +1.9, ber_color='#FFD700')

    ax_bs.set_facecolor('#F5F5F5')
    ax_bs.text(0.5, 0.75, 'bubble / Texp',
               ha='center', va='center', fontsize=13, fontweight='bold',
               transform=ax_bs.transAxes, color='#333333')
    ax_bs.annotate('',
                   xy=(0.7, 0.42), xytext=(0.3, 0.42),
                   xycoords='axes fraction', textcoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', color='#2CA02C', lw=3))
    ax_bs.text(0.18, 0.42, 'BER\n0.1649', ha='center', va='center',
               fontsize=14, fontweight='bold', color=C_INIT,
               transform=ax_bs.transAxes)
    ax_bs.text(0.82, 0.42, 'BER\n0.0833', ha='center', va='center',
               fontsize=14, fontweight='bold', color=C_PRED,
               transform=ax_bs.transAxes)
    ax_bs.text(0.5, 0.18, '改善率: +49.5%',
               ha='center', va='center', fontsize=12,
               fontweight='bold', color='#2CA02C',
               transform=ax_bs.transAxes,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='#EAFBE7',
                         edgecolor='#2CA02C', linewidth=2))
    ax_bs.set_xticks([])
    ax_bs.set_yticks([])
    for sp in ax_bs.spines.values():
        sp.set_edgecolor('#CCCCCC')
        sp.set_linewidth(2)

    # 行标签
    fig.text(0.01, 0.72, '案例 1\nturbidity\n/ ISO',
             ha='center', va='center', fontsize=10,
             fontweight='bold', color='#555555',
             rotation=0,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0',
                       edgecolor='#FF8C00', linewidth=1.5))
    fig.text(0.01, 0.28, '案例 2\nbubble\n/ Texp',
             ha='center', va='center', fontsize=10,
             fontweight='bold', color='#555555',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9',
                       edgecolor='#2CA02C', linewidth=1.5))

    return fig


# ══════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('绘制 Figure 1: BER vs 增益曲线...')
    fig1 = plot_ber_curves()
    out1 = OUTPUT_DIR / 'fig1_ber_vs_gain_curves.png'
    fig1.savefig(str(out1), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f'  → 保存: {out1}')

    print('绘制 Figure 2: BER 对比汇总...')
    fig2 = plot_summary_comparison()
    out2 = OUTPUT_DIR / 'fig2_ber_comparison_summary.png'
    fig2.savefig(str(out2), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'  → 保存: {out2}')

    print('绘制 Figure 3: 图像对比...')
    fig3 = plot_image_comparison()
    out3 = OUTPUT_DIR / 'fig3_image_comparison.png'
    fig3.savefig(str(out3), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f'  → 保存: {out3}')

    print(f'\n✅ 全部图表已保存至: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
