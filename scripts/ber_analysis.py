#!/usr/bin/env python3
"""
BER vs 增益分析 (Bit Error Rate Analysis)

流程:
  1. 对 ISO-Texp 数据集中每张图片运行解调，计算 BER
  2. 按等效增益分组，绘制 BER vs 增益曲线
  3. 模拟单次增益优化算法，验证能否将工作点移到更低 BER

输出: results/ber_analysis/
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = [
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "PingFang SC",
    "SimHei",
    "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# 添加项目 src 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from occ_gain_opt.experiment_loader import ExperimentLoader
from occ_gain_opt.demodulation import OOKDemodulator

# ─────────────────────────── 路径常量 ───────────────────────────
BASE_DIR = Path(__file__).parent.parent
LABEL_CSV = BASE_DIR / "results/base_data/Mseq_32_original.csv"
ISO_TEXP = BASE_DIR / "ISO-Texp"
OUTPUT_DIR = BASE_DIR / "results/ber_analysis"

DATA_BITS = 32  # p32: 每包数据位数
TARGET_GRAY = 242.25  # 增益优化目标灰度 (255 × 0.95)


# ─────────────────────────── 解调工具函数 ───────────────────────────


def load_ground_truth(csv_path: Path, n_bits: int = DATA_BITS) -> np.ndarray:
    """从 Mseq CSV 加载前 n_bits 位真值比特"""
    df = pd.read_csv(str(csv_path), skiprows=5, header=0)
    bits = df.iloc[:, 1].astype(int).to_numpy()
    return bits[:n_bits]


def compute_ber(decoded: np.ndarray, truth: np.ndarray) -> float:
    """计算误码率；若解码长度不足则只比较有效部分"""
    n = min(len(decoded), len(truth))
    if n == 0:
        return 0.5
    return float(np.sum(decoded[:n] != truth[:n])) / n


def demodulate_with_ook(img_bgr: np.ndarray, truth: np.ndarray) -> dict:
    """使用 OOKDemodulator 进行解调"""
    result = OOKDemodulator().demodulate(img_bgr)
    if not result.packets or len(result.packets[0]) == 0:
        return {"bits": np.array([], dtype=int), "status": "sync_fail"}
    pkt = result.packets[0].astype(int)
    n = min(len(truth), len(pkt))
    if n == 0:
        return {"bits": np.array([], dtype=int), "status": "short"}
    return {"bits": pkt[:n], "status": "ok"}


def compute_roi_mean(img_bgr: np.ndarray) -> float:
    """
    计算感兴趣区域 (ROI) 平均灰度。
    优先使用 OOKDemodulator 的 sync-based ROI 掩码；
    若失败则用全图中心 50% 区域均值。
    """
    try:
        from occ_gain_opt.demodulation import OOKDemodulator

        result = OOKDemodulator().demodulate(img_bgr)
        if result.roi_mask.sum() > 0:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            return float(np.mean(gray[result.roi_mask > 0]))
    except Exception:
        pass
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    return float(np.mean(gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]))


def gain_predict_singleshot(
    init_gain_db: float, init_roi: float, target: float = TARGET_GRAY
) -> float:
    """单次增益优化公式: G_opt(dB) = G_curr + 20·log10(Y_target / Y_curr)"""
    if init_roi <= 0:
        return init_gain_db
    return init_gain_db + 20.0 * float(np.log10(target / init_roi))


# ─────────────────────────── 核心分析 ───────────────────────────


def analyze_group(
    exp_type: str, img_type: str, loader: ExperimentLoader, truth: np.ndarray
) -> Optional[dict]:
    """
    分析一个实验子组 (e.g. bubble/ISO) 的 BER vs 增益。

    每个 (exposure_time, ISO) 组合对应一个增益档，
    每档有多张重复图（index 1/10/20/30/40/50），取平均 BER。
    """
    images = loader.load_experiment(exp_type, img_type)
    if not images:
        return None

    # 按 (exposure_time, iso) 分组
    groups: Dict[Tuple, list] = {}
    for img in images:
        key = (round(img.exposure_time, 8), int(img.iso))
        groups.setdefault(key, []).append(img)

    print(f"\n[{exp_type}/{img_type}]  {len(groups)} 个增益档, {len(images)} 张图片")

    gain_levels: List[float] = []
    ber_means: List[float] = []
    ber_stds: List[float] = []
    roi_means: List[float] = []
    demod_rates: List[float] = []

    for key in sorted(groups.keys()):
        group_imgs = groups[key]
        gains_g, bers_g, rois_g = [], [], []
        ok_count = 0

        for exp_img in group_imgs:
            img = cv2.imread(exp_img.filepath)
            if img is None:
                continue

            gdb = exp_img.calculate_equivalent_gain()
            gains_g.append(gdb)

            # 解调 → BER
            res = demodulate_with_ook(img, truth)
            if res["status"] == "ok":
                ber = compute_ber(res["bits"], truth)
                ok_count += 1
            else:
                ber = 0.5  # 解调失败 → 随机猜测水平
            bers_g.append(ber)

            # ROI 均值（供增益优化公式使用）
            rois_g.append(compute_roi_mean(img))

        if not gains_g:
            continue

        avg_gain = float(np.mean(gains_g))
        avg_ber = float(np.mean(bers_g))
        std_ber = float(np.std(bers_g))
        avg_roi = float(np.mean(rois_g))
        rate = ok_count / len(group_imgs)

        gain_levels.append(avg_gain)
        ber_means.append(avg_ber)
        ber_stds.append(std_ber)
        roi_means.append(avg_roi)
        demod_rates.append(rate)

        tag = "✅" if rate > 0 else "❌"
        print(
            f"  {tag} gain={avg_gain:+6.1f} dB | BER={avg_ber:.4f} "
            f"± {std_ber:.4f} | ROI={avg_roi:5.1f} "
            f"| 解调成功率={rate:.0%}"
        )

    if len(gain_levels) < 2:
        print("  ⚠ 有效增益档不足，跳过此组")
        return None

    gain_arr = np.array(gain_levels)
    ber_arr = np.array(ber_means)
    roi_arr = np.array(roi_means)

    # ── 增益优化模拟 (单次公式) ──
    idx_init = int(np.argmin(gain_arr))
    init_gain = float(gain_arr[idx_init])
    init_ber = float(ber_arr[idx_init])
    init_roi = float(roi_arr[idx_init])

    pred_gain = gain_predict_singleshot(init_gain, init_roi)
    idx_pred = int(np.argmin(np.abs(gain_arr - pred_gain)))
    pred_gain_act = float(gain_arr[idx_pred])
    pred_ber = float(ber_arr[idx_pred])

    idx_best = int(np.argmin(ber_arr))
    best_gain = float(gain_arr[idx_best])
    best_ber = float(ber_arr[idx_best])

    # 改善量: 使用绝对 BER 下降值（百分点）和相对改善（仅在 init_ber>0 时有意义）
    ber_abs_change = pred_ber - init_ber  # 负值 = 改善，正值 = 变差
    if init_ber > 0.001:
        improvement = (init_ber - pred_ber) / init_ber * 100.0
    elif pred_ber <= 0.001:
        improvement = 0.0  # 都是 0，无变化
    else:
        improvement = float("nan")  # 初始已是 0，但优化后变差，用 NaN 标记

    print(f"\n  ┌── 增益优化模拟结果 ──")
    print(
        f"  │ 初始  : gain={init_gain:+.1f} dB, BER={init_ber:.4f}, "
        f"ROI均值={init_roi:.1f}"
    )
    print(
        f"  │ 预测  : G_opt={pred_gain:+.1f} dB → 实际选 {pred_gain_act:+.1f} dB, "
        f"BER={pred_ber:.4f}"
    )
    print(f"  │ 最优  : gain={best_gain:+.1f} dB, BER={best_ber:.4f}")
    if not (improvement != improvement):  # not NaN
        print(f"  └── BER 改善: {improvement:+.1f}%")
    else:
        print(f"  └── 初始BER已为0，优化后BER={pred_ber:.4f} (轻微过冲)")

    return {
        "exp_type": exp_type,
        "img_type": img_type,
        "gain_levels": gain_arr,
        "ber_means": ber_arr,
        "ber_stds": np.array(ber_stds),
        "roi_means": roi_arr,
        "demod_rates": np.array(demod_rates),
        "init_gain": init_gain,
        "init_ber": init_ber,
        "init_roi": init_roi,
        "pred_gain": pred_gain,
        "pred_gain_act": pred_gain_act,
        "pred_ber": pred_ber,
        "best_gain": best_gain,
        "best_ber": best_ber,
        "improvement": improvement,
    }


# ─────────────────────────── 可视化 ───────────────────────────


def plot_group(ax: plt.Axes, data: dict) -> None:
    """在子图上绘制 BER vs 增益曲线，标注初始/预测/最优三个工作点"""
    g = data["gain_levels"]
    b = data["ber_means"]
    e = data["ber_stds"]

    ax.errorbar(
        g,
        b,
        yerr=e,
        fmt="o-",
        color="steelblue",
        linewidth=1.8,
        markersize=6,
        capsize=4,
        label="BER ± std",
        zorder=3,
    )

    # 初始工作点
    ax.axvline(
        data["init_gain"],
        color="#FF8C00",
        linestyle="--",
        linewidth=1.5,
        label=f"初始 {data['init_gain']:+.0f} dB\n(BER={data['init_ber']:.3f})",
    )
    ax.plot(
        data["init_gain"],
        data["init_ber"],
        "s",
        color="#FF8C00",
        markersize=9,
        zorder=5,
    )

    # 优化预测工作点
    ax.axvline(
        data["pred_gain_act"],
        color="#2CA02C",
        linestyle="--",
        linewidth=1.5,
        label=f"优化预测 {data['pred_gain_act']:+.0f} dB\n(BER={data['pred_ber']:.3f})",
    )
    ax.plot(
        data["pred_gain_act"],
        data["pred_ber"],
        "^",
        color="#2CA02C",
        markersize=9,
        zorder=5,
    )

    # 数据集最优
    ax.axvline(
        data["best_gain"],
        color="#D62728",
        linestyle=":",
        linewidth=1.5,
        label=f"最优 {data['best_gain']:+.0f} dB\n(BER={data['best_ber']:.3f})",
    )
    ax.plot(
        data["best_gain"],
        data["best_ber"],
        "*",
        color="#D62728",
        markersize=11,
        zorder=5,
    )

    ax.set_title(f"{data['exp_type']} / {data['img_type']}", fontsize=10)
    ax.set_xlabel("等效增益 (dB)", fontsize=9)
    ax.set_ylabel("BER", fontsize=9)
    ax.set_ylim(-0.05, 0.6)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # 解调成功率副坐标
    ax2 = ax.twinx()
    ax2.bar(
        data["gain_levels"],
        data["demod_rates"] * 100,
        width=(data["gain_levels"].max() - data["gain_levels"].min())
        / (len(data["gain_levels"]) + 1),
        alpha=0.12,
        color="gray",
        label="解调成功率",
    )
    ax2.set_ylabel("解调成功率 (%)", fontsize=8, color="gray")
    ax2.set_ylim(0, 120)
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)


def plot_summary_bar(all_results: List[dict], output_dir: Path) -> None:
    """绘制各实验组 BER 改善汇总柱状图"""
    labels = [f"{r['exp_type']}\n{r['img_type']}" for r in all_results]
    init_bers = [r["init_ber"] for r in all_results]
    pred_bers = [r["pred_ber"] for r in all_results]
    best_bers = [r["best_ber"] for r in all_results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, init_bers, width, label="初始 BER", color="#FF8C00", alpha=0.8)
    ax.bar(x, pred_bers, width, label="优化后 BER", color="#2CA02C", alpha=0.8)
    ax.bar(x + width, best_bers, width, label="最优 BER", color="#D62728", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("BER", fontsize=10)
    ax.set_title("增益优化前后 BER 对比 (各实验组)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(init_bers), 0.6) * 1.15)

    plt.tight_layout()
    out = output_dir / "ber_comparison_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"🖼  柱状图已保存: {out}")
    plt.close(fig)


def save_csv(all_results: List[dict], output_dir: Path) -> None:
    """保存详细汇总 CSV"""
    rows = []
    for r in all_results:
        rows.append(
            {
                "实验": f"{r['exp_type']}/{r['img_type']}",
                "初始增益(dB)": f"{r['init_gain']:.1f}",
                "初始ROI均值": f"{r['init_roi']:.1f}",
                "初始BER": f"{r['init_ber']:.4f}",
                "预测最优增益(dB)": f"{r['pred_gain']:.1f}",
                "实际选用增益(dB)": f"{r['pred_gain_act']:.1f}",
                "优化后BER": f"{r['pred_ber']:.4f}",
                "数据集最优增益(dB)": f"{r['best_gain']:.1f}",
                "数据集最优BER": f"{r['best_ber']:.4f}",
                "BER改善(%)": "N/A(过冲)"
                if r["improvement"] != r["improvement"]
                else f"{r['improvement']:.1f}",
            }
        )
    df = pd.DataFrame(rows)
    out = output_dir / "ber_analysis_summary.csv"
    df.to_csv(str(out), index=False, encoding="utf-8-sig")
    print(f"📄 汇总 CSV 已保存: {out}")


# ─────────────────────────── 主程序 ───────────────────────────


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载真值
    truth = load_ground_truth(LABEL_CSV, DATA_BITS)
    print(f"✅ 已加载真值: {len(truth)} bits")
    print(f"   前8位: {truth[:8].tolist()}")

    loader = ExperimentLoader(str(ISO_TEXP))

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
        result = analyze_group(exp_type, img_type, loader, truth)
        if result:
            all_results.append(result)

    if not all_results:
        print("❌ 无有效结果，请检查 ISO-Texp 目录和真值文件")
        return

    # ── 绘制 BER vs 增益曲线 ──
    ncols = 3
    nrows = (len(all_results) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes_flat = np.array(axes).flatten()

    for i, r in enumerate(all_results):
        plot_group(axes_flat[i], r)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("BER vs 增益 — 增益优化算法效果分析", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_curves = OUTPUT_DIR / "ber_vs_gain_curves.png"
    fig.savefig(str(out_curves), dpi=150, bbox_inches="tight")
    print(f"\n🖼  BER曲线图已保存: {out_curves}")
    plt.close(fig)

    # ── 汇总柱状图 ──
    plot_summary_bar(all_results, OUTPUT_DIR)

    # ── 打印汇总表 ──
    print("\n" + "═" * 72)
    print("  汇总: 增益优化算法对 BER 的影响")
    print("═" * 72)
    print(
        f"  {'实验':<22} {'初始BER':>8} {'优化后BER':>10} "
        f"{'最优BER':>9} {'BER改善':>10}"
    )
    print("  " + "─" * 64)
    for r in all_results:
        imp = r["improvement"]
        if imp != imp:  # NaN
            arrow, imp_str = "⬆", f"  +{r['pred_ber']:.4f} pts"
        elif imp > 1:
            arrow, imp_str = "⬇", f"{imp:>+8.1f}%"
        elif abs(imp) <= 1:
            arrow, imp_str = "➡", f"{imp:>+8.1f}%"
        else:
            arrow, imp_str = "⬆", f"{imp:>+8.1f}%"
        print(
            f"  {r['exp_type']}/{r['img_type']:<14} "
            f"{r['init_ber']:>8.4f} "
            f"{r['pred_ber']:>10.4f} "
            f"{r['best_ber']:>9.4f} "
            f"  {arrow} {imp_str}"
        )
    print("═" * 72)

    # ── 保存 CSV ──
    save_csv(all_results, OUTPUT_DIR)
    print("\n✅ BER 分析完成！结果保存在:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
