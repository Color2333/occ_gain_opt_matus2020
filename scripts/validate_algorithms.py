"""
算法验证脚本 — 合并 validate_algorithm_on_real_data + validate_iterative_algorithm
===================================================================

在真实数据集（ISO-Texp/）上验证三种算法：
  1. Matus 单次公式
  2. Matus 自适应迭代
  3. Ma 自适应阻尼（单步预测）

用法:
    python scripts/validate_algorithms.py
    python scripts/validate_algorithms.py --dataset-dir ISO-Texp --condition bubble_1_2_2
    python scripts/validate_algorithms.py --algorithm matus_single
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = [
    "Hiragino Sans GB", "Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"
]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

from occ_gain_opt.config import CameraParams, ROIStrategy
from occ_gain_opt.algorithms import get as algo_get, list_algorithms
from occ_gain_opt.data_sources import (
    create_sync_based_roi_mask,
    create_auto_roi_mask,
    compute_roi_stats,
)
from occ_gain_opt.experiment_loader import ExperimentLoader


# ── ROI 工具（带回退逻辑）────────────────────────────────────────────────────

def get_roi_brightness(image_bgr, strategy="sync_based"):
    """返回 (mean_gray, method_used)，优先 sync_based，失败退回 auto"""
    import cv2
    import numpy as _np

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    for strat, label in [(strategy, strategy), ("auto", "auto_brightness")]:
        try:
            if strat == "sync_based":
                mask = create_sync_based_roi_mask(image_bgr)
            else:
                mask = create_auto_roi_mask(gray)
            if _np.sum(mask) > 0:
                stats = compute_roi_stats(gray, mask)
                return float(stats["mean"]), label
        except Exception:
            continue
    return float(_np.mean(gray)), "full_image"


# ── 验证逻辑 ──────────────────────────────────────────────────────────────────

def validate_single_shot(images, algo_name="matus_single", verbose=True):
    """
    单次公式验证：对每张图像执行一步算法，比较预测增益 vs 实际结果。
    返回结果列表，每项包含 (current_iso, predicted_iso, brightness, method)
    """
    algo_cls = algo_get(algo_name)

    results = []
    for img_obj in images:
        if not img_obj.load():
            continue
        current_params = CameraParams(
            iso=img_obj.iso,
            exposure_us=img_obj.exposure_time * 1e6,
        )
        brightness, method = get_roi_brightness(img_obj.image)
        algo = algo_cls()
        next_params = algo.compute_next_params(current_params, brightness)
        results.append({
            "iso": img_obj.iso,
            "exposure_us": img_obj.exposure_time * 1e6,
            "brightness": brightness,
            "roi_method": method,
            "predicted_iso": next_params.iso,
            "predicted_gain_db": next_params.gain_db,
            "condition": img_obj.condition,
        })
        if verbose:
            print(f"  ISO={img_obj.iso:>6.1f} → 亮度={brightness:>6.1f}  "
                  f"→ 推荐ISO={next_params.iso:>7.1f} ({next_params.gain_db:>+.2f}dB)  [{method}]")
    return results


def validate_iterative(images, algo_name="matus_adaptive", n_iter=5, verbose=True):
    """
    迭代验证：对排序后的图像序列模拟多轮迭代，观察收敛行为。
    """
    algo_cls = algo_get(algo_name)
    algo = algo_cls()

    # 按 ISO 排序
    images_sorted = sorted(images, key=lambda x: x.iso)
    history = []

    params = None
    for i, img_obj in enumerate(images_sorted[:n_iter]):
        if not img_obj.load():
            continue
        if params is None:
            params = CameraParams(iso=img_obj.iso, exposure_us=img_obj.exposure_time * 1e6)

        brightness, method = get_roi_brightness(img_obj.image)
        next_params = algo.compute_next_params(params, brightness)
        history.append({
            "iter": i,
            "iso": params.iso,
            "brightness": brightness,
            "next_iso": next_params.iso,
            "converged": algo.is_converged(),
        })
        if verbose:
            print(f"  iter {i}: ISO={params.iso:.1f}, 亮度={brightness:.1f} "
                  f"→ 推荐ISO={next_params.iso:.1f}  {'[收敛]' if algo.is_converged() else ''}")
        params = next_params

    return {"history": history, "algo_history": algo.get_history()}


# ── 可视化 ─────────────────────────────────────────────────────────────────────

def plot_validation(results_by_algo, output_dir="results/algorithm_validation"):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(results_by_algo), figsize=(6 * len(results_by_algo), 5))
    if len(results_by_algo) == 1:
        axes = [axes]

    for ax, (algo_name, results) in zip(axes, results_by_algo.items()):
        if not results:
            continue
        isos = [r["iso"] for r in results]
        brightnesses = [r["brightness"] for r in results]
        pred_isos = [r["predicted_iso"] for r in results]

        ax.scatter(isos, brightnesses, label="当前亮度", alpha=0.7, c="blue")
        ax.scatter(isos, pred_isos, label="预测ISO", alpha=0.7, c="red", marker="^")
        ax.axhline(y=242.25, color="green", linestyle="--", label="目标灰度 (242.25)")
        ax.set_xlabel("当前 ISO")
        ax.set_ylabel("值")
        ax.set_title(f"{algo_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "validation_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存: {out_path}")
    plt.close(fig)


# ── 主函数 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="在真实数据集上验证增益优化算法")
    parser.add_argument("--dataset-dir", default="ISO-Texp", help="数据集根目录")
    parser.add_argument("--condition", default=None, help="条件过滤（如 bubble_1_2_2）")
    parser.add_argument("--image-type", default="ISO", choices=["ISO", "Texp", None],
                        help="图像类型过滤")
    parser.add_argument("--algorithm", default=None,
                        help=f"验证的算法名（默认全部）。可选: {list_algorithms()}")
    parser.add_argument("--n-iter", type=int, default=5, help="迭代验证步数")
    parser.add_argument("--output-dir", default="results/algorithm_validation")
    args = parser.parse_args()

    # ── 加载数据集 ──
    loader = ExperimentLoader(args.dataset_dir)
    images = loader.load_all_images(
        condition=args.condition,
        image_type=args.image_type if args.image_type != "None" else None,
    )
    if not images:
        print(f"数据集 '{args.dataset_dir}' 中未找到图像，跳过真实数据验证。")
        print("请确认 ISO-Texp/ 目录存在并包含实验图像（该目录在 .gitignore 中）。")
        return

    print(f"已加载 {len(images)} 张图像")

    # ── 确定要验证的算法 ──
    algos = [args.algorithm] if args.algorithm else list_algorithms()

    results_by_algo = {}
    for algo_name in algos:
        print(f"\n{'=' * 50}")
        print(f"验证算法: {algo_name}")
        print("─" * 50)

        # 单步验证
        results = validate_single_shot(images, algo_name=algo_name)
        results_by_algo[algo_name] = results

        # 迭代验证
        if algo_name in ("matus_adaptive", "ma_damping"):
            print(f"\n[迭代验证 — {args.n_iter} 步]")
            validate_iterative(images, algo_name=algo_name, n_iter=args.n_iter)

    # ── 生成报告 ──
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        for algo_name, results in results_by_algo.items():
            f.write(f"\n算法: {algo_name}\n")
            f.write("─" * 40 + "\n")
            if results:
                brightnesses = [r["brightness"] for r in results]
                f.write(f"  样本数: {len(results)}\n")
                f.write(f"  亮度均值: {np.mean(brightnesses):.2f}\n")
                f.write(f"  亮度std:  {np.std(brightnesses):.2f}\n")
            else:
                f.write("  无结果\n")
    print(f"\n验证报告已保存: {report_path}")

    plot_validation(results_by_algo, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
