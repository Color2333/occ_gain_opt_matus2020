"""
生成对比示意图与实验图像原理图
"""

from pathlib import Path
import argparse
from typing import List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..config import ROIStrategy, ExperimentConfig
from ..data_acquisition import DataAcquisition
from ..experiment_loader import ExperimentLoader, ExperimentImage
from ..gain_optimization import GainOptimizer, AdaptiveGainOptimizer

matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _ensure_output_dir() -> Path:
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_algorithm_comparison_diagram() -> Path:
    """
    生成基础算法 vs 自适应算法对比示意图
    """
    output_dir = _ensure_output_dir()
    out_path = output_dir / "algorithm_comparison_diagram.png"

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    # 盒子样式
    box = dict(boxstyle="round,pad=0.5", facecolor="#F5F7FF", edgecolor="#4A6CF7")
    box_adapt = dict(boxstyle="round,pad=0.5", facecolor="#F7FFF5", edgecolor="#2D8A34")

    # 标题
    ax.text(0.5, 0.95, "增益控制算法对比示意图", ha="center", va="center",
            fontsize=16, fontweight="bold")

    # 公共输入
    ax.text(0.5, 0.85, "输入: 当前图像 → ROI平均灰度 Y_curr", ha="center", va="center",
            fontsize=12)

    # 左侧: 基础算法
    ax.text(0.2, 0.72, "基础算法", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.2, 0.60,
            "公式:\nG_opt = G_curr × (目标 / Y_curr)\n\n特点:\n- 一步比例更新\n- 快速收敛",
            ha="center", va="center", fontsize=11, bbox=box)

    # 右侧: 自适应算法
    ax.text(0.8, 0.72, "自适应算法", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.8, 0.60,
            "在基础公式上加入:\n- EMA 平滑灰度\n- 自适应学习率\n- 动量项\n- 单步限幅\n\n特点:\n- 更稳定\n- 抗噪更强",
            ha="center", va="center", fontsize=11, bbox=box_adapt)

    # 目标
    ax.text(0.5, 0.40, "目标: ROI灰度接近 255(×安全因子)", ha="center", va="center",
            fontsize=12)

    # 箭头
    ax.annotate("", xy=(0.2, 0.66), xytext=(0.5, 0.80),
                arrowprops=dict(arrowstyle="->", linewidth=1.5))
    ax.annotate("", xy=(0.8, 0.66), xytext=(0.5, 0.80),
                arrowprops=dict(arrowstyle="->", linewidth=1.5))
    ax.annotate("", xy=(0.5, 0.44), xytext=(0.2, 0.54),
                arrowprops=dict(arrowstyle="->", linewidth=1.5))
    ax.annotate("", xy=(0.5, 0.44), xytext=(0.8, 0.54),
                arrowprops=dict(arrowstyle="->", linewidth=1.5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return out_path


def _draw_roi_box(ax, roi_mask):
    y, x = np.where(roi_mask == 1)
    if len(x) > 0:
        ax.plot([x.min(), x.max(), x.max(), x.min(), x.min()],
                [y.min(), y.min(), y.max(), y.max(), y.min()],
                "r-", linewidth=2)


def _center_roi_mask(image: np.ndarray, roi_size: int = 100) -> np.ndarray:
    height, width = image.shape[:2]
    roi_w = min(roi_size, width)
    roi_h = min(roi_size, height)
    x = (width - roi_w) // 2
    y = (height - roi_h) // 2
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y:y + roi_h, x:x + roi_w] = 1
    return mask


def _auto_roi_mask(image: np.ndarray, roi_size: int = 100) -> np.ndarray:
    """
    自动找最亮区域作为ROI (优先取最亮连通域, 否则取最亮窗口)
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray.shape[1] - x, w + 2 * padding)
        h = min(gray.shape[0] - y, h + 2 * padding)
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[y:y + h, x:x + w] = 1
        return mask

    h, w = gray.shape
    roi_w = min(roi_size, w)
    roi_h = min(roi_size, h)
    max_mean = -1
    best_x = 0
    best_y = 0
    step = max(roi_size // 4, 10)
    for y in range(0, h - roi_h + 1, step):
        for x in range(0, w - roi_w + 1, step):
            window = gray[y:y + roi_h, x:x + roi_w]
            mean_val = float(np.mean(window))
            if mean_val > max_mean:
                max_mean = mean_val
                best_x = x
                best_y = y
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[best_y:best_y + roi_h, best_x:best_x + roi_w] = 1
    return mask


def _pick_real_images(images: List[ExperimentImage]) -> Tuple[ExperimentImage, ExperimentImage]:
    images_sorted = sorted(images, key=lambda img: img.calculate_equivalent_gain())
    return images_sorted[0], images_sorted[-1]


def _select_image_by_target(images: List[ExperimentImage],
                            target_gray: float,
                            use_auto_roi: bool = True) -> ExperimentImage:
    best_img = None
    best_error = None
    for img in images:
        img.load()
        if img.gray_image is None:
            continue
        roi_mask = _auto_roi_mask(img.gray_image) if use_auto_roi else _center_roi_mask(img.gray_image)
        roi_mean = float(np.mean(img.gray_image[roi_mask == 1]))
        error = abs(roi_mean - target_gray)
        if best_error is None or error < best_error:
            best_error = error
            best_img = img
    if best_img is None:
        raise RuntimeError("未找到可用的实验图片")
    return best_img


def generate_real_algorithm_effect_figure(data_dir: str = "ISO-Texp",
                                          experiment_type: str = "bubble") -> Path:
    """
    基于真实实验图像生成算法效果图(前后对比)
    """
    output_dir = _ensure_output_dir()
    out_path = output_dir / "algorithm_real_effect.png"

    loader = ExperimentLoader(base_dir=data_dir)
    iso_images = loader.load_experiment(experiment_type, image_type="ISO")
    texp_images = loader.load_experiment(experiment_type, image_type="Texp")
    images = iso_images + texp_images

    if not images:
        raise FileNotFoundError(
            f"未找到 ISO-Texp 数据: {data_dir}/{experiment_type}/(ISO|Texp)/*.jpg"
        )

    # 低增益图像作为初始
    images_sorted = sorted(images, key=lambda img: img.calculate_equivalent_gain())
    before_img = images_sorted[0]
    before_img.load()
    if before_img.gray_image is None:
        raise RuntimeError(f"图片加载失败: {before_img.filepath}")

    # 目标灰度: 数据可达最大ROI均值的95%
    roi_means = []
    for img in images_sorted:
        img.load()
        if img.gray_image is None:
            continue
        roi_mask = _auto_roi_mask(img.gray_image)
        roi_means.append(float(np.mean(img.gray_image[roi_mask == 1])))
    max_mean = max(roi_means) if roi_means else 255.0
    target_gray = 0.95 * max_mean
    after_img = _select_image_by_target(images_sorted, target_gray, use_auto_roi=True)

    # 绘图
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, wspace=0.2, hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(before_img.gray_image, cmap="gray", vmin=0, vmax=255)
    roi_before = _auto_roi_mask(before_img.gray_image)
    _draw_roi_box(ax1, roi_before)
    mean_before = float(np.mean(before_img.gray_image[roi_before == 1]))
    before_gain = before_img.calculate_equivalent_gain()
    ax1.set_title(
        f"Before (低增益)\nMean={mean_before:.1f}",
        fontsize=12, fontweight="bold"
    )
    ax1.text(
        0.02, 0.98,
        f"算法前:\nISO={before_img.iso:.0f}\nT=1/{1/before_img.exposure_time:.0f}\n等效增益={before_gain:.1f} dB",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#FFF8E1", edgecolor="#C9A227", alpha=0.9),
    )
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(after_img.gray_image, cmap="gray", vmin=0, vmax=255)
    roi_after = _auto_roi_mask(after_img.gray_image)
    _draw_roi_box(ax2, roi_after)
    mean_after = float(np.mean(after_img.gray_image[roi_after == 1]))
    after_gain = after_img.calculate_equivalent_gain()
    ax2.set_title(
        f"After (算法选择)\nMean={mean_after:.1f} 目标≈{target_gray:.1f}",
        fontsize=12, fontweight="bold"
    )
    ax2.text(
        0.02, 0.98,
        f"算法后:\nISO={after_img.iso:.0f}\nT=1/{1/after_img.exposure_time:.0f}\n等效增益={after_gain:.1f} dB",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#E8F5E9", edgecolor="#2D8A34", alpha=0.9),
    )
    ax2.axis("off")

    # 原理说明 (公式)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    ax3.text(0.5, 0.80, "算法原理 (比例控制)", ha="center",
             fontsize=12, fontweight="bold")
    ax3.text(
        0.5, 0.50,
        "G_opt = G_curr × (Y_target / Y_curr)\n"
        "Y_target ≈ 255 × 安全因子",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="#FFF8E1", edgecolor="#C9A227"),
    )
    ax3.text(0.5, 0.18, "通过调 ISO / Texp 来实现等效增益",
             ha="center", fontsize=11)

    # 框图 (流程)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ax4.text(0.5, 0.90, "算法框图", ha="center",
             fontsize=12, fontweight="bold")
    box_style = dict(boxstyle="round", facecolor="#F5F7FF", edgecolor="#4A6CF7")
    ax4.text(0.5, 0.72, "输入图像\n(ROI 灰度)", ha="center", fontsize=10, bbox=box_style)
    ax4.text(0.5, 0.50, "计算最优增益\nG_opt", ha="center", fontsize=10, bbox=box_style)
    ax4.text(0.5, 0.28, "调整 ISO/Texp\n获得新图像", ha="center", fontsize=10, bbox=box_style)
    ax4.annotate("", xy=(0.5, 0.62), xytext=(0.5, 0.68),
                 arrowprops=dict(arrowstyle="->", linewidth=1.5))
    ax4.annotate("", xy=(0.5, 0.40), xytext=(0.5, 0.46),
                 arrowprops=dict(arrowstyle="->", linewidth=1.5))
    ax4.annotate("", xy=(0.5, 0.80), xytext=(0.5, 0.86),
                 arrowprops=dict(arrowstyle="->", linewidth=1.5))

    fig.suptitle("算法真实效果对比 + 原理图 + 框图 (ISO-Texp)", fontsize=14, fontweight="bold")
    fig.text(
        0.5, 0.02,
        "算法改变的是: ISO 与曝光时间 T（等效增益），使 ROI 灰度接近目标值",
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="#F5F7FF", edgecolor="#4A6CF7", alpha=0.9),
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return out_path

def generate_experiment_principle_diagram_real(data_dir: str = "ISO-Texp",
                                               experiment_type: str = "bubble") -> Path:
    """
    使用 ISO-Texp 数据生成实验图片原理图
    """
    output_dir = _ensure_output_dir()
    out_path = output_dir / "experiment_principle_diagram_real.png"

    loader = ExperimentLoader(base_dir=data_dir)
    iso_images = loader.load_experiment(experiment_type, image_type="ISO")
    texp_images = loader.load_experiment(experiment_type, image_type="Texp")

    if not iso_images or not texp_images:
        raise FileNotFoundError(
            f"未找到 ISO-Texp 数据: {data_dir}/{experiment_type}/(ISO|Texp)/*.jpg"
        )

    iso_low, iso_high = _pick_real_images(iso_images)
    texp_mid = sorted(texp_images, key=lambda img: img.calculate_equivalent_gain())[len(texp_images) // 2]

    for img in [iso_low, iso_high, texp_mid]:
        img.load()
        if img.gray_image is None:
            raise RuntimeError(f"图片加载失败: {img.filepath}")

    images = [iso_low, iso_high, texp_mid]
    titles = [
        f"前: 低增益(偏暗)\nISO 低增益\nT={1/iso_low.exposure_time:.0f}, ISO={iso_low.iso:.0f}",
        f"后: 提升ISO(更亮)\nISO 高增益\nT={1/iso_high.exposure_time:.0f}, ISO={iso_high.iso:.0f}",
        f"后: 增加曝光(更亮)\nTexp 中等增益\nT={1/texp_mid.exposure_time:.0f}, ISO={texp_mid.iso:.0f}",
    ]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.2)

    roi_masks = []
    for idx, (exp_img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(exp_img.gray_image, cmap="gray", vmin=0, vmax=255)
        roi_mask = _center_roi_mask(exp_img.gray_image)
        _draw_roi_box(ax, roi_mask)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
        roi_masks.append(roi_mask)

    # 添加前后对比箭头
    fig.text(0.33, 0.62, "→", fontsize=28, fontweight="bold", color="#4A6CF7")
    fig.text(0.66, 0.62, "→", fontsize=28, fontweight="bold", color="#4A6CF7")

    ax_hist = fig.add_subplot(gs[1, :])
    for exp_img, roi_mask, label, color in zip(
        images,
        roi_masks,
        ["ISO 低增益", "ISO 高增益", "Texp 中等增益"],
        ["#888888", "#4A6CF7", "#2D8A34"],
    ):
        gray_vals = exp_img.gray_image[roi_mask == 1]
        ax_hist.hist(gray_vals, bins=40, alpha=0.4, label=label, color=color)

    ax_hist.set_title("ROI 灰度分布对比（真实实验图像）", fontsize=12, fontweight="bold")
    ax_hist.set_xlabel("灰度值")
    ax_hist.set_ylabel("像素数")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    # 算法解释说明
    explanation = (
        "算法原理说明:\n"
        "1) 先测 ROI 平均灰度 Y_curr\n"
        "2) 目标灰度接近 255(饱和值)\n"
        "3) 通过提高 ISO 或曝光时间(=等效增益)\n"
        "   将 ROI 灰度分布整体右移\n"
        "4) 直到靠近目标且不过曝"
    )
    fig.text(0.02, 0.03, explanation, fontsize=11,
             bbox=dict(boxstyle="round", facecolor="#FFF8E1", edgecolor="#C9A227"))

    fig.suptitle("ISO-Texp 实验图片对比与原理说明（前后对比 + 算法解释）",
                 fontsize=15, fontweight="bold")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return out_path


def generate_experiment_principle_diagram() -> Path:
    """
    生成基于实验图片对比的原理图
    """
    output_dir = _ensure_output_dir()
    out_path = output_dir / "experiment_principle_diagram.png"

    data_acq = DataAcquisition()
    optimizer_basic = GainOptimizer(data_acq)
    optimizer_adapt = AdaptiveGainOptimizer(data_acq)

    led_duty_cycle = 50
    background_light = 50
    led_intensity = (led_duty_cycle / 100.0) * 255

    # 低增益图像
    low_gain = 0.0
    image_low = data_acq.capture_image(led_intensity, low_gain, background_light,
                                       noise_std=ExperimentConfig.NOISE_STD)
    roi_mask_low = data_acq.select_roi(strategy=ROIStrategy.CENTER)

    # 基础算法优化结果
    result_basic = optimizer_basic.optimize_gain(
        led_duty_cycle=led_duty_cycle,
        initial_gain=0.0,
        background_light=background_light,
        noise_std=ExperimentConfig.NOISE_STD,
        roi_strategy=ROIStrategy.CENTER
    )
    image_basic = result_basic["final_image"]
    roi_mask_basic = result_basic["roi_mask"]
    if roi_mask_basic is None:
        roi_mask_basic = data_acq.select_roi(strategy=ROIStrategy.CENTER)

    # 自适应算法优化结果
    result_adapt = optimizer_adapt.optimize_gain(
        led_duty_cycle=led_duty_cycle,
        initial_gain=0.0,
        background_light=background_light,
        noise_std=ExperimentConfig.NOISE_STD,
        roi_strategy=ROIStrategy.CENTER
    )
    image_adapt = result_adapt["final_image"]
    roi_mask_adapt = result_adapt["roi_mask"]
    if roi_mask_adapt is None:
        roi_mask_adapt = data_acq.select_roi(strategy=ROIStrategy.CENTER)

    # 绘图布局
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.2)

    # 图像对比
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_low, cmap="gray", vmin=0, vmax=255)
    _draw_roi_box(ax1, roi_mask_low)
    ax1.set_title("低增益(0 dB) - 偏暗", fontsize=12, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(image_basic, cmap="gray", vmin=0, vmax=255)
    _draw_roi_box(ax2, roi_mask_basic)
    ax2.set_title("基础算法优化后", fontsize=12, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(image_adapt, cmap="gray", vmin=0, vmax=255)
    _draw_roi_box(ax3, roi_mask_adapt)
    ax3.set_title("自适应算法优化后", fontsize=12, fontweight="bold")
    ax3.axis("off")

    # 灰度直方图对比
    ax4 = fig.add_subplot(gs[1, :])
    gray_low = image_low[roi_mask_low == 1]
    gray_basic = image_basic[roi_mask_basic == 1]
    gray_adapt = image_adapt[roi_mask_adapt == 1]

    ax4.hist(gray_low, bins=40, alpha=0.4, label="低增益", color="#888888")
    ax4.hist(gray_basic, bins=40, alpha=0.4, label="基础算法", color="#4A6CF7")
    ax4.hist(gray_adapt, bins=40, alpha=0.4, label="自适应算法", color="#2D8A34")
    ax4.axvline(255, color="red", linestyle="--", linewidth=2, label="目标灰度(255)")
    ax4.set_title("ROI灰度分布对比 (原理解释)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("灰度值")
    ax4.set_ylabel("像素数")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle("实验图片对比与原理说明", fontsize=15, fontweight="bold")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return out_path


def generate_adaptive_iteration_strip(data_dir: str = "ISO-Texp",
                                      experiment_type: str = "bubble",
                                      condition_pattern: str = "bubble_1_2",
                                      num_steps: int = 6,
                                      output_name: str = None,
                                      output_dir: Path = None) -> Path:
    """
    基于真实实验图像生成“迭代过程”可视化
    """
    output_dir = output_dir or _ensure_output_dir()
    out_path = output_dir / (output_name or "algorithm_adaptive_iteration.png")

    loader = ExperimentLoader(base_dir=data_dir)
    images = loader.get_images_by_condition(experiment_type, condition_pattern, image_type="ISO")
    images += loader.get_images_by_condition(experiment_type, condition_pattern, image_type="Texp")

    if len(images) < num_steps:
        raise FileNotFoundError(
            f"条件 {condition_pattern} 图片不足: {len(images)} 张"
        )

    # 按等效增益排序，取均匀采样作为迭代步骤
    images_sorted = sorted(images, key=lambda img: img.calculate_equivalent_gain())
    indices = np.linspace(0, len(images_sorted) - 1, num_steps, dtype=int)
    step_images = [images_sorted[i] for i in indices]

    for img in step_images:
        img.load()
        if img.gray_image is None:
            raise RuntimeError(f"图片加载失败: {img.filepath}")

    cols = 3
    rows = int(np.ceil(num_steps / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.0))
    if rows == 1:
        axes = np.array([axes])

    for ax in axes.flatten():
        ax.axis("off")

    for idx, (ax, exp_img) in enumerate(zip(axes.flatten(), step_images), start=1):
        ax.imshow(exp_img.gray_image, cmap="gray", vmin=0, vmax=255)
        roi_mask = _center_roi_mask(exp_img.gray_image)
        _draw_roi_box(ax, roi_mask)
        roi_mean = float(np.mean(exp_img.gray_image[roi_mask == 1]))
        gain_db = exp_img.calculate_equivalent_gain()
        texp = f"1/{1/exp_img.exposure_time:.0f}"
        ax.set_title(f"Iter {idx}", fontsize=11, fontweight="bold")
        ax.text(
            0.02, 0.98,
            f"{exp_img.image_type}\nISO={exp_img.iso:.0f}\nT={texp}\nGain≈{gain_db:.1f} dB\nMean={roi_mean:.1f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="#F5F7FF", edgecolor="#4A6CF7", alpha=0.9),
        )

    fig.suptitle(
        f"算法自适应迭代过程 (条件: {condition_pattern})",
        fontsize=13, fontweight="bold"
    )
    fig.text(
        0.5, 0.02,
        "每一步对应算法选择的 ISO / 曝光(Texp) 组合，ROI 灰度逐步逼近目标值",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#FFF8E1", edgecolor="#C9A227", alpha=0.9),
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return out_path


def generate_real_iteration_strip_from_dataset(data_dir: str,
                                               experiment_type: str,
                                               condition_pattern: str,
                                               max_steps: int = 6,
                                               output_name: str = None,
                                               output_dir: Path = None) -> Path:
    """
    使用真实数据 + 论文公式进行迭代，并选择最接近目标增益的图片序列。
    """
    output_dir = output_dir or _ensure_output_dir()
    out_path = output_dir / (output_name or "algorithm_real_iteration.png")

    loader = ExperimentLoader(base_dir=data_dir)
    images = loader.get_images_by_condition(experiment_type, condition_pattern, image_type="ISO")
    images += loader.get_images_by_condition(experiment_type, condition_pattern, image_type="Texp")
    if not images:
        raise FileNotFoundError(
            f"条件 {condition_pattern} 无可用图片: {data_dir}/{experiment_type}"
        )

    # 预加载所有图片并计算 ROI 均值、等效增益
    records = []
    for img in images:
        img.load()
        if img.gray_image is None:
            continue
        roi_mask = _auto_roi_mask(img.gray_image)
        roi_mean = float(np.mean(img.gray_image[roi_mask == 1]))
        gain_db = img.calculate_equivalent_gain()
        records.append((img, roi_mean, gain_db))

    if not records:
        raise RuntimeError("未找到可用的实验图片")

    # 初始：最低等效增益
    records.sort(key=lambda r: r[2])
    current_img, current_mean, current_gain = records[0]

    max_mean = max([r[1] for r in records]) if records else 255.0
    target_gray = 0.95 * max_mean
    history = [(current_img, current_mean, current_gain)]

    for _ in range(max_steps - 1):
        if current_mean <= 0:
            break
        # 论文公式: G_opt = G_curr * (Y_target / Y_curr)
        optimal_gain = current_gain + 20 * np.log10(target_gray / current_mean)
        # 选取等效增益最接近的真实图像
        next_img, next_mean, next_gain = min(
            records, key=lambda r: abs(r[2] - optimal_gain)
        )
        history.append((next_img, next_mean, next_gain))
        if abs(next_gain - current_gain) < 1e-3:
            break
        current_img, current_mean, current_gain = next_img, next_mean, next_gain

    # 绘图
    cols = 3
    rows = int(np.ceil(len(history) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.0))
    if rows == 1:
        axes = np.array([axes])

    for ax in axes.flatten():
        ax.axis("off")

    for idx, (ax, (exp_img, roi_mean, gain_db)) in enumerate(zip(axes.flatten(), history), start=1):
        ax.imshow(exp_img.gray_image, cmap="gray", vmin=0, vmax=255)
        roi_mask = _auto_roi_mask(exp_img.gray_image)
        _draw_roi_box(ax, roi_mask)
        texp = f"1/{1/exp_img.exposure_time:.0f}"
        ax.set_title(f"Iter {idx}", fontsize=11, fontweight="bold")
        ax.text(
            0.02, 0.98,
            f"{exp_img.image_type}\nISO={exp_img.iso:.0f}\nT={texp}\nGain≈{gain_db:.1f} dB\nMean={roi_mean:.1f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="#F5F7FF", edgecolor="#4A6CF7", alpha=0.9),
        )

    fig.suptitle(
        f"真实算法迭代过程 (条件: {condition_pattern})",
        fontsize=13, fontweight="bold"
    )
    fig.text(
        0.5, 0.02,
        "每一步按公式计算 G_opt，并选取等效增益最接近的真实实验图片",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#FFF8E1", edgecolor="#C9A227", alpha=0.9),
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return out_path


def generate_real_algorithm_effect_figure_for_condition(data_dir: str,
                                                        experiment_type: str,
                                                        condition_pattern: str,
                                                        output_name: str,
                                                        output_dir: Path = None) -> Path:
    """
    基于指定条件生成算法效果图(前后对比)
    """
    output_dir = output_dir or _ensure_output_dir()
    out_path = output_dir / output_name

    loader = ExperimentLoader(base_dir=data_dir)
    images = loader.get_images_by_condition(experiment_type, condition_pattern, image_type="ISO")
    images += loader.get_images_by_condition(experiment_type, condition_pattern, image_type="Texp")
    if not images:
        raise FileNotFoundError(
            f"条件 {condition_pattern} 无可用图片: {data_dir}/{experiment_type}"
        )

    images_sorted = sorted(images, key=lambda img: img.calculate_equivalent_gain())
    before_img = images_sorted[0]
    before_img.load()
    if before_img.gray_image is None:
        raise RuntimeError(f"图片加载失败: {before_img.filepath}")

    roi_means = []
    for img in images_sorted:
        img.load()
        if img.gray_image is None:
            continue
        roi_mask = _auto_roi_mask(img.gray_image)
        roi_means.append(float(np.mean(img.gray_image[roi_mask == 1])))
    max_mean = max(roi_means) if roi_means else 255.0
    target_gray = 0.95 * max_mean
    after_img = _select_image_by_target(images_sorted, target_gray, use_auto_roi=True)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(before_img.gray_image, cmap="gray", vmin=0, vmax=255)
    roi_before = _auto_roi_mask(before_img.gray_image)
    _draw_roi_box(ax1, roi_before)
    mean_before = float(np.mean(before_img.gray_image[roi_before == 1]))
    before_gain = before_img.calculate_equivalent_gain()
    ax1.set_title(
        f"Before (低增益)\nMean={mean_before:.1f}",
        fontsize=12, fontweight="bold"
    )
    ax1.text(
        0.02, 0.98,
        f"ISO={before_img.iso:.0f}\nT=1/{1/before_img.exposure_time:.0f}\nGain={before_gain:.1f} dB",
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#FFF8E1", edgecolor="#C9A227", alpha=0.9),
    )
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(after_img.gray_image, cmap="gray", vmin=0, vmax=255)
    roi_after = _auto_roi_mask(after_img.gray_image)
    _draw_roi_box(ax2, roi_after)
    mean_after = float(np.mean(after_img.gray_image[roi_after == 1]))
    after_gain = after_img.calculate_equivalent_gain()
    ax2.set_title(
        f"After (算法选择)\nMean={mean_after:.1f} 目标≈{target_gray:.1f}",
        fontsize=12, fontweight="bold"
    )
    ax2.text(
        0.02, 0.98,
        f"ISO={after_img.iso:.0f}\nT=1/{1/after_img.exposure_time:.0f}\nGain={after_gain:.1f} dB",
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#E8F5E9", edgecolor="#2D8A34", alpha=0.9),
    )
    ax2.axis("off")

    fig.suptitle(
        f"算法真实效果对比 (条件: {condition_pattern})",
        fontsize=12, fontweight="bold"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return out_path


def generate_all_real_dataset_figures(data_dir: str = "ISO-Texp",
                                      experiment_types: List[str] = None,
                                      num_steps: int = 6) -> Path:
    """
    生成数据集中所有条件的迭代图与前后对比图
    """
    output_dir = Path("results/real_dataset_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = ExperimentLoader(base_dir=data_dir)
    exp_types = experiment_types or ["bubble", "tap water", "turbidity"]

    for exp_type in exp_types:
        all_images = loader.load_experiment(exp_type, image_type="ISO")
        all_images += loader.load_experiment(exp_type, image_type="Texp")
        conditions = sorted({img.condition for img in all_images})

        for condition in conditions:
            safe_condition = condition.replace(" ", "_")
            try:
                generate_real_iteration_strip_from_dataset(
                    data_dir=data_dir,
                    experiment_type=exp_type,
                    condition_pattern=condition,
                    max_steps=num_steps,
                    output_name=f"{exp_type}_{safe_condition}_iteration.png",
                    output_dir=output_dir
                )
                generate_real_algorithm_effect_figure_for_condition(
                    data_dir=data_dir,
                    experiment_type=exp_type,
                    condition_pattern=condition,
                    output_name=f"{exp_type}_{safe_condition}_before_after.png",
                    output_dir=output_dir
                )
            except FileNotFoundError:
                continue

    return output_dir

def main():
    parser = argparse.ArgumentParser(description="生成算法对比示意图与实验原理图")
    parser.add_argument("--real-data-dir", default="ISO-Texp", help="ISO-Texp 数据目录")
    parser.add_argument("--experiment-type", default="bubble", help="实验类型(bubble/tap water/turbidity)")
    parser.add_argument("--skip-sim", action="store_true", help="跳过仿真实验原理图")
    parser.add_argument("--all-real", action="store_true", help="生成全数据集迭代图和前后对比图")
    args = parser.parse_args()

    diagram1 = generate_algorithm_comparison_diagram()
    print(f"已生成: {diagram1}")

    if not args.skip_sim:
        diagram2 = generate_experiment_principle_diagram()
        print(f"已生成: {diagram2}")

    try:
        diagram3 = generate_experiment_principle_diagram_real(
            data_dir=args.real_data_dir,
            experiment_type=args.experiment_type
        )
        print(f"已生成: {diagram3}")
    except FileNotFoundError as exc:
        print(f"未生成真实实验图: {exc}")

    try:
        diagram4 = generate_real_algorithm_effect_figure(
            data_dir=args.real_data_dir,
            experiment_type=args.experiment_type
        )
        print(f"已生成: {diagram4}")
    except FileNotFoundError as exc:
        print(f"未生成算法真实效果图: {exc}")

    try:
        diagram5 = generate_adaptive_iteration_strip(
            data_dir=args.real_data_dir,
            experiment_type=args.experiment_type,
            condition_pattern="bubble_1_2",
            num_steps=6
        )
        print(f"已生成: {diagram5}")
    except FileNotFoundError as exc:
        print(f"未生成自适应迭代图: {exc}")

    if args.all_real:
        output_dir = generate_all_real_dataset_figures(
            data_dir=args.real_data_dir,
            experiment_types=None,
            num_steps=6
        )
        print(f"已生成全数据集图像到: {output_dir}")


if __name__ == "__main__":
    main()
