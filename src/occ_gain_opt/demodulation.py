"""
OOK 解调模块
从滚动快门相机捕获的条纹图像中提取数据，检测同步头，定位完整数据包。

同步头特征: 一段超长的连续全亮 (all-1) 区域，在随机数据中不可能自然出现。
典型结构: [Sync: N bits 全1] [Data: 32 bits 随机] [Sync] [Data] ...
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import DemodulationConfig


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class DemodulationResult:
    """解调结果"""
    row_profile: np.ndarray                # 行均值曲线
    binary_profile: np.ndarray             # 二值化后的行曲线 (每行 0/1)
    threshold: float                       # 二值化阈值
    bit_period: float                      # 位周期 (行数/bit)
    bit_sequence: np.ndarray               # 采样后的比特序列
    sample_positions: np.ndarray           # 采样位置 (行号)
    sync_pattern: Optional[np.ndarray]     # 同步头模式 (全1)
    sync_positions_bit: List[int]          # 同步头在 bit_sequence 中的起始位置
    sync_positions_row: List[int]          # 同步头在图像行中的起始行号
    packets: List[np.ndarray]              # 提取的数据包列表 (不含同步头)
    col_bounds: Tuple[int, int]            # LED 列边界 (col_start, col_end)
    roi_mask: np.ndarray                   # 完整数据包区域 ROI 掩码
    confidence: float                      # 检测置信度 (0-1)
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 基础信号处理函数
# ---------------------------------------------------------------------------

def detect_led_column_bounds(image: np.ndarray,
                             channel: str = DemodulationConfig.LED_CHANNEL,
                             margin_ratio: float = DemodulationConfig.COL_MARGIN_RATIO
                             ) -> Tuple[int, int]:
    """
    检测 LED 光源的列范围

    通过列均值 Otsu 分割找到 LED 所在列区间，再向内收缩避免边缘噪声。
    """
    ch = _extract_channel(image, channel)
    col_profile = np.mean(ch, axis=0)

    col_u8 = np.clip(col_profile, 0, 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(col_u8, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    above = np.where(col_profile > thresh_val)[0]
    if len(above) == 0:
        w = ch.shape[1]
        return int(w * 0.2), int(w * 0.8)

    col_start, col_end = int(above[0]), int(above[-1]) + 1
    margin = int((col_end - col_start) * margin_ratio)
    col_start = min(col_start + margin, col_end - 1)
    col_end = max(col_end - margin, col_start + 1)
    return col_start, col_end


def extract_row_profile(image: np.ndarray,
                        col_start: int, col_end: int,
                        channel: str = DemodulationConfig.LED_CHANNEL,
                        smoothing_sigma: float = DemodulationConfig.SMOOTHING_SIGMA
                        ) -> np.ndarray:
    """在 LED 列范围内计算行均值 + 高斯平滑"""
    ch = _extract_channel(image, channel)
    roi = ch[:, col_start:col_end].astype(np.float64)
    row_profile = np.mean(roi, axis=1)

    if smoothing_sigma > 0:
        row_profile = gaussian_filter1d(row_profile, sigma=smoothing_sigma)
    return row_profile


def binarize_profile(row_profile: np.ndarray,
                     method: str = DemodulationConfig.THRESHOLD_METHOD
                     ) -> Tuple[np.ndarray, float]:
    """将行均值曲线二值化 (支持 otsu / midpoint / percentile)"""
    if method == "otsu":
        p_min, p_max = row_profile.min(), row_profile.max()
        if p_max - p_min < 1e-6:
            return np.zeros_like(row_profile, dtype=np.uint8), float(p_min)
        norm = ((row_profile - p_min) / (p_max - p_min) * 255).astype(np.uint8)
        thresh_u8, _ = cv2.threshold(norm, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = float(thresh_u8) / 255.0 * (p_max - p_min) + p_min
    elif method == "midpoint":
        threshold = (row_profile.max() + row_profile.min()) / 2.0
    elif method == "percentile":
        low = np.percentile(row_profile, DemodulationConfig.PERCENTILE_LOW)
        high = np.percentile(row_profile, DemodulationConfig.PERCENTILE_HIGH)
        threshold = (low + high) / 2.0
    else:
        raise ValueError(f"未知二值化方法: {method}")

    binary = (row_profile >= threshold).astype(np.uint8)
    return binary, threshold


# ---------------------------------------------------------------------------
# 同步头检测 (核心: 找超长全亮游程)
# ---------------------------------------------------------------------------

def find_sync_runs(binary_profile: np.ndarray,
                   min_bit_period: int = DemodulationConfig.MIN_BIT_PERIOD,
                   max_bit_period: int = DemodulationConfig.MAX_BIT_PERIOD,
                   data_bits: int = DemodulationConfig.DATA_BITS,
                   sync_value: int = 1
                   ) -> Tuple[List[Tuple[int, int]], float, int]:
    """
    从二值化行信号中找到同步头 (超长连续全亮/全暗段)

    策略:
    1. 计算所有游程及其值
    2. 统计 sync_value (默认1=亮) 方向的游程长度分布
    3. 数据区的最长游程 ≈ 几个位周期 (p32中连续相同位最多约5-6个)
    4. 同步头 = 远超数据区最长游程的超长段 (≥ 数据区最长的2倍)

    Args:
        binary_profile: 二值化行数组
        min_bit_period: 最小位周期
        max_bit_period: 最大位周期
        data_bits: 数据位长度
        sync_value: 同步头的值 (1=全亮, 0=全暗)

    Returns:
        (sync_runs, bit_period, sync_bits)
        sync_runs: [(start_row, length), ...] 同步头列表
        bit_period: 精确位周期
        sync_bits: 同步头的位数
    """
    runs, values, starts = _compute_runs_detailed(binary_profile)
    if len(runs) == 0:
        return [], float(min_bit_period), 0

    # 分离 sync_value 方向的游程
    sv_mask = values == sync_value
    sv_runs = runs[sv_mask]
    sv_starts = starts[sv_mask]

    if len(sv_runs) == 0:
        return [], float(min_bit_period), 0

    # --- Step 1: 粗估位周期 (用最短有效游程) ---
    all_valid = runs[(runs >= min_bit_period) & (runs <= max_bit_period)]
    if len(all_valid) == 0:
        all_valid = runs[runs >= min_bit_period // 2]
    if len(all_valid) == 0:
        return [], float(min_bit_period), 0

    min_run = all_valid.min()
    cluster = all_valid[all_valid <= min_run * 1.5]
    bit_period_rough = float(np.median(cluster))

    # --- Step 2: 识别超长游程 = 同步头 ---
    # 在 p32 随机序列中，连续相同位最长约 5-6 bit (概率极低)
    # 同步头应 >= 6 个位周期长度
    sync_threshold_rows = bit_period_rough * 5.5

    sync_mask = (sv_runs >= sync_threshold_rows)
    sync_indices = np.where(sync_mask)[0]

    if len(sync_indices) == 0:
        # 降低阈值重试 (4个位周期)
        sync_threshold_rows = bit_period_rough * 4.0
        sync_mask = (sv_runs >= sync_threshold_rows)
        sync_indices = np.where(sync_mask)[0]

    if len(sync_indices) == 0:
        return [], bit_period_rough, 0

    sync_run_list = [(int(sv_starts[i]), int(sv_runs[i])) for i in sync_indices]

    # --- Step 3: 用同步头间距精确校准位周期 ---
    if len(sync_run_list) >= 2:
        spacings = []
        for i in range(len(sync_run_list) - 1):
            spacing = sync_run_list[i + 1][0] - sync_run_list[i][0]
            spacings.append(spacing)

        # 同步头间距 = sync_bits + data_bits 个位周期
        # 先用粗估位周期估算 sync_bits
        avg_sync_rows = np.mean([r[1] for r in sync_run_list])
        sync_bits_est = round(avg_sync_rows / bit_period_rough)
        packet_bits = sync_bits_est + data_bits

        if packet_bits > 0:
            # 用间距精确反推位周期
            median_spacing = float(np.median(spacings))
            bit_period_precise = median_spacing / packet_bits
            # 合理性检查
            if min_bit_period * 0.5 <= bit_period_precise <= max_bit_period * 2:
                bit_period_rough = bit_period_precise
                # 重新校准 sync_bits
                sync_bits_est = round(avg_sync_rows / bit_period_rough)

        sync_bits = max(sync_bits_est, 1)
    else:
        avg_sync_rows = sync_run_list[0][1]
        sync_bits = max(round(avg_sync_rows / bit_period_rough), 1)

    return sync_run_list, bit_period_rough, sync_bits


def sample_bits_aligned(binary_profile: np.ndarray,
                        sync_start_row: int,
                        sync_len_rows: int,
                        bit_period: float
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    以同步头为锚点对齐采样

    从同步头结束位置开始，向前向后按位周期采样。

    Args:
        binary_profile: 二值化行数组
        sync_start_row: 同步头起始行
        sync_len_rows: 同步头行数
        bit_period: 位周期

    Returns:
        (bit_values, sample_positions) — 覆盖整个图像的采样
    """
    n = len(binary_profile)
    half = bit_period / 2.0

    # 同步头结束后第一个数据位的中心
    data_start = sync_start_row + sync_len_rows + half

    # 向后采样 (数据 + 后续同步头 + 更多数据)
    positions_fwd = []
    pos = data_start
    while pos < n:
        positions_fwd.append(pos)
        pos += bit_period

    # 同步头本身的采样 (从同步头起始向后)
    positions_sync = []
    pos = sync_start_row + half
    while pos < sync_start_row + sync_len_rows:
        positions_sync.append(pos)
        pos += bit_period

    # 向前采样 (同步头之前的数据)
    positions_bwd = []
    pos = sync_start_row - half
    while pos >= 0:
        positions_bwd.append(pos)
        pos -= bit_period
    positions_bwd.reverse()

    all_pos = np.array(positions_bwd + positions_sync + positions_fwd)
    valid = (all_pos >= 0) & (all_pos < n)
    all_pos = all_pos[valid]

    indices = np.clip(np.round(all_pos).astype(int), 0, n - 1)
    bit_values = binary_profile[indices].astype(np.uint8)

    return bit_values, all_pos


def locate_sync_in_bits(bit_sequence: np.ndarray,
                        sync_bits: int,
                        data_bits: int = DemodulationConfig.DATA_BITS,
                        sync_value: int = 1
                        ) -> Tuple[np.ndarray, List[int], float]:
    """
    在采样后的比特序列中精确定位同步头

    同步头 = sync_bits 个连续 sync_value

    Returns:
        (sync_pattern, positions, confidence)
    """
    n = len(bit_sequence)
    sync_pattern = np.full(sync_bits, sync_value, dtype=np.uint8)

    if n < sync_bits:
        return sync_pattern, [], 0.0

    # 计算每个位置的连续 sync_value 计数
    consec = np.zeros(n, dtype=int)
    count = 0
    for i in range(n):
        if bit_sequence[i] == sync_value:
            count += 1
        else:
            count = 0
        consec[i] = count

    # 找到连续 >= sync_bits 的段的起始位置
    positions = []
    i = 0
    while i < n:
        if consec[i] >= sync_bits:
            # 找这段的起始
            start = i - consec[i] + 1
            if not positions or start > positions[-1] + sync_bits + data_bits // 2:
                positions.append(start)
            # 跳过当前段
            i += 1
            while i < n and bit_sequence[i] == sync_value:
                i += 1
        else:
            i += 1

    confidence = 1.0 if len(positions) >= 2 else (0.5 if len(positions) == 1 else 0.0)
    return sync_pattern, positions, confidence


def create_sync_roi_mask(image_shape: Tuple[int, ...],
                         sync_positions_row: List[int],
                         packet_len_rows: int,
                         col_start: int,
                         col_end: int) -> np.ndarray:
    """
    根据同步头位置创建精确的数据包 ROI 掩码

    ROI 覆盖: 从每个同步头起始到下一个同步头起始 (一个完整数据包)。
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(len(sync_positions_row)):
        row_start = sync_positions_row[i]
        if i + 1 < len(sync_positions_row):
            row_end = sync_positions_row[i + 1]
        else:
            row_end = min(row_start + packet_len_rows, h)
        if row_end > row_start:
            mask[row_start:row_end, col_start:col_end] = 1

    return mask


# ---------------------------------------------------------------------------
# 高级封装: OOK 解调器
# ---------------------------------------------------------------------------

class OOKDemodulator:
    """OOK 解调器 — 从滚动快门图像中提取数据并定位同步头"""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.channel = cfg.get('channel', DemodulationConfig.LED_CHANNEL)
        self.margin_ratio = cfg.get('margin_ratio', DemodulationConfig.COL_MARGIN_RATIO)
        self.smoothing_sigma = cfg.get('smoothing_sigma', DemodulationConfig.SMOOTHING_SIGMA)
        self.threshold_method = cfg.get('threshold_method', DemodulationConfig.THRESHOLD_METHOD)
        self.min_bit_period = cfg.get('min_bit_period', DemodulationConfig.MIN_BIT_PERIOD)
        self.max_bit_period = cfg.get('max_bit_period', DemodulationConfig.MAX_BIT_PERIOD)
        self.data_bits = cfg.get('data_bits', DemodulationConfig.DATA_BITS)

    def demodulate(self, image: np.ndarray) -> DemodulationResult:
        """
        完整解调流水线

        1. 检测 LED 列范围
        2. 提取行均值曲线 + 二值化
        3. 找同步头 (超长全亮段) + 精确位周期
        4. 以同步头为锚点对齐位采样
        5. 在比特序列中定位同步头 + 提取数据包
        6. 生成精确 ROI 掩码
        """
        # 1. LED 列范围
        col_start, col_end = detect_led_column_bounds(
            image, self.channel, self.margin_ratio)

        # 2. 行均值 + 二值化
        row_profile = extract_row_profile(
            image, col_start, col_end, self.channel, self.smoothing_sigma)
        binary_profile, threshold = binarize_profile(
            row_profile, self.threshold_method)

        # 3. 找同步头 (行级) + 精确位周期
        sync_runs, bit_period, sync_bits = find_sync_runs(
            binary_profile, self.min_bit_period, self.max_bit_period,
            self.data_bits)

        # 4. 以第一个同步头为锚点对齐采样
        if sync_runs:
            first_sync_start, first_sync_len = sync_runs[0]
            sync_len_rows = int(round(sync_bits * bit_period))
            bit_sequence, sample_positions = sample_bits_aligned(
                binary_profile, first_sync_start, sync_len_rows, bit_period)
        else:
            # 无同步头: 退化为均匀采样
            bit_sequence, sample_positions = _uniform_sample(
                binary_profile, bit_period)
            sync_bits = 0

        # 5. 在比特序列中定位同步头
        if sync_bits > 0:
            sync_pattern, sync_pos_bit, confidence = locate_sync_in_bits(
                bit_sequence, sync_bits, self.data_bits)
        else:
            sync_pattern = None
            sync_pos_bit = []
            confidence = 0.0

        # 6. 行级同步位置 + ROI
        sync_pos_row = []
        for bp in sync_pos_bit:
            if bp < len(sample_positions):
                row = int(round(sample_positions[bp] - bit_period / 2.0))
                sync_pos_row.append(max(0, row))

        packet_bits = sync_bits + self.data_bits
        packet_len_rows = int(round(packet_bits * bit_period))

        roi_mask = create_sync_roi_mask(
            image.shape, sync_pos_row, packet_len_rows, col_start, col_end)

        # 提取数据包 (同步头之后的 data_bits 个位)
        packets = []
        for bp in sync_pos_bit:
            pkt_start = bp + sync_bits
            pkt_end = pkt_start + self.data_bits
            if pkt_end <= len(bit_sequence):
                packets.append(bit_sequence[pkt_start:pkt_end].copy())

        stats = self._compute_stats(row_profile, binary_profile)

        return DemodulationResult(
            row_profile=row_profile,
            binary_profile=binary_profile,
            threshold=threshold,
            bit_period=bit_period,
            bit_sequence=bit_sequence,
            sample_positions=sample_positions,
            sync_pattern=sync_pattern,
            sync_positions_bit=sync_pos_bit,
            sync_positions_row=sync_pos_row,
            packets=packets,
            col_bounds=(col_start, col_end),
            roi_mask=roi_mask,
            confidence=confidence,
            stats=stats,
        )

    def get_packet_roi_mask(self, image: np.ndarray) -> np.ndarray:
        """仅返回数据包区域的 ROI 掩码"""
        return self.demodulate(image).roi_mask

    def get_signal_quality(self, image: np.ndarray) -> dict:
        """计算通信相关信号质量指标"""
        return self.demodulate(image).stats

    @staticmethod
    def _compute_stats(row_profile: np.ndarray,
                       binary_profile: np.ndarray) -> dict:
        bright = row_profile[binary_profile == 1]
        dark = row_profile[binary_profile == 0]

        if len(bright) == 0 or len(dark) == 0:
            return {'eye_opening': 0, 'snr_db': 0}

        bright_mean = float(np.mean(bright))
        dark_mean = float(np.mean(dark))
        eye_opening = bright_mean - dark_mean
        noise_std = (float(np.std(bright)) + float(np.std(dark))) / 2.0
        snr = eye_opening / noise_std if noise_std > 1e-6 else 0.0

        return {
            'eye_opening': eye_opening,
            'bright_mean': bright_mean,
            'dark_mean': dark_mean,
            'snr_linear': snr,
            'snr_db': 20 * np.log10(max(snr, 1e-6)),
        }


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

def _extract_channel(image: np.ndarray, channel: str) -> np.ndarray:
    """提取指定颜色通道"""
    if image.ndim == 2:
        return image
    mapping = {"blue": 0, "green": 1, "red": 2}
    if channel == "gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image[:, :, mapping.get(channel, 1)]


def _compute_runs_detailed(binary: np.ndarray):
    """
    计算二值序列的游程，返回 (长度, 值, 起始位置)
    """
    if len(binary) == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=int))
    diffs = np.diff(binary.astype(np.int8))
    change_idx = np.where(diffs != 0)[0] + 1
    boundaries = np.concatenate([[0], change_idx, [len(binary)]])
    lengths = np.diff(boundaries)
    values = binary[boundaries[:-1]]
    return lengths, values, boundaries[:-1]


def _uniform_sample(binary_profile: np.ndarray,
                    bit_period: float
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """无同步头时的均匀采样 (退化模式)"""
    n = len(binary_profile)
    edges = np.where(np.diff(binary_profile.astype(np.int8)) != 0)[0]
    if len(edges) == 0:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.float64)

    start = edges[0] + bit_period / 2.0
    positions = np.arange(start, n, bit_period)
    indices = np.clip(np.round(positions).astype(int), 0, n - 1)
    return binary_profile[indices].astype(np.uint8), positions
