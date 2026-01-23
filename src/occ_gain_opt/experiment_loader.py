"""
实验图片加载器
用于加载和解析真实实验图片数据
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class ExperimentImage:
    """实验图片数据类"""

    def __init__(self, filepath: str, exposure_time: float, iso: float,
                 sequence_length: int, condition: str, index: int, image_type: str):
        """
        初始化实验图片

        Args:
            filepath: 图片文件路径
            exposure_time: 曝光时间（秒）
            iso: 感光度
            sequence_length: 随机序列长度
            condition: 实验条件描述（如bubble_1_2_2）
            index: 图片索引
            image_type: 图片类型（ISO或Texp）
        """
        self.filepath = filepath
        self.exposure_time = exposure_time
        self.iso = iso
        self.sequence_length = sequence_length
        self.condition = condition
        self.index = index
        self.image_type = image_type
        self.image = None
        self.gray_image = None

    def load(self) -> bool:
        """加载图片"""
        self.image = cv2.imread(self.filepath)
        if self.image is not None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            return True
        return False

    def calculate_equivalent_gain(self, base_exposure: float = 1/27000,
                                  base_iso: float = 35) -> float:
        """
        计算等效增益值（相对于基准条件的增益）

        Args:
            base_exposure: 基准曝光时间（默认1/27000秒，来自说明文档）
            base_iso: 基准ISO（默认35，来自说明文档）

        Returns:
            等效增益值（dB）
        """
        # 曝光时间增益（线性）
        exposure_gain = self.exposure_time / base_exposure
        # ISO增益（线性）
        iso_gain = self.iso / base_iso
        # 总增益（线性）
        total_gain_linear = exposure_gain * iso_gain
        # 转换为dB
        total_gain_db = 20 * np.log10(total_gain_linear) if total_gain_linear > 0 else 0
        return float(total_gain_db)

    def get_filename_info(self) -> Dict:
        """获取文件名信息字典"""
        return {
            'filepath': self.filepath,
            'exposure_time': self.exposure_time,
            'iso': self.iso,
            'sequence_length': self.sequence_length,
            'condition': self.condition,
            'index': self.index,
            'image_type': self.image_type,
            'equivalent_gain': self.calculate_equivalent_gain()
        }


class ExperimentLoader:
    """实验数据加载器"""

    def __init__(self, base_dir: str = "ISO-Texp"):
        """
        初始化实验数据加载器

        Args:
            base_dir: 实验数据根目录
        """
        self.base_dir = Path(base_dir)
        self.images: List[ExperimentImage] = []

    @staticmethod
    def parse_filename(filename: str) -> Optional[Dict]:
        """
        解析文件名，提取参数

        文件名格式: 曝光时间_感光度_p32_条件_索引.jpg
        例如: 35800_1595_p32_bubble_1_2_1.jpg
        """
        pattern = r'(\d+)_(\d+)_p(\d+)_(.+?)(?:_\d+\.jpg|\.jpg)'
        match = re.match(pattern, filename)

        if match:
            exposure_denominator = int(match.group(1))
            iso = float(match.group(2))
            seq_length = int(match.group(3))
            condition = match.group(4)

            # 曝光时间 = 1 / 分母
            exposure_time = 1.0 / exposure_denominator

            return {
                'exposure_time': exposure_time,
                'iso': iso,
                'sequence_length': seq_length,
                'condition': condition
            }
        return None

    def load_experiment(self, experiment_type: str = "bubble",
                       image_type: str = "ISO") -> List[ExperimentImage]:
        """
        加载特定实验的图片

        Args:
            experiment_type: 实验类型（bubble, tap water, turbidity）
            image_type: 图片类型（ISO或Texp）

        Returns:
            ExperimentImage列表
        """
        exp_dir = self.base_dir / experiment_type / image_type
        if not exp_dir.exists():
            print(f"目录不存在: {exp_dir}")
            return []

        images = []
        for filepath in exp_dir.glob("*.jpg"):
            info = self.parse_filename(filepath.name)
            if info:
                # 从完整文件名中提取索引
                index_pattern = r'.*?_(\d+)\.jpg$'
                index_match = re.search(index_pattern, filepath.name)
                index = int(index_match.group(1)) if index_match else 0

                exp_image = ExperimentImage(
                    filepath=str(filepath),
                    exposure_time=info['exposure_time'],
                    iso=info['iso'],
                    sequence_length=info['sequence_length'],
                    condition=info['condition'],
                    index=index,
                    image_type=image_type
                )
                images.append(exp_image)

        # 按曝光时间和ISO排序
        images.sort(key=lambda x: (x.exposure_time, x.iso))
        return images

    def load_all_experiments(self) -> Dict[str, Dict[str, List[ExperimentImage]]]:
        """
        加载所有实验数据

        Returns:
            嵌套字典: {experiment_type: {image_type: [images]}}
        """
        experiment_types = ["bubble", "tap water", "turbidity"]
        image_types = ["ISO", "Texp"]

        all_data = {}
        for exp_type in experiment_types:
            all_data[exp_type] = {}
            for img_type in image_types:
                images = self.load_experiment(exp_type, img_type)
                all_data[exp_type][img_type] = images
                print(f"加载 {exp_type}/{img_type}: {len(images)} 张图片")

        return all_data

    def get_images_by_condition(self, experiment_type: str,
                                condition_pattern: str,
                                image_type: str = "ISO") -> List[ExperimentImage]:
        """
        根据条件筛选图片

        Args:
            experiment_type: 实验类型
            condition_pattern: 条件模式（如"bubble_1_2"）
            image_type: 图片类型

        Returns:
            匹配的图片列表
        """
        images = self.load_experiment(experiment_type, image_type)
        filtered = [img for img in images if condition_pattern in img.condition]
        return filtered

    def get_images_by_iso(self, experiment_type: str, iso: float,
                         image_type: str = "ISO") -> List[ExperimentImage]:
        """根据ISO值筛选图片"""
        images = self.load_experiment(experiment_type, image_type)
        filtered = [img for img in images if abs(img.iso - iso) < 0.1]
        return filtered

    def get_images_by_exposure(self, experiment_type: str,
                               exposure_time: float,
                               tolerance: float = 1e-6,
                               image_type: str = "ISO") -> List[ExperimentImage]:
        """根据曝光时间筛选图片"""
        images = self.load_experiment(experiment_type, image_type)
        filtered = [img for img in images
                   if abs(img.exposure_time - exposure_time) < tolerance]
        return filtered

    def analyze_exposure_distribution(self, experiment_type: str,
                                      image_type: str = "ISO") -> Dict:
        """
        分析实验图片的曝光和ISO分布

        Returns:
            统计信息字典
        """
        images = self.load_experiment(experiment_type, image_type)

        if not images:
            return {}

        exposure_times = [img.exposure_time for img in images]
        isos = [img.iso for img in images]
        gains = [img.calculate_equivalent_gain() for img in images]

        return {
            'total_images': len(images),
            'exposure_range': (min(exposure_times), max(exposure_times)),
            'iso_range': (min(isos), max(isos)),
            'gain_range': (min(gains), max(gains)),
            'unique_exposures': len(set(exposure_times)),
            'unique_isos': len(set(isos)),
            'conditions': list(set(img.condition for img in images))
        }


def print_experiment_summary(experiment_type: str = "bubble"):
    """打印实验数据摘要"""
    loader = ExperimentLoader()

    print(f"\n=== {experiment_type} 实验数据摘要 ===\n")

    for img_type in ["ISO", "Texp"]:
        stats = loader.analyze_exposure_distribution(experiment_type, img_type)
        if stats:
            print(f"[{img_type}]")
            print(f"  总图片数: {stats['total_images']}")
            print(f"  曝光时间范围: {stats['exposure_range'][0]:.6f} - {stats['exposure_range'][1]:.6f} 秒")
            print(f"  ISO范围: {stats['iso_range'][0]:.0f} - {stats['iso_range'][1]:.0f}")
            print(f"  等效增益范围: {stats['gain_range'][0]:.2f} - {stats['gain_range'][1]:.2f} dB")
            print(f"  唯一曝光值数: {stats['unique_exposures']}")
            print(f"  唯一ISO值数: {stats['unique_isos']}")
            print(f"  实验条件: {', '.join(sorted(stats['conditions']))}")
            print()
