"""
性能评估模块
实现MSE、PSNR、SSIM等性能指标计算
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import ndimage

from .config import PerformanceConfig


class PerformanceEvaluator:
    """性能评估器 - 计算图像质量和通信性能指标"""

    def __init__(self):
        self.metrics_history = []

    def calculate_mse(self, original: np.ndarray,
                      processed: np.ndarray) -> float:
        """计算均方误差 (MSE) - 论文公式10"""
        if original.shape != processed.shape:
            raise ValueError("图像尺寸不匹配")
        mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
        return float(mse)

    def calculate_local_mse(self, original: np.ndarray,
                            processed: np.ndarray,
                            window_size: int = PerformanceConfig.WINDOW_SIZE) -> np.ndarray:
        """计算局部MSE"""
        if original.shape != processed.shape:
            raise ValueError("图像尺寸不匹配")

        se = (original.astype(float) - processed.astype(float)) ** 2
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        local_mse = ndimage.convolve(se, kernel, mode='constant')

        return local_mse

    def calculate_psnr(self, original: np.ndarray,
                       processed: np.ndarray,
                       max_value: int = 255) -> float:
        """计算峰值信噪比 (PSNR)"""
        mse = self.calculate_mse(original, processed)
        if mse == 0:
            return float('inf')
        psnr = 10 * np.log10((max_value ** 2) / mse)
        return float(psnr)

    def calculate_snr(self, signal: np.ndarray,
                      noise: np.ndarray) -> float:
        """计算信噪比 (SNR)"""
        signal_power = np.mean(signal.astype(float) ** 2)
        noise_power = np.mean(noise.astype(float) ** 2)
        if noise_power == 0:
            return float('inf')
        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)

    def calculate_ssim(self, original: np.ndarray,
                       processed: np.ndarray,
                       window_size: int = 7,
                       k1: float = 0.01,
                       k2: float = 0.03) -> float:
        """计算结构相似性指数 (SSIM)"""
        C1 = (k1 * 255) ** 2
        C2 = (k2 * 255) ** 2

        orig = original.astype(float)
        proc = processed.astype(float)

        mu_orig = ndimage.uniform_filter(orig, window_size)
        mu_proc = ndimage.uniform_filter(proc, window_size)

        sigma_orig_sq = ndimage.uniform_filter(orig ** 2, window_size) - mu_orig ** 2
        sigma_proc_sq = ndimage.uniform_filter(proc ** 2, window_size) - mu_proc ** 2
        sigma_orig_proc = ndimage.uniform_filter(orig * proc, window_size) - mu_orig * mu_proc

        numerator = (2 * mu_orig * mu_proc + C1) * (2 * sigma_orig_proc + C2)
        denominator = (mu_orig ** 2 + mu_proc ** 2 + C1) * (sigma_orig_sq + sigma_proc_sq + C2)

        ssim_map = numerator / denominator
        ssim = np.mean(ssim_map)
        return float(ssim)

    def _apply_roi(self, image: np.ndarray, roi_mask: Optional[np.ndarray]) -> np.ndarray:
        if roi_mask is None:
            return image
        return image[roi_mask == 1]

    def evaluate_reference_metrics(self, reference: np.ndarray,
                                   processed: np.ndarray,
                                   roi_mask: Optional[np.ndarray] = None) -> Dict:
        """在可选ROI上计算图像质量指标"""
        ref = self._apply_roi(reference, roi_mask)
        proc = self._apply_roi(processed, roi_mask)

        metrics = {
            'mse': self.calculate_mse(ref, proc),
            'psnr': self.calculate_psnr(ref, proc),
            'ssim': self.calculate_ssim(ref, proc)
        }
        return metrics

    def evaluate_communication_performance(self, received_signal: np.ndarray,
                                           original_signal: np.ndarray,
                                           threshold: float = 127) -> Dict:
        """评估通信性能"""
        received_binary = (received_signal > threshold).astype(int)
        original_binary = (original_signal > threshold).astype(int)

        total_bits = len(received_binary)
        error_bits = np.sum(received_binary != original_binary)
        ber = error_bits / total_bits if total_bits > 0 else 0
        accuracy = 1 - ber

        signal_mean = np.mean(received_signal)
        signal_std = np.std(received_signal)
        noise_estimate = signal_std / np.sqrt(2)

        metrics = {
            'ber': ber,
            'accuracy': accuracy,
            'total_bits': total_bits,
            'error_bits': error_bits,
            'signal_mean': signal_mean,
            'signal_std': signal_std,
            'noise_estimate': noise_estimate,
            'snr': self.calculate_snr(
                received_signal - np.mean(received_signal),
                noise_estimate * np.random.randn(*received_signal.shape)
            )
        }
        return metrics

    def evaluate_optimization_result(self, result: Dict,
                                     target_gray: float = 255,
                                     reference_image: Optional[np.ndarray] = None,
                                     roi_mask: Optional[np.ndarray] = None) -> Dict:
        """评估优化结果"""
        final_gray = result['final_gray']
        optimal_gain = result['optimal_gain']

        gray_error = abs(target_gray - final_gray)
        gray_error_percent = (gray_error / target_gray) * 100

        saturated_ratio = result['final_stats']['saturated_ratio']
        converged = result['converged']
        iterations = result['iterations']

        stats = result['final_stats']
        signal_quality = {
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max'],
            'dynamic_range': stats['max'] - stats['min']
        }

        image_metrics = None
        if reference_image is not None:
            image_metrics = self.evaluate_reference_metrics(
                reference_image, result['final_image'], roi_mask=roi_mask
            )

        evaluation = {
            'gain_optimal': optimal_gain,
            'gray_value': final_gray,
            'gray_error': gray_error,
            'gray_error_percent': gray_error_percent,
            'saturated_ratio': saturated_ratio,
            'converged': converged,
            'iterations': iterations,
            'signal_quality': signal_quality,
            'image_metrics': image_metrics,
            'optimization_score': self._calculate_optimization_score(
                gray_error, saturated_ratio, iterations
            )
        }

        return evaluation

    def _calculate_optimization_score(self, gray_error: float,
                                      saturated_ratio: float,
                                      iterations: int) -> float:
        """计算优化得分 (0-100)"""
        error_score = max(0, 100 - (gray_error / 255) * 100)

        if saturated_ratio < 0.1:
            saturation_score = 100
        elif saturated_ratio < 0.5:
            saturation_score = 100 - (saturated_ratio / 0.5) * 20
        else:
            saturation_score = 80 - ((saturated_ratio - 0.5) / 0.5) * 80

        efficiency_score = max(0, 100 - (iterations / 20) * 20)

        total_score = (0.5 * error_score +
                       0.3 * saturation_score +
                       0.2 * efficiency_score)

        return float(total_score)

    def compare_algorithms(self, results1: Dict, results2: Dict,
                           algorithm_names: Tuple[str, str] = ("Algorithm 1", "Algorithm 2")) -> Dict:
        """比较两种算法的性能"""
        eval1 = self.evaluate_optimization_result(
            results1,
            reference_image=results1.get('reference_image'),
            roi_mask=results1.get('roi_mask')
        )
        eval2 = self.evaluate_optimization_result(
            results2,
            reference_image=results2.get('reference_image'),
            roi_mask=results2.get('roi_mask')
        )

        comparison = {
            'algorithm1': algorithm_names[0],
            'algorithm2': algorithm_names[1],
            'metrics': {
                'gray_error': {
                    algorithm_names[0]: eval1['gray_error'],
                    algorithm_names[1]: eval2['gray_error'],
                    'winner': algorithm_names[0] if eval1['gray_error'] < eval2['gray_error'] else algorithm_names[1]
                },
                'iterations': {
                    algorithm_names[0]: eval1['iterations'],
                    algorithm_names[1]: eval2['iterations'],
                    'winner': algorithm_names[0] if eval1['iterations'] < eval2['iterations'] else algorithm_names[1]
                },
                'score': {
                    algorithm_names[0]: eval1['optimization_score'],
                    algorithm_names[1]: eval2['optimization_score'],
                    'winner': algorithm_names[0] if eval1['optimization_score'] > eval2['optimization_score'] else algorithm_names[1]
                }
            }
        }

        return comparison

    def batch_evaluate(self, results: List[Dict]) -> Dict:
        """批量评估多个结果"""
        evaluations = [
            self.evaluate_optimization_result(
                r, reference_image=r.get('reference_image'), roi_mask=r.get('roi_mask')
            )
            for r in results
        ]

        gray_errors = [e['gray_error'] for e in evaluations]
        iterations = [e['iterations'] for e in evaluations]
        scores = [e['optimization_score'] for e in evaluations]
        saturated_ratios = [e['saturated_ratio'] for e in evaluations]

        summary = {
            'count': len(evaluations),
            'gray_error': {
                'mean': float(np.mean(gray_errors)),
                'std': float(np.std(gray_errors)),
                'min': float(np.min(gray_errors)),
                'max': float(np.max(gray_errors))
            },
            'iterations': {
                'mean': float(np.mean(iterations)),
                'std': float(np.std(iterations)),
                'min': float(np.min(iterations)),
                'max': float(np.max(iterations))
            },
            'optimization_score': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            },
            'saturated_ratio': {
                'mean': float(np.mean(saturated_ratios)),
                'std': float(np.std(saturated_ratios))
            },
            'convergence_rate': float(np.sum([e['converged'] for e in evaluations]) / len(evaluations))
        }

        return summary
