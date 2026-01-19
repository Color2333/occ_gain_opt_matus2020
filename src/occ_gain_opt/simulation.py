"""
仿真实验模块
模拟论文中的实验场景
"""

from typing import Dict, List

import numpy as np

from .config import ExperimentConfig, CameraConfig, ROIStrategy
from .data_acquisition import DataAcquisition
from .gain_optimization import GainOptimizer, AdaptiveGainOptimizer
from .performance_evaluation import PerformanceEvaluator


class ExperimentSimulation:
    """实验仿真类 - 模拟论文中的实验场景"""

    def __init__(self):
        self.data_acq = DataAcquisition()
        self.optimizer = GainOptimizer(self.data_acq)
        self.adaptive_optimizer = AdaptiveGainOptimizer(self.data_acq)
        self.evaluator = PerformanceEvaluator()
        self.results = {
            'basic': [],
            'adaptive': [],
            'comparison': []
        }

    def experiment_1_fixed_led_gain_sweep(self):
        """实验1: 固定LED强度,扫描增益值 (论文图4)"""
        print("\n=== 实验1: 固定LED强度,扫描增益值 ===")
        led_duty_cycle = 50
        gains = np.linspace(CameraConfig.GAIN_MIN, CameraConfig.GAIN_MAX, 50)
        background_lights = [20, 50, 100, 150]

        results = []
        for bg_light in background_lights:
            print(f"\n背景光强: {bg_light}")
            capture_results = self.data_acq.simulate_capture_sequence(
                led_duty_cycle, gains, bg_light, noise_std=ExperimentConfig.NOISE_STD,
                roi_strategy=ROIStrategy.CENTER
            )

            gray_values = [r['stats']['mean'] for r in capture_results]
            std_values = [r['stats']['std'] for r in capture_results]

            result = {
                'background_light': bg_light,
                'gains': gains,
                'gray_values': gray_values,
                'std_values': std_values
            }
            results.append(result)

        self.results['experiment_1'] = results
        return results

    def experiment_2_gain_optimization(self):
        """实验2: 增益优化算法"""
        print("\n=== 实验2: 增益优化算法 ===")
        led_duty_cycles = [20, 40, 60, 80]
        background_lights = [30, 80, 130]

        results = []
        for led_dc in led_duty_cycles:
            for bg_light in background_lights:
                print(f"\nLED占空比: {led_dc}%, 背景光: {bg_light}")

                result_basic = self.optimizer.optimize_gain(
                    led_duty_cycle=led_dc,
                    initial_gain=0.0,
                    background_light=bg_light,
                    noise_std=ExperimentConfig.NOISE_STD,
                    roi_strategy=ROIStrategy.CENTER
                )

                result_adaptive = self.adaptive_optimizer.optimize_gain(
                    led_duty_cycle=led_dc,
                    initial_gain=0.0,
                    background_light=bg_light,
                    noise_std=ExperimentConfig.NOISE_STD,
                    roi_strategy=ROIStrategy.CENTER
                )

                eval_basic = self.evaluator.evaluate_optimization_result(
                    result_basic,
                    reference_image=result_basic.get('reference_image'),
                    roi_mask=result_basic.get('roi_mask')
                )
                eval_adaptive = self.evaluator.evaluate_optimization_result(
                    result_adaptive,
                    reference_image=result_adaptive.get('reference_image'),
                    roi_mask=result_adaptive.get('roi_mask')
                )

                comparison = self.evaluator.compare_algorithms(
                    result_basic, result_adaptive,
                    ("Basic", "Adaptive")
                )

                result = {
                    'led_duty_cycle': led_dc,
                    'background_light': bg_light,
                    'basic': {
                        'result': result_basic,
                        'evaluation': eval_basic
                    },
                    'adaptive': {
                        'result': result_adaptive,
                        'evaluation': eval_adaptive
                    },
                    'comparison': comparison
                }

                results.append(result)
                print(f"  基础算法: 增益={result_basic['optimal_gain']:.2f}dB, "
                      f"灰度={result_basic['final_gray']:.2f}, "
                      f"迭代={result_basic['iterations']}")
                print(f"  自适应算法: 增益={result_adaptive['optimal_gain']:.2f}dB, "
                      f"灰度={result_adaptive['final_gray']:.2f}, "
                      f"迭代={result_adaptive['iterations']}")

        self.results['experiment_2'] = results
        return results

    def experiment_3_noise_analysis(self):
        """实验3: 噪声分析 (在优化闭环中注入噪声)"""
        print("\n=== 实验3: 噪声分析 ===")
        led_duty_cycle = 50
        background_light = 50
        noise_levels = [0.5, 1.0, 2.0, 5.0, 10.0]

        results = []
        for noise_std in noise_levels:
            print(f"\n噪声标准差: {noise_std}")
            runs = 10
            run_results = []

            for _ in range(runs):
                result = self.optimizer.optimize_gain(
                    led_duty_cycle=led_duty_cycle,
                    initial_gain=0.0,
                    background_light=background_light,
                    noise_std=noise_std,
                    roi_strategy=ROIStrategy.CENTER
                )

                eval_result = self.evaluator.evaluate_optimization_result(
                    result,
                    reference_image=result.get('reference_image'),
                    roi_mask=result.get('roi_mask')
                )
                eval_result['noise_std'] = noise_std
                run_results.append(eval_result)

            summary = {
                'noise_std': noise_std,
                'gray_error_mean': np.mean([r['gray_error'] for r in run_results]),
                'gray_error_std': np.std([r['gray_error'] for r in run_results]),
                'iterations_mean': np.mean([r['iterations'] for r in run_results]),
                'score_mean': np.mean([r['optimization_score'] for r in run_results]),
                'mse_mean': np.mean([r['image_metrics']['mse'] for r in run_results
                                     if r.get('image_metrics') is not None]),
                'runs': run_results
            }

            results.append(summary)
            print(f"  平均灰度误差: {summary['gray_error_mean']:.2f} ± {summary['gray_error_std']:.2f}")
            print(f"  平均迭代次数: {summary['iterations_mean']:.2f}")

        self.results['experiment_3'] = results
        return results

    def experiment_4_convergence_analysis(self):
        """实验4: 收敛性分析"""
        print("\n=== 实验4: 收敛性分析 ===")
        led_duty_cycle = 50
        background_light = 50
        initial_gains = np.linspace(0, 20, 11)

        results = {
            'basic': [],
            'adaptive': []
        }

        for init_gain in initial_gains:
            print(f"\n初始增益: {init_gain:.1f}dB")

            result_basic = self.optimizer.optimize_gain(
                led_duty_cycle=led_duty_cycle,
                initial_gain=init_gain,
                background_light=background_light,
                noise_std=ExperimentConfig.NOISE_STD,
                roi_strategy=ROIStrategy.CENTER
            )

            result_adaptive = self.adaptive_optimizer.optimize_gain(
                led_duty_cycle=led_duty_cycle,
                initial_gain=init_gain,
                background_light=background_light,
                noise_std=ExperimentConfig.NOISE_STD,
                roi_strategy=ROIStrategy.CENTER
            )

            analysis_basic = self.optimizer.analyze_optimization_curve(result_basic)
            analysis_adaptive = self.adaptive_optimizer.analyze_optimization_curve(result_adaptive)

            results['basic'].append({
                'initial_gain': init_gain,
                'result': result_basic,
                'analysis': analysis_basic
            })

            results['adaptive'].append({
                'initial_gain': init_gain,
                'result': result_adaptive,
                'analysis': analysis_adaptive
            })

            print(f"  基础: {result_basic['iterations']}次迭代, "
                  f"增益范围: {analysis_basic['gain_range']:.2f}dB")
            print(f"  自适应: {result_adaptive['iterations']}次迭代, "
                  f"增益范围: {analysis_adaptive['gain_range']:.2f}dB")

        self.results['experiment_4'] = results
        return results

    def run_all_experiments(self):
        """运行所有实验"""
        print("\n" + "=" * 60)
        print("开始运行所有实验")
        print("=" * 60)

        self.experiment_1_fixed_led_gain_sweep()
        self.experiment_2_gain_optimization()
        self.experiment_3_noise_analysis()
        self.experiment_4_convergence_analysis()

        print("\n" + "=" * 60)
        print("所有实验完成!")
        print("=" * 60)
        return self.results

    def generate_report(self) -> str:
        """生成实验报告"""
        report = []
        report.append("=" * 80)
        report.append("光相机通信模拟增益优化 - 实验报告")
        report.append("=" * 80)

        if 'experiment_1' in self.results:
            report.append("\n## 实验1: 固定LED强度,扫描增益值")
            report.append("-" * 80)
            for result in self.results['experiment_1']:
                report.append(f"\n背景光强: {result['background_light']}")
                max_gray = max(result['gray_values'])
                max_gain = result['gains'][np.argmax(result['gray_values'])]
                report.append(f"  最大灰度值: {max_gray:.2f} (在增益 {max_gain:.2f}dB 时)")

        if 'experiment_2' in self.results:
            report.append("\n## 实验2: 增益优化算法性能比较")
            report.append("-" * 80)
            basic_scores = [r['basic']['evaluation']['optimization_score']
                            for r in self.results['experiment_2']]
            adaptive_scores = [r['adaptive']['evaluation']['optimization_score']
                               for r in self.results['experiment_2']]

            report.append(f"\n基础算法平均得分: {np.mean(basic_scores):.2f}")
            report.append(f"自适应算法平均得分: {np.mean(adaptive_scores):.2f}")
            report.append(f"改进: {((np.mean(adaptive_scores) - np.mean(basic_scores)) / np.mean(basic_scores) * 100):.2f}%")

        if 'experiment_3' in self.results:
            report.append("\n## 实验3: 噪声鲁棒性分析")
            report.append("-" * 80)
            for result in self.results['experiment_3']:
                report.append(f"\n噪声标准差: {result['noise_std']}")
                report.append(f"  平均灰度误差: {result['gray_error_mean']:.2f} ± {result['gray_error_std']:.2f}")
                report.append(f"  平均得分: {result['score_mean']:.2f}")
                report.append(f"  平均MSE: {result['mse_mean']:.2f}")

        if 'experiment_4' in self.results:
            report.append("\n## 实验4: 收敛性分析")
            report.append("-" * 80)
            basic_iters = [r['result']['iterations'] for r in self.results['experiment_4']['basic']]
            adaptive_iters = [r['result']['iterations'] for r in self.results['experiment_4']['adaptive']]

            report.append(f"\n基础算法平均迭代次数: {np.mean(basic_iters):.2f}")
            report.append(f"自适应算法平均迭代次数: {np.mean(adaptive_iters):.2f}")

        report.append("\n" + "=" * 80)
        report.append("报告结束")
        report.append("=" * 80)

        return "\n".join(report)
