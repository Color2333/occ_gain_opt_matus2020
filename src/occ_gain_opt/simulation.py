"""
仿真实验模块
使用统一架构: algorithms/ + data_sources/
"""

from typing import Dict, List

import numpy as np

from .config import ExperimentConfig, CameraConfig, ROIStrategy, CameraParams
from .data_sources import SimulatedDataSource, create_center_roi_mask, compute_roi_stats
from .algorithms import get as algo_get
from .performance_evaluation import PerformanceEvaluator


class ExperimentSimulation:
    """实验仿真类 - 使用统一架构"""

    def __init__(self):
        self.data_source = SimulatedDataSource()
        self.evaluator = PerformanceEvaluator()
        self.results: Dict = {
            'basic': [],
            'adaptive': [],
            'comparison': []
        }

    def _run_optimization(
        self,
        algo_name: str,
        led_intensity: float,
        initial_gain_db: float,
        background_light: float,
        noise_std: float,
        max_iterations: int = 20,
    ) -> Dict:
        """使用新算法架构运行优化"""
        # 重置数据源
        self.data_source = SimulatedDataSource(
            width=CameraConfig.IMAGE_WIDTH,
            height=CameraConfig.IMAGE_HEIGHT,
            noise_std=noise_std,
        )
        self.data_source.led_intensity = led_intensity
        self.data_source.background_light = background_light

        # 初始化算法
        algo = algo_get(algo_name)()

        # 设置初始参数
        current_params = CameraParams.from_gain_db(initial_gain_db, 5000.0)
        self.data_source.set_params(current_params)

        history = []
        converged = False
        iteration = 0

        while iteration < max_iterations and not converged:
            # 采集图像
            image = self.data_source.get_frame()

            # 选择ROI并计算亮度
            roi_mask = create_center_roi_mask(image, roi_size=300)
            stats = compute_roi_stats(image, roi_mask)
            brightness = stats['mean']

            # 记录状态
            state = {
                'iteration': iteration,
                'gain': current_params.gain_db,
                'mean_gray': brightness,
                'std_gray': stats['std'],
            }
            history.append(state)

            # 检查收敛
            if algo.is_converged():
                converged = True
                break

            # 计算下一步参数
            next_params = algo.compute_next_params(current_params, brightness)

            # 检查参数是否变化
            if abs(next_params.gain_db - current_params.gain_db) < 0.01:
                converged = True
                break

            current_params = next_params
            self.data_source.set_params(current_params)
            iteration += 1

        # 最终图像
        final_image = self.data_source.get_frame()
        roi_mask = create_center_roi_mask(final_image, roi_size=300)
        final_stats = compute_roi_stats(final_image, roi_mask)

        return {
            'optimal_gain': current_params.gain_db,
            'final_gray': final_stats['mean'],
            'iterations': iteration + 1,
            'converged': converged,
            'history': history,
            'final_image': final_image,
            'roi_mask': roi_mask,
            'final_stats': final_stats,
        }

    def experiment_1_fixed_led_gain_sweep(self):
        """实验1: 固定LED强度,扫描增益值 (论文图4)"""
        print("\n=== 实验1: 固定LED强度,扫描增益值 ===")
        led_duty_cycle = 50
        gains = np.linspace(CameraConfig.GAIN_MIN, CameraConfig.GAIN_MAX, 50)
        background_lights = [20, 50, 100, 150]

        led_intensity = (led_duty_cycle / 100.0) * 255

        results = []
        for bg_light in background_lights:
            print(f"\n背景光强: {bg_light}")

            # 使用SimulatedDataSource扫描增益
            data_source = SimulatedDataSource(
                width=CameraConfig.IMAGE_WIDTH,
                height=CameraConfig.IMAGE_HEIGHT,
                noise_std=ExperimentConfig.NOISE_STD,
            )
            data_source.led_intensity = led_intensity
            data_source.background_light = bg_light

            gray_values = []
            std_values = []

            for gain_db in gains:
                params = CameraParams.from_gain_db(gain_db, 5000.0)
                data_source.set_params(params)
                image = data_source.get_frame()
                roi_mask = create_center_roi_mask(image, roi_size=300)
                stats = compute_roi_stats(image, roi_mask)
                gray_values.append(stats['mean'])
                std_values.append(stats['std'])

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
        """实验2: 增益优化算法对比"""
        print("\n=== 实验2: 增益优化算法对比 ===")
        led_duty_cycles = [20, 40, 60, 80]
        background_lights = [30, 80, 130]

        results = []
        for led_dc in led_duty_cycles:
            for bg_light in background_lights:
                print(f"\nLED占空比: {led_dc}%, 背景光: {bg_light}")
                led_intensity = (led_dc / 100.0) * 255

                result_basic = self._run_optimization(
                    "single_shot",
                    led_intensity,
                    0.0,
                    bg_light,
                    ExperimentConfig.NOISE_STD,
                )

                result_adaptive = self._run_optimization(
                    "adaptive_iter",
                    led_intensity,
                    0.0,
                    bg_light,
                    ExperimentConfig.NOISE_STD,
                )

                eval_basic = self.evaluator.evaluate_optimization_result(
                    result_basic,
                    reference_image=result_basic.get('final_image'),
                    roi_mask=result_basic.get('roi_mask')
                )
                eval_adaptive = self.evaluator.evaluate_optimization_result(
                    result_adaptive,
                    reference_image=result_adaptive.get('final_image'),
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
        """实验3: 噪声分析"""
        print("\n=== 实验3: 噪声分析 ===")
        led_duty_cycle = 50
        background_light = 50
        noise_levels = [0.5, 1.0, 2.0, 5.0, 10.0]
        led_intensity = (led_duty_cycle / 100.0) * 255

        results = []
        for noise_std in noise_levels:
            print(f"\n噪声标准差: {noise_std}")
            runs = 10
            run_results = []

            for _ in range(runs):
                result = self._run_optimization(
                    "single_shot",
                    led_intensity,
                    0.0,
                    background_light,
                    noise_std,
                )

                eval_result = self.evaluator.evaluate_optimization_result(
                    result,
                    reference_image=result.get('final_image'),
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
        led_intensity = (led_duty_cycle / 100.0) * 255

        results = {
            'basic': [],
            'adaptive': []
        }

        for init_gain in initial_gains:
            print(f"\n初始增益: {init_gain:.1f}dB")

            result_basic = self._run_optimization(
                "single_shot",
                led_intensity,
                init_gain,
                background_light,
                ExperimentConfig.NOISE_STD,
            )

            result_adaptive = self._run_optimization(
                "adaptive_iter",
                led_intensity,
                init_gain,
                background_light,
                ExperimentConfig.NOISE_STD,
            )

            # 分析收敛曲线
            def _analyze(history):
                if len(history) > 1:
                    gains = [h['gain'] for h in history]
                    gain_range = max(gains) - min(gains)
                    gain_changes = [abs(gains[i + 1] - gains[i]) for i in range(len(gains) - 1)]
                    avg_change = float(np.mean(gain_changes)) if gain_changes else 0.0
                else:
                    gain_range = 0.0
                    avg_change = 0.0
                return {'gain_range': gain_range, 'average_gain_change': avg_change}

            analysis_basic = _analyze(result_basic['history'])
            analysis_adaptive = _analyze(result_adaptive['history'])

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
