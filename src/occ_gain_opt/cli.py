"""
命令行入口
"""

import argparse
import os

from .simulation import ExperimentSimulation
from .visualization import ResultVisualizer
from .examples import main as run_examples


def run_all():
    print("\n初始化仿真环境...")
    sim = ExperimentSimulation()
    viz = ResultVisualizer()

    print("\n开始运行实验...")
    all_results = sim.run_all_experiments()

    print("\n生成实验报告...")
    report = sim.generate_report()

    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)

    report_path = 'results/experiment_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存: {report_path}")
    print("\n" + report)

    print("\n生成可视化图表...")
    if 'experiment_1' in all_results:
        print("\n[1/4] 绘制实验1: 增益扫描...")
        viz.plot_experiment_1(all_results['experiment_1'])
    if 'experiment_2' in all_results:
        print("[2/4] 绘制实验2: 算法比较...")
        viz.plot_experiment_2_comparison(all_results['experiment_2'])
    if 'experiment_3' in all_results:
        print("[3/4] 绘制实验3: 噪声分析...")
        viz.plot_experiment_3_noise(all_results['experiment_3'])
    if 'experiment_4' in all_results:
        print("[4/4] 绘制实验4: 收敛性分析...")
        viz.plot_experiment_4_convergence(all_results['experiment_4'])

    print("\n生成综合报告图表...")
    viz.create_summary_report(all_results)

    if 'experiment_2' in all_results and len(all_results['experiment_2']) > 0:
        print("\n绘制典型优化过程...")
        sample_result = all_results['experiment_2'][0]['adaptive']['result']
        viz.plot_optimization_process(sample_result)


def quick_test():
    from .data_acquisition import DataAcquisition
    from .gain_optimization import GainOptimizer
    from .performance_evaluation import PerformanceEvaluator
    from .config import ExperimentConfig, ROIStrategy

    print("\n运行快速测试...")
    data_acq = DataAcquisition()
    optimizer = GainOptimizer(data_acq)
    evaluator = PerformanceEvaluator()

    led_duty_cycle = 50
    background_light = 50

    print(f"\n测试条件:")
    print(f"  LED占空比: {led_duty_cycle}%")
    print(f"  背景光强: {background_light}")

    print("\n运行增益优化...")
    result = optimizer.optimize_gain(
        led_duty_cycle=led_duty_cycle,
        initial_gain=0.0,
        background_light=background_light,
        noise_std=ExperimentConfig.NOISE_STD,
        roi_strategy=ROIStrategy.CENTER
    )

    evaluation = evaluator.evaluate_optimization_result(
        result,
        reference_image=result.get('reference_image'),
        roi_mask=result.get('roi_mask')
    )

    print("\n结果:")
    print(f"  最优增益: {result['optimal_gain']:.2f} dB")
    print(f"  最终灰度值: {result['final_gray']:.2f}")
    print(f"  目标灰度值: 255")
    print(f"  灰度误差: {evaluation['gray_error']:.2f}")
    print(f"  迭代次数: {result['iterations']}")
    print(f"  是否收敛: {'是' if result['converged'] else '否'}")
    print(f"  优化得分: {evaluation['optimization_score']:.2f}/100")

    print("\n优化过程:")
    print("  迭代 |   增益(dB)  |  灰度值  | 标准差")
    print("-" * 50)
    for state in result['history']:
        print(f"   {state['iteration']:2d}   |  {state['gain']:6.2f}   | "
              f"{state['mean_gray']:6.2f} | {state['std_gray']:5.2f}")

    print("\n✓ 快速测试完成!")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCC 相机增益优化算法复现")
    parser.add_argument("--test", action="store_true", help="运行快速测试")
    parser.add_argument("--examples", action="store_true", help="运行示例")
    parser.add_argument("--example-id", type=str, default=None, help="运行指定示例编号(1-6)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.examples:
        if args.example_id:
            import sys
            sys.argv = [sys.argv[0], args.example_id]
        run_examples()
        return

    if args.test:
        quick_test()
        return

    run_all()


if __name__ == "__main__":
    main()
