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

    # 也测试新算法层
    from .algorithms import list_algorithms, get as algo_get
    from .config import CameraParams
    print(f"\n已注册算法: {list_algorithms()}")
    algo = algo_get("single_shot")()
    params = CameraParams(iso=35, exposure_us=27.9)
    next_p = algo.compute_next_params(params, roi_brightness=110.0)
    print(f"单次公式示例: ISO 35 + 亮度110 → ISO {next_p.iso:.1f} ({next_p.gain_db:+.2f}dB)")
    print("✓ 算法层测试完成!")

    return result


# ── advisor 子命令 ──────────────────────────────────────────────────────────────

def cmd_advisor(args) -> None:
    import numpy as np
    from PIL import Image as PILImage
    from .config import CameraParams
    from .experiments.advisor import run_advisor

    img_pil = PILImage.open(args.image).convert("RGB")
    import cv2
    import numpy as _np
    img = cv2.cvtColor(_np.array(img_pil), cv2.COLOR_RGB2BGR)

    current_params = CameraParams(iso=args.iso, exposure_us=args.exposure)
    run_advisor(
        image=img,
        current_params=current_params,
        label_csv=args.label_csv,
        roi_strategy=args.roi,
        alpha=args.alpha,
        target_gray=args.target_gray,
        target_brightness=args.target_brightness,
        ma_strategy=args.ma_strategy,
        iso_min=args.iso_min,
        iso_max=args.iso_max,
        verbose=True,
    )


# ── experiment 子命令 ──────────────────────────────────────────────────────────

def cmd_experiment(args) -> None:
    from .config import CameraParams
    from .experiments.closed_loop import ClosedLoopExperiment

    initial_params = CameraParams(iso=args.initial_iso, exposure_us=args.initial_exposure)
    exp = ClosedLoopExperiment(
        rtsp_url=args.rtsp_url,
        initial_params=initial_params,
        label_csv=args.label_csv,
        save_dir=args.save_dir,
        max_rounds=args.max_rounds,
        n_frames=args.n_frames,
        alpha=args.alpha,
        target_gray=args.target_gray,
        target_brightness=args.target_brightness,
        ma_strategy=args.ma_strategy,
        iso_min=args.iso_min,
        iso_max=args.iso_max,
        camera_mode=args.camera_mode,
        resume=not args.no_resume,
    )
    exp.run()


# ── 参数解析 ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCC 相机增益优化算法复现",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test", action="store_true", help="运行快速测试")
    parser.add_argument("--examples", action="store_true", help="运行示例")
    parser.add_argument("--example-id", type=str, default=None, help="运行指定示例编号(1-6)")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ── advisor 子命令 ──
    adv = subparsers.add_parser(
        "advisor",
        help="单帧多算法参数建议（给定图像 + 当前参数，输出三算法推荐）",
    )
    adv.add_argument("--image", required=True, help="输入图像路径")
    adv.add_argument("--iso", type=float, required=True, help="当前相机 ISO")
    adv.add_argument("--exposure", type=float, required=True, help="当前曝光时间 (µs)")
    adv.add_argument("--label-csv", default="results/base_data/Mseq_32_original.csv")
    adv.add_argument("--roi", default="sync_based", choices=["sync_based", "auto", "center"])
    adv.add_argument("--alpha", type=float, default=0.5)
    adv.add_argument("--target-gray", type=float, default=242.25)
    adv.add_argument("--target-brightness", type=float, default=125.0)
    adv.add_argument("--ma-strategy", default="exposure_priority",
                     choices=["exposure_priority", "gain_priority"])
    adv.add_argument("--iso-min", type=float, default=30.0)
    adv.add_argument("--iso-max", type=float, default=10000.0)

    # ── experiment 子命令 ──
    exp = subparsers.add_parser(
        "experiment",
        help="三算法闭环对比实验（RTSP 采集 + 参数自动调节）",
    )
    exp.add_argument("--rtsp-url", default="none", help="RTSP 流地址（none=手动模式）")
    exp.add_argument("--initial-iso", type=float, default=35.0)
    exp.add_argument("--initial-exposure", type=float, default=27.9, help="初始曝光时间 (µs)")
    exp.add_argument("--label-csv", default="results/base_data/Mseq_32_original.csv")
    exp.add_argument("--save-dir", default="exp_data/session_001")
    exp.add_argument("--max-rounds", type=int, default=5)
    exp.add_argument("--n-frames", type=int, default=50)
    exp.add_argument("--alpha", type=float, default=0.5)
    exp.add_argument("--target-gray", type=float, default=242.25)
    exp.add_argument("--target-brightness", type=float, default=125.0)
    exp.add_argument("--ma-strategy", default="exposure_priority",
                     choices=["exposure_priority", "gain_priority"])
    exp.add_argument("--iso-min", type=float, default=30.0)
    exp.add_argument("--iso-max", type=float, default=10000.0)
    exp.add_argument("--camera-mode", default="manual", choices=["manual", "hikvision"])
    exp.add_argument("--no-resume", action="store_true", help="不从断点恢复，重新开始")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "advisor":
        cmd_advisor(args)
        return

    if args.command == "experiment":
        cmd_experiment(args)
        return

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
