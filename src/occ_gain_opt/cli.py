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
    """使用统一架构的快速测试"""
    from .data_sources import SimulatedDataSource, create_center_roi_mask, compute_roi_stats
    from .algorithms import list_algorithms, get as algo_get
    from .performance_evaluation import PerformanceEvaluator
    from .config import CameraParams

    print("\n运行快速测试...")
    print("=" * 60)

    # 创建仿真数据源
    data_source = SimulatedDataSource(
        width=640, height=480,
        led_intensity=128, background_light=50, noise_std=2.0
    )
    evaluator = PerformanceEvaluator()

    # 测试所有算法
    algorithms_to_test = ["single_shot", "adaptive_iter", "adaptive_damping"]
    print(f"\n已注册算法: {list_algorithms()}")

    for algo_name in algorithms_to_test:
        try:
            print(f"\n--- 测试 {algo_name} 算法 ---")
            algo = algo_get(algo_name)()

            # 初始参数
            current_params = CameraParams(iso=35, exposure_us=27.9)
            data_source.set_params(current_params)

            # 模拟3轮迭代
            for i in range(3):
                image = data_source.get_frame()
                roi_mask = create_center_roi_mask(image, roi_size=300)
                stats = compute_roi_stats(image, roi_mask)
                brightness = stats['mean']

                print(f"  迭代 {i}: ISO={current_params.iso:.1f}, "
                      f"增益={current_params.gain_db:+.2f}dB, 亮度={brightness:.1f}")

                next_params = algo.compute_next_params(current_params, brightness)
                current_params = next_params
                data_source.set_params(current_params)

            print(f"  ✓ {algo_name} 测试通过")
        except Exception as e:
            print(f"  ✗ {algo_name} 测试失败: {e}")

    print("\n" + "=" * 60)
    print("✓ 快速测试完成!")
    print("=" * 60)


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
