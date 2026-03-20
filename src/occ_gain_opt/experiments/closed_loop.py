"""
闭环实验控制器 (三算法对比)
从 real/adaptive_experiment.py 移植为包内模块。

流程:
  Round 0: 初始参数 → 采集 N 帧 → 解调 → BER_0  (三算法共享)
  Round k: 算法1推荐参数 → 采集 N 帧 → BER_k_A1
           算法2推荐参数 → 采集 N 帧 → BER_k_A2
           算法3推荐参数 → 采集 N 帧 → BER_k_A3
  输出: 三算法 BER 收敛曲线对比图 + 状态 JSON

用法示例（手动模式）:
    from occ_gain_opt.config import CameraParams
    from occ_gain_opt.experiments import ClosedLoopExperiment

    exp = ClosedLoopExperiment(
        rtsp_url="none",
        initial_params=CameraParams(iso=35, exposure_us=27.9),
        label_csv="results/base_data/Mseq_32_original.csv",
        save_dir="exp_data/session_001",
        max_rounds=5,
        n_frames=50,
    )
    exp.run()
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import CameraParams
from ..algorithms import get as algo_get
from ..algorithms.base import AlgorithmBase
from ..data_sources.camera import CameraDataSource
from ..hardware.camera_controller import CameraController
from .advisor import compute_ber, get_roi_brightness


# ── 辅助：获取均值帧亮度 ──────────────────────────────────────────────────────

def _mean_brightness(frames: List[np.ndarray]) -> float:
    return float(np.mean([np.mean(f) for f in frames]))


def _demod_frames(frames: List[np.ndarray], label_csv: str) -> Optional[float]:
    """从多帧中选一帧（均值最稳定的）解调；失败返回 None"""
    if not os.path.isfile(label_csv):
        return None
    import tempfile
    import cv2
    # 选均值最接近中位数的帧
    means = [np.mean(f) for f in frames]
    med = float(np.median(means))
    idx = int(np.argmin([abs(m - med) for m in means]))
    frame = frames[idx]
    # 写临时文件解调
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp_path = tf.name
    try:
        cv2.imwrite(tmp_path, frame)
        # 使用灰度图
        import cv2 as _cv2
        img = _cv2.imread(tmp_path, _cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        ber, _, _ = compute_ber(img, label_csv)
        return ber
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── 主实验类 ──────────────────────────────────────────────────────────────────

class ClosedLoopExperiment:
    """
    三算法闭环对比实验控制器。

    Args:
        rtsp_url:       RTSP 流地址（"none" = 手动/离线模式）
        initial_params: 初始相机参数
        label_csv:      发射序列 CSV 路径（用于 BER 解调）
        save_dir:       结果保存目录
        max_rounds:     最大轮次数
        n_frames:       每轮每算法采集帧数
        alpha:          Matus 自适应学习率
        target_gray:    Matus 目标灰度
        target_brightness: Ma 目标亮度
        ma_strategy:    Ma 控制策略
        iso_min / iso_max: ISO 范围
        camera_mode:    "manual" 或 "hikvision"
        hikvision_url:  Hikvision 相机 URL
        resume:         是否从上次断点恢复
    """

    ALGO_NAMES = ["single_shot", "adaptive_iter", "adaptive_damping"]

    def __init__(
        self,
        rtsp_url: str = "none",
        initial_params: Optional[CameraParams] = None,
        label_csv: str = "results/base_data/Mseq_32_original.csv",
        save_dir: str = "exp_data/session",
        max_rounds: int = 5,
        n_frames: int = 50,
        alpha: float = 0.5,
        target_gray: float = 242.25,
        target_brightness: float = 125.0,
        ma_strategy: str = "exposure_priority",
        iso_min: float = 30.0,
        iso_max: float = 10000.0,
        camera_mode: str = "manual",
        hikvision_url: Optional[str] = None,
        resume: bool = True,
    ) -> None:
        self.initial_params = initial_params or CameraParams(iso=35, exposure_us=27.9)
        self.label_csv = label_csv
        self.save_dir = Path(save_dir)
        self.max_rounds = max_rounds
        self.n_frames = n_frames
        self.resume = resume

        # 实例化三个算法（使用默认参数）
        self._algos: Dict[str, AlgorithmBase] = {
            "single_shot": algo_get("single_shot")(),
            "adaptive_iter": algo_get("adaptive_iter")(),
            "adaptive_damping": algo_get("adaptive_damping")(),
        }
        # 各算法当前参数（独立跟踪）
        self._algo_params: Dict[str, CameraParams] = {
            name: CameraParams(iso=self.initial_params.iso,
                               exposure_us=self.initial_params.exposure_us)
            for name in self.ALGO_NAMES
        }

        # 相机控制器
        self._ctrl = CameraController(
            mode=camera_mode,
            base_url=hikvision_url,
        )

        # RTSP 相机源（可选，手动模式下为 None）
        self._cam_source: Optional[CameraDataSource] = None
        if rtsp_url.lower() != "none":
            self._cam_source = CameraDataSource(
                rtsp_url=rtsp_url,
                initial_params=self.initial_params,
                on_set_params=self._ctrl.set_params,
            )

        # 历史记录
        self._history: Dict[str, List] = {
            name: [] for name in self.ALGO_NAMES
        }
        self._round = 0

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self.save_dir / "state.json"

    # ── 帧采集 ────────────────────────────────────────────────────────────────

    def _capture_frames(
        self, params: CameraParams, algo_name: str, round_idx: int
    ) -> List[np.ndarray]:
        """设置相机参数，采集 n_frames 帧，保存到磁盘并返回帧列表"""
        import cv2

        self._ctrl.set_params(params)
        frames_dir = self.save_dir / f"round_{round_idx:02d}" / algo_name
        frames_dir.mkdir(parents=True, exist_ok=True)

        frames = []
        if self._cam_source is not None:
            self._cam_source.set_params(params)
            for i in range(self.n_frames):
                try:
                    frame = self._cam_source.get_frame()
                    frames.append(frame)
                    cv2.imwrite(str(frames_dir / f"frame_{i:03d}.jpg"), frame)
                except RuntimeError as e:
                    print(f"    采集警告: {e}")
        else:
            # 手动模式：提示用户上传图像
            print(f"\n  [手动模式] 算法 {algo_name}，轮次 {round_idx}")
            print(f"    请将相机设置为: ISO={params.iso:.0f}, 曝光={params.exposure_us:.2f}µs")
            print(f"    拍摄 {self.n_frames} 张图像并上传到: {frames_dir}")
            try:
                input("    上传完成后按 Enter 继续...")
            except EOFError:
                pass
            # 读取已上传的图像
            for fpath in sorted(frames_dir.glob("*.jpg"))[:self.n_frames]:
                img = cv2.imread(str(fpath))
                if img is not None:
                    frames.append(img)

        return frames

    # ── 状态持久化 ────────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        state = {
            "round": self._round,
            "history": {
                name: [
                    {k: (v if not isinstance(v, np.generic) else float(v))
                     for k, v in rec.items()}
                    for rec in recs
                ]
                for name, recs in self._history.items()
            },
        }
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load_state(self) -> bool:
        if not self._state_file.exists():
            return False
        try:
            with open(self._state_file, encoding="utf-8") as f:
                state = json.load(f)
            self._round = state.get("round", 0)
            for name, recs in state.get("history", {}).items():
                if name in self._history:
                    self._history[name] = recs
            print(f"  [恢复] 从轮次 {self._round} 继续实验")
            return True
        except Exception as e:
            print(f"  [警告] 无法恢复状态: {e}")
            return False

    # ── 主循环 ────────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """运行闭环实验，返回各算法历史记录"""
        if self.resume:
            self._load_state()

        print(f"\n{'=' * 60}")
        print("  OCC 闭环实验 — 三算法对比")
        print(f"{'=' * 60}")
        print(f"  初始参数: {self.initial_params}")
        print(f"  最大轮次: {self.max_rounds}")
        print(f"  每轮帧数: {self.n_frames}")

        # Round 0: 共享初始采集
        if self._round == 0:
            print(f"\n[Round 0] 初始采集（三算法共享）...")
            frames0 = self._capture_frames(self.initial_params, "shared", 0)
            brightness0 = _mean_brightness(frames0) if frames0 else 0.0
            ber0 = _demod_frames(frames0, self.label_csv) if frames0 else None
            print(f"  亮度: {brightness0:.1f},  BER: {ber0}")

            for name in self.ALGO_NAMES:
                self._history[name].append({
                    "round": 0, "brightness": brightness0, "ber": ber0,
                    "iso": self.initial_params.iso,
                    "exposure_us": self.initial_params.exposure_us,
                })
            self._round = 1
            self._save_state()

        # Round 1+: 各算法独立
        while self._round <= self.max_rounds:
            print(f"\n[Round {self._round}]")
            for name, algo in self._algos.items():
                params = self._algo_params[name]
                # 从上一轮历史获取亮度
                last = self._history[name][-1] if self._history[name] else {}
                brightness_prev = last.get("brightness", 0.0) or 0.0

                # 计算下一步参数
                ber_prev = last.get("ber")
                next_params = algo.compute_next_params(params, brightness_prev, ber_prev)
                self._algo_params[name] = next_params

                # 采集
                frames = self._capture_frames(next_params, name, self._round)
                brightness = _mean_brightness(frames) if frames else 0.0
                ber = _demod_frames(frames, self.label_csv) if frames else None

                rec = {
                    "round": self._round,
                    "brightness": brightness,
                    "ber": ber,
                    "iso": next_params.iso,
                    "exposure_us": next_params.exposure_us,
                    "gain_db": next_params.gain_db,
                }
                self._history[name].append(rec)
                print(f"  {name}: ISO={next_params.iso:.0f}, "
                      f"exp={next_params.exposure_us:.2f}µs, "
                      f"brightness={brightness:.1f}, BER={ber}")

            self._round += 1
            self._save_state()

        print(f"\n{'=' * 60}")
        print("  实验完成！")
        self._plot_results()
        return self._history

    # ── 可视化 ────────────────────────────────────────────────────────────────

    def _plot_results(self) -> None:
        """生成四面板实时可视化图（BER / 亮度 / ISO / 曝光）"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        colors = {"single_shot": "blue", "adaptive_iter": "orange", "adaptive_damping": "green"}
        labels = {"single_shot": "单次公式", "adaptive_iter": "自适应迭代", "adaptive_damping": "自适应阻尼"}

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("三算法闭环实验对比", fontsize=14)

        metrics = [
            ("ber", axes[0, 0], "BER 收敛曲线", "BER"),
            ("brightness", axes[0, 1], "ROI 亮度变化", "亮度"),
            ("iso", axes[1, 0], "ISO 变化", "ISO"),
            ("exposure_us", axes[1, 1], "曝光时间变化 (µs)", "曝光时间 (µs)"),
        ]
        for key, ax, title, ylabel in metrics:
            for name, history in self._history.items():
                rounds = [r["round"] for r in history]
                vals = [r.get(key) for r in history]
                vals_clean = [v if v is not None else float("nan") for v in vals]
                ax.plot(rounds, vals_clean, marker="o", color=colors[name],
                        label=labels[name], linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("轮次")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = self.save_dir / "results.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"  结果图已保存: {out_path}")
        plt.close(fig)
