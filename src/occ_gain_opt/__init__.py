"""OCC 相机增益优化算法复现包."""

__all__ = [
    # 核心模块（不变）
    "config",
    "demodulation",
    "performance_evaluation",
    "experiment_loader",
    "visualization",
    # 仿真层（向后兼容，内部已重构）
    "data_acquisition",
    "gain_optimization",
    "simulation",
    "realtime",
    "examples",
    # 新增层
    "algorithms",
    "data_sources",
    "hardware",
    "experiments",
]

__version__ = "0.2.0"
