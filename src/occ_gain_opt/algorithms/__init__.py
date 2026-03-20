"""
算法层公共入口
- 接口:  AlgorithmBase (base.py)
- 注册器: REGISTRY / register / get / list_algorithms (registry.py)
- 内置算法: single_shot, adaptive_iter, adaptive_damping, ber_explore
  各算法文件只含算法本体，注册在此处统一完成。

新增算法示例:
    # my_algo.py — 只写算法类，不 import 注册器
    class MyAlgo(AlgorithmBase):
        name = "my_algo"
        ...

    # 在本文件末尾添加一行:
    register(MyAlgo)
"""

from .base import AlgorithmBase
from .registry import REGISTRY, register, get, list_algorithms

from .single_shot import SingleShotAlgorithm
from .adaptive_iter import AdaptiveIterAlgorithm
from .adaptive_damping import AdaptiveDampingAlgorithm
from .ber_explore import BerExploreAlgorithm

# ── 注册所有内置算法 ───────────────────────────────────────────────────────────
register(SingleShotAlgorithm)
register(AdaptiveIterAlgorithm)
register(AdaptiveDampingAlgorithm)
register(BerExploreAlgorithm)

__all__ = [
    "AlgorithmBase",
    "REGISTRY",
    "register",
    "get",
    "list_algorithms",
    "SingleShotAlgorithm",
    "AdaptiveIterAlgorithm",
    "AdaptiveDampingAlgorithm",
    "BerExploreAlgorithm",
]
