"""
算法注册中心
"""

from typing import Dict, List, Type

from .base import AlgorithmBase

REGISTRY: Dict[str, Type[AlgorithmBase]] = {}


def register(cls: Type[AlgorithmBase]) -> Type[AlgorithmBase]:
    """将算法类注册到 REGISTRY（可作装饰器，也可直接调用）"""
    if not cls.name:
        raise ValueError(f"算法类 {cls.__name__} 未设置 name 属性")
    REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> Type[AlgorithmBase]:
    """按名称获取算法类；未找到时抛出 KeyError"""
    if name not in REGISTRY:
        raise KeyError(f"算法 '{name}' 未注册。已注册: {list(REGISTRY.keys())}")
    return REGISTRY[name]


def list_algorithms() -> List[str]:
    """返回所有已注册算法名称列表"""
    return list(REGISTRY.keys())
