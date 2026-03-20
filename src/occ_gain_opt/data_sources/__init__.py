"""数据源层：统一仿真、数据集、RTSP 相机的帧获取接口"""

from .base import DataSource
from .roi import (
    create_center_roi_mask,
    create_auto_roi_mask,
    create_sync_based_roi_mask,
    compute_roi_stats,
)
from .simulated import SimulatedDataSource
from .dataset import DatasetDataSource
from .camera import CameraDataSource

__all__ = [
    "DataSource",
    "create_center_roi_mask",
    "create_auto_roi_mask",
    "create_sync_based_roi_mask",
    "compute_roi_stats",
    "SimulatedDataSource",
    "DatasetDataSource",
    "CameraDataSource",
]
