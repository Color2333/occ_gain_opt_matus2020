"""
RTSP 相机数据源
基于 real/comm_capture.py 的 ThreadedCamera，移植为 DataSource 接口。
set_params() 通知外部相机控制器（由 hardware.camera_controller 负责硬件通信）。
"""

import time
from queue import Empty, Queue
from threading import Thread
from typing import Callable, Optional

import cv2
import numpy as np

from ..config import CameraParams
from .base import DataSource


class ThreadedCamera:
    """后台持续读取 RTSP 流，始终持有最新帧（不积压）"""

    def __init__(self, source) -> None:
        self.capture = cv2.VideoCapture(source)
        self.running = True
        self.status = False
        self.frame: Optional[np.ndarray] = None
        self._thread = Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self) -> None:
        while self.running:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(0.01)

    def grab_frame(self) -> Optional[np.ndarray]:
        return self.frame if self.status else None

    def stop(self) -> None:
        self.running = False
        self._thread.join(timeout=3)
        self.capture.release()


class CameraDataSource(DataSource):
    """
    RTSP 实时相机数据源。

    set_params() 调用可选的 on_set_params 回调，
    由调用方（如 hardware.CameraController）负责实际发送 ISAPI 命令。

    Args:
        rtsp_url:      RTSP 流地址（None 表示手动/离线模式）
        initial_params: 初始相机参数
        on_set_params:  回调函数，签名 (CameraParams) -> None，
                        set_params() 时被调用以驱动硬件
        connect_timeout: 连接等待秒数
    """

    def __init__(
        self,
        rtsp_url: Optional[str],
        initial_params: CameraParams,
        on_set_params: Optional[Callable[[CameraParams], None]] = None,
        connect_timeout: float = 5.0,
    ) -> None:
        self._params = initial_params
        self._on_set_params = on_set_params
        self._camera: Optional[ThreadedCamera] = None

        if rtsp_url and rtsp_url.lower() != "none":
            self._camera = ThreadedCamera(rtsp_url)
            deadline = time.time() + connect_timeout
            while time.time() < deadline:
                if self._camera.grab_frame() is not None:
                    break
                time.sleep(0.1)

    @property
    def current_params(self) -> CameraParams:
        return self._params

    def set_params(self, params: CameraParams) -> None:
        self._params = params
        if self._on_set_params is not None:
            self._on_set_params(params)

    def get_frame(self) -> np.ndarray:
        """
        从 RTSP 流抓取最新帧。
        相机未连接时抛出 RuntimeError。
        """
        if self._camera is None:
            raise RuntimeError(
                "相机未连接（rtsp_url=None 或连接超时）。"
                "请使用手动模式或先上传图像帧。"
            )
        frame = self._camera.grab_frame()
        if frame is None:
            raise RuntimeError("无法从相机获取帧（流断开？）")
        return frame

    def stop(self) -> None:
        """停止后台采集线程"""
        if self._camera is not None:
            self._camera.stop()
            self._camera = None
