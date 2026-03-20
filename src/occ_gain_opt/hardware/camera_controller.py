"""
相机控制器
支持两种模式：
  manual    — 终端提示用户手动设置相机参数（实验室手工调节）
  hikvision — 通过 Hikvision ISAPI HTTP 接口自动设置

从 real/adaptive_experiment.py 拆分。
"""

import time
from typing import Optional

from ..config import CameraParams


class CameraController:
    """
    相机参数控制器。

    Args:
        mode:     "manual" 或 "hikvision"
        base_url: Hikvision 相机基础 URL（仅 hikvision 模式使用），
                  如 "http://192.168.1.19"
        username: ISAPI 认证用户名（默认 "admin"）
        password: ISAPI 认证密码
    """

    def __init__(
        self,
        mode: str = "manual",
        base_url: Optional[str] = None,
        username: str = "admin",
        password: str = "",
    ) -> None:
        if mode not in ("manual", "hikvision"):
            raise ValueError(f"mode 必须是 'manual' 或 'hikvision'，得到: {mode!r}")
        self.mode = mode
        self.base_url = base_url
        self.username = username
        self.password = password

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def set_params(self, params: CameraParams) -> None:
        """
        设置相机参数。
        manual 模式：打印提示，等待用户手动操作。
        hikvision 模式：发送 ISAPI 请求。
        """
        if self.mode == "manual":
            self._manual_set(params)
        else:
            self._hikvision_set(params)

    # ── Manual 模式 ───────────────────────────────────────────────────────────

    def _manual_set(self, params: CameraParams) -> None:
        exp_us = params.exposure_us
        iso = params.iso
        print(f"\n  [相机手动设置]")
        print(f"    请将相机 ISO 设置为: {iso:.0f}")
        print(f"    请将曝光时间设置为: {exp_us:.2f} µs  ({exp_us * 1e-3:.4f} ms)")
        try:
            input("    设置完成后按 Enter 继续...")
        except EOFError:
            time.sleep(1)

    # ── Hikvision ISAPI 模式 ──────────────────────────────────────────────────

    def _hikvision_set(self, params: CameraParams) -> None:
        """
        通过 Hikvision ISAPI 设置 ISO 和曝光时间。

        TODO: 在确认相机实际支持的 XML 字段名后填写。
        参考步骤：
          1. GET http://{ip}/ISAPI/Image/channels/1/ISP
             查看响应 XML 中 ISO 和曝光时间字段名
          2. 对应修改下方 xml_body 的字段名称
          3. PUT 到相同 URL 更新参数

        已确认字段（示例，实际相机可能不同）：
          ISO:          <isoSensitivity>...</isoSensitivity>
          曝光时间:     <exposureTime>...</exposureTime>  （单位通常为 µs 或 1/10000 s，需确认）
        """
        if not self.base_url:
            raise RuntimeError("hikvision 模式需要设置 base_url")

        try:
            import requests
        except ImportError:
            raise ImportError("hikvision 模式需要安装 requests: pip install requests")

        url = f"{self.base_url}/ISAPI/Image/channels/1/ISP"
        auth = (self.username, self.password)

        # ── 先 GET 当前配置（可用于调试确认字段名）──────────────────────────
        try:
            resp = requests.get(url, auth=auth, timeout=5)
            resp.raise_for_status()
        except Exception as e:
            print(f"  [ISAPI GET] 失败: {e}")

        # ── 构造 PUT 请求（字段名待确认）────────────────────────────────────
        # 注意：曝光时间单位需根据实际相机规格确认（µs / 1/n 秒 / 直接数值）
        xml_body = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<ISPMode>"
            f"<isoSensitivity>{int(params.iso)}</isoSensitivity>"
            f"<exposureTime>{int(params.exposure_us)}</exposureTime>"
            "</ISPMode>"
        )
        headers = {"Content-Type": "application/xml"}
        try:
            resp = requests.put(url, data=xml_body, headers=headers, auth=auth, timeout=5)
            resp.raise_for_status()
            print(f"  [ISAPI] 已设置 ISO={params.iso:.0f}, exp={params.exposure_us:.2f}µs")
        except Exception as e:
            print(f"  [ISAPI PUT] 失败: {e}")
