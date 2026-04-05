"""
Hikvision ISAPI 相机参数读取和设置
支持: 获取当前参数、设置增益
"""

import requests
from requests.auth import HTTPDigestAuth
from typing import Optional, Dict, Any, Tuple
import re


class HikvisionCamera:
    """海康摄像机 ISAPI 接口"""

    def __init__(self, ip: str, username: str = "admin", password: str = "", port: int = 80):
        self.ip = ip
        self.username = username
        self.password = password
        self.port = port
        self.base_url = f"http://{ip}:{port}"

    def _get(self, path: str) -> Optional[str]:
        """发送 GET 请求"""
        url = f"{self.base_url}{path}"
        try:
            resp = requests.get(url, auth=HTTPDigestAuth(self.username, self.password), timeout=5)
            if resp.status_code == 200:
                return resp.text
        except Exception as e:
            print(f"GET {path} 失败: {e}")
        return None

    def _put(self, path: str, xml_data: str) -> Tuple[bool, str]:
        """发送 PUT 请求"""
        url = f"{self.base_url}{path}"
        try:
            headers = {"Content-Type": "application/xml"}
            resp = requests.put(url, data=xml_data, auth=HTTPDigestAuth(self.username, self.password),
                              headers=headers, timeout=5)
            return (resp.status_code == 200), resp.text[:500]
        except Exception as e:
            return (False, str(e))

    def get_current_params(self) -> Optional[Dict[str, Any]]:
        """获取当前相机参数"""
        xml = self._get("/ISAPI/Image/channels/1")
        if not xml:
            return None

        params = {}

        # 解析关键参数
        patterns = {
            'shutter_level': r'<ShutterLevel>([^<]+)</ShutterLevel>',
            'gain_level': r'<GainLevel>([^<]+)</GainLevel>',
            'gain_limit': r'<GainLimit>([^<]+)</GainLimit>',
            'exposure_type': r'<ExposureType>([^<]+)</ExposureType>',
            'brightness': r'<brightnessLevel>([^<]+)</brightnessLevel>',
            'contrast': r'<contrastLevel>([^<]+)</contrastLevel>',
            'saturation': r'<saturationLevel>([^<]+)</saturationLevel>',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, xml)
            if match:
                params[key] = match.group(1)

        return params

    def get_exposure_us(self) -> Optional[float]:
        """获取曝光时间（微秒）"""
        params = self.get_current_params()
        if params and 'shutter_level' in params:
            shutter = params['shutter_level']
            match = re.match(r'1/(\d+)', shutter)
            if match:
                return 1000000.0 / int(match.group(1))
        return None

    def get_gain_level(self) -> Optional[int]:
        """获取增益级别 (0-100)"""
        params = self.get_current_params()
        if params and 'gain_level' in params:
            try:
                return int(params['gain_level'])
            except:
                pass
        return None

    def get_gain_limit(self) -> Optional[int]:
        """获取增益上限 (0-100)"""
        params = self.get_current_params()
        if params and 'gain_limit' in params:
            try:
                return int(params['gain_limit'])
            except:
                pass
        return None

    def set_gain(self, gain_value: int) -> bool:
        """
        设置增益级别 (0-100)

        Args:
            gain_value: 增益级别，范围通常是 0-100

        Returns:
            是否设置成功
        """
        xml = self._get("/ISAPI/Image/channels/1")
        if not xml:
            print("获取当前配置失败")
            return False

        # 验证增益范围
        if not 0 <= gain_value <= 100:
            print(f"增益值 {gain_value} 超出范围 0-100")
            return False

        # 替换 GainLevel
        new_xml = re.sub(
            r'<GainLevel>([^<]+)</GainLevel>',
            f'<GainLevel>{gain_value}</GainLevel>',
            xml
        )

        success, response = self._put("/ISAPI/Image/channels/1", new_xml)

        if success:
            print(f"✓ 增益设置成功: {gain_value}")
            return True
        else:
            print(f"✗ 增益设置失败: {response}")
            return False

    def set_gain_limit(self, gain_limit: int) -> bool:
        """
        设置增益上限 (0-100)

        Args:
            gain_limit: 增益上限，范围通常是 0-100

        Returns:
            是否设置成功
        """
        xml = self._get("/ISAPI/Image/channels/1")
        if not xml:
            print("获取当前配置失败")
            return False

        if not 0 <= gain_limit <= 100:
            print(f"增益上限 {gain_limit} 超出范围 0-100")
            return False

        new_xml = re.sub(
            r'<GainLimit>([^<]+)</GainLimit>',
            f'<GainLimit>{gain_limit}</GainLimit>',
            xml
        )

        success, response = self._put("/ISAPI/Image/channels/1", new_xml)

        if success:
            print(f"✓ 增益上限设置成功: {gain_limit}")
            return True
        else:
            print(f"✗ 增益上限设置失败: {response}")
            return False

    def get_readable_params(self) -> str:
        """获取易读的参数信息"""
        params = self.get_current_params()
        if not params:
            return "无法获取参数"

        lines = []
        lines.append("=" * 50)
        lines.append("  相机当前参数")
        lines.append("=" * 50)

        if 'shutter_level' in params:
            shutter = params['shutter_level']
            match = re.match(r'1/(\d+)', shutter)
            if match:
                us = 1000000 / int(match.group(1))
                lines.append(f"  快门时间: {shutter} = {us:.0f} µs")
            else:
                lines.append(f"  快门时间: {shutter}")

        if 'gain_level' in params:
            lines.append(f"  增益级别: {params['gain_level']}")

        if 'gain_limit' in params:
            lines.append(f"  增益上限: {params['gain_limit']}")

        if 'exposure_type' in params:
            lines.append(f"  曝光模式: {params['exposure_type']}")

        if 'brightness' in params:
            lines.append(f"  亮度: {params['brightness']}")

        if 'contrast' in params:
            lines.append(f"  对比度: {params['contrast']}")

        if 'saturation' in params:
            lines.append(f"  饱和度: {params['saturation']}")

        lines.append("=" * 50)
        return "\n".join(lines)


def main():
    camera = HikvisionCamera(
        ip="192.168.1.19",
        username="admin",
        password="abcd1234"
    )

    print("读取当前参数...")
    print(camera.get_readable_params())

    print("\n测试增益设置...")
    camera.set_gain(30)
    camera.set_gain(60)
    camera.set_gain(0)

    print("\n读取设置后的参数:")
    print(camera.get_readable_params())

    print("\n参数含义说明:")
    print("  快门时间 (Shutter): 曝光时间越短，图像越暗")
    print("  增益级别 (Gain): 图像信号放大，越大越亮但噪点增多")
    print("  曝光模式: ShutterFirst 表示快门优先")


if __name__ == "__main__":
    main()
