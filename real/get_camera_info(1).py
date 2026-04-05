import requests
import base64

class CameraExposureController:
    def __init__(self, ip, port, username, password, channel=0):
        """初始化摄像头曝光控制器"""
        self.base_url = f"http://{ip}:{port}/cgi-bin"
        self.channel = channel
        self.auth = base64.b64encode(f"{username}:{password}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {self.auth}",
            "User-Agent": "CameraExposureController"
        }
    
    def set_shutter(self, shutter_value):
        """
        设置曝光时间（快门速度）
        
        Args:
            shutter_value: 快门值 (0-19)
                0：自动曝光
                1：1/25S (PAL) 或 1/30S (NTSC)
                2：1/50S (PAL) 或 1/60S (NTSC)
                ...
                19：1/10000S (PAL) 或 1/12000S (NTSC)
        """
        url = f"{self.base_url}/param.cgi?action=update&group=DIS&channel={self.channel}&DIS.shutter={shutter_value}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            if self._check_response(response):
                print(f"曝光时间设置成功: {self._shutter_to_text(shutter_value)}")
                return True
            return False
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return False
    
    def get_shutter(self):
        """获取当前曝光时间设置"""
        url = f"{self.base_url}/param.cgi?action=list&group=DIS&channel={self.channel}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            if self._check_response(response):
                lines = response.text.split('\n')
                for line in lines:
                    if 'root.DIS.shutter=' in line:
                        shutter_value = int(line.split('=')[1])
                        return shutter_value
            return None
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return None
    
    def auto_exposure(self):
        """设置为自动曝光"""
        return self.set_shutter(0)
    
    def set_specific_shutter(self, denominator):
        """
        设置具体的快门分母值（如1/1000秒则输入1000）
        
        Args:
            denominator: 快门分母值
        """
        shutter_map = {
            25: 1, 30: 1,
            50: 2, 60: 2,
            75: 3, 90: 3,
            100: 4, 120: 4,
            150: 5, 180: 5,
            200: 6, 240: 6,
            250: 7, 300: 7,
            300: 8, 450: 8,
            400: 9, 600: 9,
            500: 10, 750: 10,
            750: 11, 900: 11,
            1000: 12, 1200: 12,
            1500: 13, 1800: 13,
            2000: 14, 2400: 14,
            3000: 15, 3600: 15,
            4000: 16, 4800: 16,
            5000: 17, 6000: 17,
            7500: 18, 9000: 18,
            10000: 19, 12000: 19
        }
        
        if denominator in shutter_map:
            return self.set_shutter(shutter_map[denominator])
        else:
            print(f"不支持的快门值: 1/{denominator}S")
            return False
    
    def _check_response(self, response):
        """检查响应是否成功"""
        return "root.ERR.no=0" in response.text
    
    def _shutter_to_text(self, shutter_value):
        """将快门值转换为可读文本"""
        shutter_texts = [
            "自动曝光", "1/25S(PAL)或1/30S(NTSC)", "1/50S(PAL)或1/60S(NTSC)",
            "1/75S(PAL)或1/90S(NTSC)", "1/100S(PAL)或1/120S(NTSC)",
            "1/150S(PAL)或1/180S(NTSC)", "1/200S(PAL)或1/240S(NTSC)",
            "1/250S(PAL)或1/300S(NTSC)", "1/300S(PAL)或1/450S(NTSC)",
            "1/400S(PAL)或1/600S(NTSC)", "1/500S(PAL)或1/750S(NTSC)",
            "1/750S(PAL)或1/900S(NTSC)", "1/1000S(PAL)或1/1200S(NTSC)",
            "1/1500S(PAL)或1/1800S(NTSC)", "1/2000S(PAL)或1/2400S(NTSC)",
            "1/3000S(PAL)或1/3600S(NTSC)", "1/4000S(PAL)或1/4800S(NTSC)",
            "1/5000S(PAL)或1/6000S(NTSC)", "1/7500S(PAL)或1/9000S(NTSC)",
            "1/10000S(PAL)或1/12000S(NTSC)"
        ]
        return shutter_texts[shutter_value] if 0 <= shutter_value < len(shutter_texts) else f"未知值: {shutter_value}"


# 使用示例
if __name__ == "__main__":
    # 初始化
    camera = CameraExposureController(
        ip="192.168.1.160",  # 摄像头IP
        port=80,            # 摄像头CGI端口80
        username="admin",   # 用户名admin
        password="12345" # 密码12345
    )
    
    # 获取当前曝光设置
    current = camera.get_shutter()
    if current is not None:
        print(f"当前曝光时间: {camera._shutter_to_text(current)}")
    
    # 设置为自动曝光
    #camera.auto_exposure()
    
    # 设置为1/1000秒
    camera.set_specific_shutter(5000)  # 1/1000秒
    
    # 或者直接设置快门值
    #camera.set_shutter(12)  # 1/1000S