import cv2
import requests
import base64
import os
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from threading import Thread

class DualCameraExposureController:
    def __init__(self, ip, cgi_port=80, username="admin", password="12345", rtsp_port=8554):
        """
        初始化双目相机曝光控制器
        """
        self.ip = ip
        self.cgi_port = cgi_port
        self.rtsp_port = rtsp_port
        self.username = username
        self.password = password
        
        # CGI控制接口URL
        self.base_url = f"http://{ip}:{cgi_port}/cgi-bin"
        
        # 认证信息
        self.auth = base64.b64encode(f"{username}:{password}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {self.auth}",
            "User-Agent": "CameraExposureController"
        }
        
        # 控制端口映射
        self.control_channels = {
            'left': 0,
            'right': 1,
        }
        
        # 视频输出通道
        self.video_channels = {
            'stitched': 0,
        }
        
        # 曝光时间映射表（值到曝光时间字符串的映射）
        self.exposure_time_map = self._create_exposure_time_map()
        
        # ISO映射表
        self.iso_map = self._create_iso_map()
    
    def _create_exposure_time_map(self):
        """创建曝光值到曝光时间字符串的映射"""
        return {
            0: "auto",      # 自动曝光
            1: "1_25s",
            2: "1_50s",
            3: "1_75s",
            4: "1_100s",
            5: "1_150s",
            6: "1_200s",
            7: "1_250s",
            8: "1_300s",
            9: "1_400s",
            10: "1_500s",
            11: "1_750s",
            12: "1_1000s",
            13: "1_1500s",
            14: "1_2000s",
            15: "1_3000s",
            16: "1_4000s",
            17: "1_5000s",
            18: "1_7500s",
            19: "1_10000s",
            20: "1_50000s",
            21: "1_100000s"
        }
    
    def _create_iso_map(self):
        """创建ISO值到ISO数值的映射"""
        return {
            0: "auto",      # 自动ISO
            1: "100",
            2: "200",
            4: "400",
            8: "800",
            16: "1600",
            32: "3200",
            64: "6400",
            128: "12800",
            160: "16000",
            192: "19200",
            194: "25600",
            196: "32000",
            198: "38400",
            202: "51200",
            210: "76800",
            218: "102400",
            234: "153600",
            250: "204800"
        }
    
    def get_exposure_time_string(self, shutter_value):
        """
        根据快门值获取曝光时间字符串
        
        Args:
            shutter_value: 快门值 (0-21)
            
        Returns:
            str: 曝光时间字符串，如 "1_100s"、"auto" 等
        """
        if shutter_value in self.exposure_time_map:
            return self.exposure_time_map[shutter_value]
        else:
            return f"unknown_{shutter_value}"
    
    def get_iso_string(self, iso_value):
        """
        根据ISO值获取ISO字符串
        
        Args:
            iso_value: ISO值 (0-18)
            
        Returns:
            str: ISO字符串，如 "100"、"auto" 等
        """
        if iso_value in self.iso_map:
            return self.iso_map[iso_value]
        else:
            return f"unknown_{iso_value}"
    
    def _iso_to_text(self, iso_value):
        """将ISO值转换为可读文本"""
        if iso_value == 0:
            return "自动ISO"
        elif iso_value in self.iso_map:
            iso_str = self.iso_map[iso_value]
            if iso_str != "auto":
                return f"ISO {iso_str}"
            else:
                return "自动ISO"
        else:
            return f"未知ISO值: {iso_value}"
    
    def set_shutter_for_dual(self, shutter_value):
        """
        设置左右双目的曝光时间
        """
        if not 0 <= shutter_value <= 21:
            print(f"❌ 无效的快门值: {shutter_value} (支持范围: 0-21)")
            return False
            
        success = True
        for camera_name in ['left', 'right']:
            control_port = self.control_channels[camera_name]
            if not self._set_single_camera_shutter(control_port, shutter_value):
                success = False
                print(f"❌ {camera_name} 快门设置失败")
        return success
    
    def set_iso_for_dual(self, iso_value):
        """
        设置左右双目的ISO值
        同时设置最小ISO值（isoMin），与ISO值相同
        """
        if not 0 <= iso_value <= 255:
            print(f"❌ 无效的ISO值: {iso_value} (支持范围: 0-255)")
            return False
            
        success = True
        for camera_name in ['left', 'right']:
            control_port = self.control_channels[camera_name]
            if not self._set_single_camera_iso(control_port, iso_value):
                success = False
                print(f"❌ {camera_name} ISO设置失败")
        return success
    
    def _set_single_camera_shutter(self, control_port, shutter_value):
        """
        内部方法：通过控制端口设置单个相机的曝光时间
        """
        url = f"{self.base_url}/param.cgi?action=update&group=DIS&channel={control_port}&DIS.shutter={shutter_value}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            if self._check_response(response):
                exposure_time_str = self.get_exposure_time_string(shutter_value)
                print(f"  ✅ 控制端口{control_port}: {self._shutter_to_text(shutter_value)}")
                return True
            else:
                print(f"  ❌ 控制端口{control_port}: 快门设置失败")
                return False
        except requests.exceptions.RequestException as e:
            print(f"  ❌ 控制端口{control_port}: 快门请求失败 - {e}")
            return False
    
    def _set_single_camera_iso(self, control_port, iso_value):
        """
        内部方法：通过控制端口设置单个相机的ISO值
        同时设置最小ISO值（isoMin），与ISO值相同
        """
        # 首先设置ISO值
        url = f"{self.base_url}/param.cgi?action=update&group=DIS&channel={control_port}&DIS.iso={iso_value}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            if not self._check_response(response):
                print(f"  ❌ 控制端口{control_port}: ISO设置失败")
                return False
        except requests.exceptions.RequestException as e:
            print(f"  ❌ 控制端口{control_port}: ISO请求失败 - {e}")
            return False
        
        # 设置最小ISO值（isoMin），与ISO值相同
        url_min = f"{self.base_url}/param.cgi?action=update&group=DIS&channel={control_port}&DIS.isoMin={iso_value}"
        
        try:
            response_min = requests.get(url_min, headers=self.headers, timeout=5)
            if self._check_response(response_min):
                iso_str = self.get_iso_string(iso_value)
                print(f"  ✅ 控制端口{control_port}: {self._iso_to_text(iso_value)} (最小ISO: {self._iso_to_text(iso_value)})")
                return True
            else:
                print(f"  ❌ 控制端口{control_port}: 最小ISO设置失败")
                return False
        except requests.exceptions.RequestException as e:
            print(f"  ❌ 控制端口{control_port}: 最小ISO请求失败 - {e}")
            return False
    
    def _check_response(self, response):
        """检查响应是否成功"""
        return "root.ERR.no=0" in response.text
    
    def _shutter_to_text(self, shutter_value):
        """将快门值转换为可读文本"""
        shutter_texts = [
            "自动曝光",
            "1/25秒", "1/50秒", "1/75秒", "1/100秒", 
            "1/150秒", "1/200秒", "1/250秒", "1/300秒", 
            "1/400秒", "1/500秒", "1/750秒", "1/1000秒", 
            "1/1500秒", "1/2000秒", "1/3000秒", "1/4000秒", 
            "1/5000秒", "1/7500秒", "1/10000秒",
            "1/50000秒", "1/100000秒"
        ]
        return shutter_texts[shutter_value] if 0 <= shutter_value < len(shutter_texts) else f"未知值: {shutter_value}"


class ThreadedCamera:
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.status = False
        self.frame = None
 
    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
 
    def grab_frame(self):
        if self.status:
            return self.frame
        return None
    
    def release(self):
        self.capture.release()


class ExposureTestCapture:
    def __init__(self, camera_ip, base_save_folder="exposure_test"):
        """
        初始化曝光测试捕获器
        
        Args:
            camera_ip: 相机IP地址
            base_save_folder: 基础保存文件夹
        """
        self.camera_ip = camera_ip
        self.base_save_folder = base_save_folder
        
        # 初始化相机控制器
        self.camera_controller = DualCameraExposureController(
            ip=camera_ip,
            cgi_port=80,
            username="admin",
            password="12345",
            rtsp_port=8554
        )
        
        # RTSP URL
        self.rtsp_url = f"rtsp://{camera_ip}:8554/0"
        
        # 初始化视频流捕获器
        self.streamer = ThreadedCamera(self.rtsp_url)
        
        # 创建按次序编号的子文件夹
        self.save_folder = self._create_sequential_folder()
        
        print(f"📁 创建保存文件夹: {self.save_folder}")
    
    def _create_sequential_folder(self):
        """
        创建按次序编号的子文件夹
        
        Returns:
            str: 新创建的子文件夹路径
        """
        # 确保基础文件夹存在
        if not os.path.exists(self.base_save_folder):
            os.makedirs(self.base_save_folder)
        
        # 查找已存在的序号文件夹
        existing_folders = []
        for item in os.listdir(self.base_save_folder):
            item_path = os.path.join(self.base_save_folder, item)
            if os.path.isdir(item_path) and item.startswith("test_"):
                try:
                    # 提取序号，如 "test_001" -> 1
                    num = int(item.split("_")[1])
                    existing_folders.append(num)
                except (ValueError, IndexError):
                    continue
        
        # 确定下一个序号
        if existing_folders:
            next_num = max(existing_folders) + 1
        else:
            next_num = 1
        
        # 创建新的子文件夹（格式：test_001, test_002, ...）
        folder_name = f"test_{next_num:03d}"
        new_folder_path = os.path.join(self.base_save_folder, folder_name)
        
        # 添加时间戳作为子文件夹的一部分
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_folder_path = os.path.join(self.base_save_folder, f"{folder_name}_{timestamp}")
        
        os.makedirs(new_folder_path)
        return new_folder_path
    
    def capture_frame_with_exposure_and_iso(self, exposure_value, iso_value, wait_time=2.5):
        """
        设置曝光和ISO并捕获一帧图像
        
        Args:
            exposure_value: 曝光值 (0-21)
            iso_value: ISO值 (0-18)
            wait_time: 设置后等待的时间（秒）
            
        Returns:
            (success, frame, save_path): 成功标志、捕获的帧、保存路径
        """
        exposure_time_str = self.camera_controller.get_exposure_time_string(exposure_value)
        iso_str = self.camera_controller.get_iso_string(iso_value)
        
        print(f"\n{'='*60}")
        print(f"📸 设置参数:")
        print(f"  曝光值: {exposure_value} ({self.camera_controller._shutter_to_text(exposure_value)})")
        print(f"  曝光时间: {exposure_time_str}")
        print(f"  ISO值: {iso_value} ({self.camera_controller._iso_to_text(iso_value)})")
        print(f"  最小ISO值: {iso_value} ({self.camera_controller._iso_to_text(iso_value)})")  # 显示最小ISO信息
        
        # 设置曝光时间
        if not self.camera_controller.set_shutter_for_dual(exposure_value):
            print("❌ 曝光设置失败")
            return False, None, None
        
        # 设置ISO（同时设置最小ISO）
        if not self.camera_controller.set_iso_for_dual(iso_value):
            print("❌ ISO设置失败")
            return False, None, None
        
        # 等待调整生效
        print(f"⏳ 等待参数调整生效 ({wait_time}秒)...")
        time.sleep(wait_time)
        
        # 捕获帧
        print("🖼️  捕获帧...")
        frame = None
        capture_attempts = 0
        max_attempts = 10
        
        while frame is None and capture_attempts < max_attempts:
            frame = self.streamer.grab_frame()
            if frame is None:
                time.sleep(0.1)
                capture_attempts += 1
        
        if frame is None:
            print("❌ 帧捕获失败")
            return False, None, None
        
        # 保存截图
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 使用曝光时间和ISO作为文件名
        if exposure_time_str == "auto":
            exposure_prefix = "exp_auto"
        else:
            exposure_prefix = f"exp_{exposure_time_str}"
        
        if iso_str == "auto":
            iso_prefix = "iso_auto"
        else:
            iso_prefix = f"iso_{iso_str}"
        
        filename = f"{exposure_prefix}_{iso_prefix}_{timestamp}.jpg"
        save_path = os.path.join(self.save_folder, filename)
        
        success = cv2.imwrite(save_path, frame)
        if success:
            print(f"✅ 截图已保存: {save_path}")
            print(f"  文件名: {filename}")
        else:
            print(f"❌ 截图保存失败: {save_path}")
        
        return success, frame, save_path
    
    def run_exposure_and_iso_test(self, exposure_sequence, iso_sequence, wait_time=2.5):
        """
        按顺序运行曝光和ISO测试组合
        
        Args:
            exposure_sequence: 曝光值列表 [0, 4, 12, ...]
            iso_sequence: ISO值列表 [0, 1, 2, ...]
            wait_time: 每次设置后等待的时间（秒）
            
        Returns:
            list: 保存的图片路径列表
        """
        print(f"\n{'='*60}")
        print("🚀 开始曝光和ISO组合测试")
        print(f"曝光测试序列: {exposure_sequence}")
        print(f"ISO测试序列: {iso_sequence}")
        
        print(f"总测试组合数: {len(exposure_sequence) * len(iso_sequence)}")
        
        # 显示测试序列对应的参数
        print("\n📊 曝光序列对应参数:")
        for exp_val in exposure_sequence:
            exp_str = self.camera_controller.get_exposure_time_string(exp_val)
            exp_text = self.camera_controller._shutter_to_text(exp_val)
            print(f"  {exp_val:2d} -> {exp_str:10s} ({exp_text})")
        
        print("\n📊 ISO序列对应参数 (同时设置最小ISO):")
        for iso_val in iso_sequence:
            iso_str = self.camera_controller.get_iso_string(iso_val)
            iso_text = self.camera_controller._iso_to_text(iso_val)
            print(f"  {iso_val:2d} -> {iso_str:6s} ({iso_text}) (最小ISO: {iso_text})")
        
        print(f"📁 保存文件夹: {self.save_folder}")
        print(f"📷 相机IP: {self.camera_ip}")
        print(f"{'='*60}")
        
        saved_paths = []
        total_tests = len(exposure_sequence) * len(iso_sequence)
        current_test = 0
        
        for exp_idx, exposure_value in enumerate(exposure_sequence):
            print(f"\n🔧 曝光值设置: {exposure_value} ({self.camera_controller._shutter_to_text(exposure_value)})")
            
            for iso_idx, iso_value in enumerate(iso_sequence):
                current_test += 1
                print(f"\n🔢 测试 {current_test}/{total_tests}")
                print(f"  曝光设置: {exposure_value} ({self.camera_controller.get_exposure_time_string(exposure_value)})")
                print(f"  ISO设置: {iso_value} ({self.camera_controller.get_iso_string(iso_value)}) (最小ISO: {self.camera_controller._iso_to_text(iso_value)})")
                
                success, frame, save_path = self.capture_frame_with_exposure_and_iso(
                    exposure_value, 
                    iso_value,
                    wait_time
                )
                
                if success and frame is not None:
                    saved_paths.append(save_path)
                else:
                    print(f"⚠️  测试 {current_test} 失败，继续下一个...")
            
            # 当完成一个ISO序列后，更换曝光时间前，等待3倍时间
            '''if exp_idx < len(exposure_sequence) - 1:  # 不是最后一个曝光值
                print(f"\n{'='*60}")
                print(f"🔄 完成ISO序列测试，准备切换曝光时间")
                print(f"⏳ 等待{3*wait_time}秒让相机稳定...")
                time.sleep(3 * wait_time)'''
        
        return saved_paths
    
    def run_exposure_test_only(self, exposure_sequence, wait_time=2.0):
        """
        仅运行曝光测试（不调整ISO，保持自动ISO）
        
        Args:
            exposure_sequence: 曝光值列表 [0, 4, 12, ...]
            wait_time: 每次曝光后等待的时间（秒）
            
        Returns:
            list: 保存的图片路径列表
        """
        print(f"\n{'='*60}")
        print("🚀 开始仅曝光测试（ISO保持自动）")
        print(f"曝光测试序列: {exposure_sequence}")
        
        print(f"总测试数: {len(exposure_sequence)}")
        
        # 显示测试序列对应的曝光时间
        print("对应曝光时间:")
        for exp_val in exposure_sequence:
            exp_str = self.camera_controller.get_exposure_time_string(exp_val)
            exp_text = self.camera_controller._shutter_to_text(exp_val)
            print(f"  {exp_val:2d} -> {exp_str:10s} ({exp_text})")
        
        print(f"📁 保存文件夹: {self.save_folder}")
        print(f"📷 相机IP: {self.camera_ip}")
        print(f"{'='*60}")
        
        saved_paths = []
        
        for i, exposure_value in enumerate(exposure_sequence):
            print(f"\n🔢 测试 {i+1}/{len(exposure_sequence)}")
            
            # 使用自动ISO (iso=0)
            success, frame, save_path = self.capture_frame_with_exposure_and_iso(
                exposure_value, 
                0,  # 自动ISO
                wait_time
            )
            
            if success and frame is not None:
                saved_paths.append(save_path)
            else:
                print(f"⚠️  测试 {i+1} 失败，继续下一个...")
            
            '''# 在曝光值切换之间增加额外等待时间
            if i < len(exposure_sequence) - 1:  # 不是最后一个曝光值
                print(f"\n🔄 准备切换曝光时间")
                print(f"⏳ 等待{3*wait_time}秒让相机稳定...")
                time.sleep(3 * wait_time)'''
        
        return saved_paths
    
    def create_summary_report(self, saved_paths, exposure_sequence=None, iso_sequence=None, test_type="combined"):
        """
        创建测试总结报告
        
        Args:
            saved_paths: 保存的图片路径列表
            exposure_sequence: 使用的曝光序列
            iso_sequence: 使用的ISO序列
            test_type: 测试类型 ("combined" 或 "exposure_only")
        """
        print(f"\n{'='*60}")
        print("📊 测试总结报告")
        print(f"{'='*60}")
        print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📷 相机IP: {self.camera_ip}")
        print(f"📁 保存文件夹: {os.path.abspath(self.save_folder)}")
        print(f"🔧 测试类型: {'曝光+ISO组合测试' if test_type == 'combined' else '仅曝光测试'}")
        print(f"📝 备注: ISO设置时同时设置了最小ISO值")
        
        if test_type == "combined" and exposure_sequence and iso_sequence:
            total_expected = len(exposure_sequence) * len(iso_sequence)
            print(f"📈 测试组合数: {len(exposure_sequence)} × {len(iso_sequence)} = {total_expected}")
        elif test_type == "exposure_only" and exposure_sequence:
            print(f"📈 测试次数: {len(exposure_sequence)}")
        
        print(f"✅ 成功保存: {len(saved_paths)}/{total_expected if test_type == 'combined' and exposure_sequence and iso_sequence else len(exposure_sequence) if exposure_sequence else '?'} 张图片")
        
        # 统计不同曝光和ISO组合
        '''if saved_paths:
            print("\n📋 保存的文件详情:")
            
            for i, path in enumerate(saved_paths):
                filename = os.path.basename(path)
                file_size = os.path.getsize(path) / 1024  # KB
                
                # 从文件名提取信息
                parts = filename.split('_')
                if len(parts) >= 4:
                    exp_info = parts[0] + "_" + parts[1]
                    iso_info = parts[2] + "_" + parts[3]
                    print(f"  {i+1:3d}. {exp_info} | {iso_info} | {file_size:.1f} KB | {filename}")
                else:0
                    print(f"  {i+1:3d}. {filename} ({file_size:.1f} KB)")
        
        print(f"\n💡 建议:")
        print("  1. 检查不同曝光和ISO组合下的图像质量")
        print("  2. 根据噪点、亮度等指标选择最佳组合")
        print("  3. 可以重复测试以验证结果一致性")
        print("  4. 注意：ISO设置时已同时设置了最小ISO值")
        print(f"{'='*60}")'''
    
    def cleanup(self):
        """清理资源"""
        print("\n🧹 清理资源...")
        self.streamer.release()
        cv2.destroyAllWindows()
        print("✅ 资源已清理")


def main():
    """
    主函数：测试曝光和ISO组合
    """
    # 配置参数
    CAMERA_IP = "192.168.1.160"  # 相机IP地址
    BASE_SAVE_FOLDER = "/media/pc/新加卷/YU/cap_photo"  # 基础保存文件夹0
    
    # 定义曝光测试序列
    # 示例：测试几个关键的曝光值
    EXPOSURE_SEQUENCE = [2, 4, 8, 15, 17, 19, 20, 21]  # 自动, 1/100s, 1/300s, 1/1000s, 1/4000s
    
    # 定义ISO测试序列
    # 示例：测试几个关键的ISO值
    ISO_SEQUENCE = [250, 218, 202, 194, 128, 32, 8, 2, 1]  # 自动, 00,1600 12800 25600 51200 102400 204800
    
    # 每次设置后等待时间（秒）
    WAIT_TIME = 2.0
    
    print("👁️ 双目相机曝光和ISO组合测试工具")
    print("=" * 60)
    print(f"📁 基础保存文件夹: {BASE_SAVE_FOLDER}")
    print("每次运行将创建新的编号子文件夹")
    print("注意：ISO设置时同时设置最小ISO值")
    print(f"注意：每个ISO序列完成后将等待{3*WAIT_TIME}秒再切换曝光时间")
    print("=" * 60)
    
    try:
        # 初始化测试捕获器
        tester = ExposureTestCapture(CAMERA_IP, BASE_SAVE_FOLDER)
        
        # 选择测试模式
        '''print("\n🔧 请选择测试模式:")
        print("1. 曝光和ISO组合测试")
        print("2. 仅曝光测试（ISO保持自动）") '''
        
        #choice = input("请输入选择 (1 或 2): ").strip()
        choice = "1"  # 默认模式1

        if choice == "1":
            # 运行曝光和ISO组合测试
            saved_paths = tester.run_exposure_and_iso_test(
                EXPOSURE_SEQUENCE, 
                ISO_SEQUENCE, 
                WAIT_TIME
            )
            
            # 创建测试总结报告
            tester.create_summary_report(
                saved_paths, 
                EXPOSURE_SEQUENCE, 
                ISO_SEQUENCE,
                test_type="combined"
            )
        elif choice == "2":
            # 运行仅曝光测试
            saved_paths = tester.run_exposure_test_only(
                EXPOSURE_SEQUENCE, 
                WAIT_TIME
            )
            
            # 创建测试总结报告
            tester.create_summary_report(
                saved_paths, 
                EXPOSURE_SEQUENCE,
                test_type="exposure_only"
            )
        else:
            print("❌ 无效选择，使用默认模式1（曝光和ISO组合测试）")
            saved_paths = tester.run_exposure_and_iso_test(
                EXPOSURE_SEQUENCE, 
                ISO_SEQUENCE, 
                WAIT_TIME
            )
            tester.create_summary_report(
                saved_paths, 
                EXPOSURE_SEQUENCE, 
                ISO_SEQUENCE,
                test_type="combined"
            )
        
        # 可选：显示一张样本图像
        '''if saved_paths:
            sample_path = saved_paths[0]
            sample_img = cv2.imread(sample_path)
            if sample_img is not None:
                cv2.namedWindow("样本图像", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("样本图像", 1280, 360)
                cv2.imshow("样本图像", sample_img)
                print("\n🖼️  按任意键关闭样本图像窗口...")
                cv2.waitKey(0)'''
        
        # 清理资源
        tester.cleanup()
        
        print("\n🎉 测试完成！")
        print(f"📁 所有文件已保存到: {tester.save_folder}")
        
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        print("请检查:")
        print("  1. 相机IP地址是否正确")
        print("  2. 相机是否已连接并开启")
        print("  3. 网络连接是否正常")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()