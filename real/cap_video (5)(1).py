import subprocess
import requests
import base64
import os
import time
import signal
import sys
from datetime import datetime

class CameraController:
    def __init__(self, ip, cgi_port=80, username="admin", password="12345"):
        """
        初始化相机控制器
        """
        self.ip = ip
        self.cgi_port = cgi_port
        self.username = username
        self.password = password
        
        # CGI控制接口URL
        self.base_url = f"http://{ip}:{cgi_port}/cgi-bin"
        
        # 认证信息
        self.auth = base64.b64encode(f"{username}:{password}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {self.auth}",
            "User-Agent": "CameraController"
        }
        
        # 控制端口映射
        self.control_channels = {'left': 0, 'right': 1}
    
    def set_shutter_for_dual(self, shutter_value):
        """设置左右双目的曝光时间"""
        if not 0 <= shutter_value <= 21:
            print(f"无效的快门值: {shutter_value} (支持范围: 0-21)")
            return False
            
        success = True
        for camera_name in ['left', 'right']:
            control_port = self.control_channels[camera_name]
            if not self._set_single_camera_shutter(control_port, shutter_value):
                success = False
        return success
    
    def set_iso_for_dual(self, iso_value):
        """设置左右双目的ISO值"""
        if not 0 <= iso_value <= 255:
            print(f"无效的ISO值: {iso_value}")
            return False
            
        success = True
        for camera_name in ['left', 'right']:
            control_port = self.control_channels[camera_name]
            if not self._set_single_camera_iso(control_port, iso_value):
                success = False
        return success
    
    def _set_single_camera_shutter(self, control_port, shutter_value):
        """设置单个相机的曝光时间"""
        url = f"{self.base_url}/param.cgi?action=update&group=DIS&channel={control_port}&DIS.shutter={shutter_value}"
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            return "root.ERR.no=0" in response.text
        except:
            return False
    
    def _set_single_camera_iso(self, control_port, iso_value):
        """设置单个相机的ISO值"""
        url = f"{self.base_url}/param.cgi?action=update&group=DIS&channel={control_port}&DIS.iso={iso_value}"
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            return "root.ERR.no=0" in response.text
        except:
            return False


class FFmpegRecorder:
    """使用ffmpeg录制RTSP流"""
    
    def __init__(self, camera_ip, save_folder="recordings"):
        """
        初始化FFmpeg录制器
        
        Args:
            camera_ip: 相机IP地址
            save_folder: 视频保存文件夹
        """
        self.camera_ip = camera_ip
        self.save_folder = save_folder
        self.process = None
        
        # RTSP URL
        self.rtsp_url = f"rtsp://{camera_ip}:8554/0"
        
        print(f"相机IP: {camera_ip}")
    
    def get_stream_info(self):
        """使用ffprobe获取流信息"""
        print("使用ffprobe获取流信息...")
        
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,duration,bit_rate,codec_name',
            '-of', 'csv=p=0',
            self.rtsp_url
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                info = result.stdout.strip().split(',')
                if len(info) >= 3:
                    width = int(info[0])
                    height = int(info[1])
                    
                    # 解析帧率（可能是分数形式如 "30000/1001"）
                    r_frame_rate = info[2]
                    if '/' in r_frame_rate:
                        num, den = map(int, r_frame_rate.split('/'))
                        fps = num / den
                    else:
                        fps = float(r_frame_rate) if r_frame_rate else 30.0
                    
                    codec = info[5] if len(info) > 5 else 'unknown'
                    
                    print(f"视频流信息:")
                    print(f"  分辨率: {width}x{height}")
                    print(f"  帧率: {fps:.2f} fps")
                    print(f"  编码: {codec}")
                    
                    if len(info) > 3 and info[3]:
                        print(f"  时长: {float(info[3]):.2f}秒")
                    if len(info) > 4 and info[4]:
                        bitrate = int(info[4]) / 1000  # kbps
                        print(f"  码率: {bitrate:.0f} kbps")
                    
                    return {
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'codec': codec
                    }
        except subprocess.TimeoutExpired:
            print("ffprobe超时")
        except Exception as e:
            print(f"获取流信息失败: {e}")
        
        return None
    
    def record_video(self, duration=10, shutter_value=4, iso_value=4, 
                    filename=None, crf=23, preset='medium', fps=None,
                    test_folder=None):
        """
        使用ffmpeg录制RTSP流视频
        
        Args:
            duration: 录制时长(秒)
            shutter_value: 快门值
            iso_value: ISO值
            filename: 保存文件名
            crf: 视频质量 (0-51, 越低质量越好)
            preset: 编码速度预设 (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
            fps: 目标帧率，None使用原始帧率
            test_folder: 测试子文件夹路径
            
        Returns:
            str: 保存的视频文件路径
        """
        print(f"\n{'='*50}")
        print(f"开始使用ffmpeg录制视频")
        print(f"录制时长: {duration}秒")
        print(f"曝光值: {shutter_value}")
        print(f"ISO值: {iso_value}")
        print(f"{'='*50}")
        
        # 创建测试子文件夹
        if test_folder:
            save_folder = test_folder
        else:
            save_folder = self.save_folder
            
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print(f"创建保存文件夹: {save_folder}")
        
        # 设置文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_shutter{shutter_value}_iso{iso_value}_{timestamp}.mp4"
        
        save_path = os.path.join(save_folder, filename)
        print(f"保存文件夹: {save_folder}")
        
        # 构建ffmpeg命令
        # 使用 -t 参数精确控制录制时长
        # 使用 -c copy 直接复制流，避免重新编码（如果需要调整参数则使用编码）
        cmd = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-rtsp_transport', 'tcp',  # 使用TCP传输，更稳定
            '-i', self.rtsp_url,
            '-t', str(duration),  # 录制时长
            '-c:v', 'libx264',  # H.264编码
            '-crf', str(crf),  # 质量参数
            '-preset', preset,  # 编码速度
            '-pix_fmt', 'yuv420p',  # 像素格式
            '-movflags', '+faststart',  # 快速启动
        ]
        
        # 如果指定了帧率，添加帧率参数
        if fps:
            cmd.extend(['-r', str(fps)])
        
        # 添加音频支持（如果流有音频）
        cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
        
        # 输出文件
        cmd.append(save_path)
        
        print(f"\nffmpeg命令:")
        print(' '.join(cmd))
        
        try:
            print(f"\n开始录制 {duration} 秒视频...")
            print(f"保存到: {save_path}")
            
            start_time = time.time()
            
            # 启动ffmpeg进程
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            # 监控进度
            last_progress_time = start_time
            while True:
                # 检查进程是否结束
                return_code = self.process.poll()
                if return_code is not None:
                    break
                
                # 显示进度
                current_time = time.time()
                elapsed = current_time - start_time
                
                if current_time - last_progress_time >= 1.0:
                    progress = min(100.0, (elapsed / duration) * 100)
                    print(f"录制进度: {elapsed:.1f}/{duration}秒 ({progress:.1f}%)")
                    last_progress_time = current_time
                
                # 检查是否超时
                if elapsed > duration + 5:  # 给5秒宽容时间
                    print("录制超时，强制结束")
                    self.stop()
                    break
                
                time.sleep(0.1)
            
            # 等待进程结束
            if self.process:
                stdout, stderr = self.process.communicate(timeout=5)
                if stderr:
                    # 提取关键信息
                    for line in stderr.split('\n'):
                        if 'frame=' in line and 'fps=' in line:
                            print(f"编码信息: {line.strip()}")
            
            actual_duration = time.time() - start_time
            
            print(f"\n录制完成!")
            print(f"目标时长: {duration}秒")
            print(f"实际时长: {actual_duration:.2f}秒")
            print(f"文件保存到: {save_path}")
            
            # 验证文件
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path) / (1024*1024)  # MB
                print(f"文件大小: {file_size:.2f} MB")
                
                # 使用ffprobe验证视频信息
                verify_cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    save_path
                ]
                
                try:
                    result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        video_duration = float(result.stdout.strip())
                        print(f"视频实际时长: {video_duration:.2f}秒")
                        print(f"时长误差: {abs(video_duration - duration):.2f}秒")
                except:
                    pass
            else:
                print("警告: 输出文件不存在")
            
            return save_path
            
        except subprocess.TimeoutExpired:
            print("ffmpeg进程超时")
            return None
        except KeyboardInterrupt:
            print("\n用户中断录制")
            self.stop()
            return None
        except Exception as e:
            print(f"录制过程中出错: {e}")
            return None
        finally:
            self.process = None
    
    def record_video_copy(self, duration=10, shutter_value=4, iso_value=4, 
                         filename=None, test_folder=None):
        """
        使用流复制模式录制（最快，不重新编码）
        适用于只想保存原始流的场景
        
        Args:
            test_folder: 测试子文件夹路径
        """
        print(f"\n使用流复制模式录制（不重新编码）")
        
        # 创建测试子文件夹
        if test_folder:
            save_folder = test_folder
        else:
            save_folder = self.save_folder
            
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_raw_shutter{shutter_value}_iso{iso_value}_{timestamp}.mkv"
        
        save_path = os.path.join(save_folder, filename)
        print(f"保存文件夹: {save_folder}")
        
        # 流复制命令，保持原始编码
        cmd = [
            'ffmpeg',
            '-y',
            '-rtsp_transport', 'tcp',
            '-i', self.rtsp_url,
            '-t', str(duration),
            '-c:v', 'copy',  # 直接复制视频流
            '-c:a', 'copy',  # 直接复制音频流
            '-f', 'matroska',  # 使用MKV容器，兼容性好
            save_path
        ]
        
        print(f"\nffmpeg命令（流复制）:")
        print(' '.join(cmd))
        
        try:
            print(f"开始录制 {duration} 秒视频...")
            start_time = time.time()
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,  # 不显示输出
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # 简单等待
            while True:
                return_code = self.process.poll()
                if return_code is not None:
                    break
                
                elapsed = time.time() - start_time
                if elapsed > duration + 2:  # 稍微多等2秒
                    print(f"录制已进行 {elapsed:.1f} 秒，强制结束...")
                    self.stop()
                    break
                
                time.sleep(0.5)
            
            print(f"\n录制完成!")
            print(f"文件保存到: {save_path}")
            
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path) / (1024*1024)
                print(f"文件大小: {file_size:.2f} MB")
            
            return save_path
            
        except Exception as e:
            print(f"录制失败: {e}")
            return None
    
    def stop(self):
        """停止录制"""
        if self.process and self.process.poll() is None:
            print("正在停止ffmpeg进程...")
            try:
                # 发送Ctrl+C信号
                self.process.send_signal(signal.SIGINT)
                self.process.wait(timeout=5)
            except:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=2)
                except:
                    self.process.kill()
            finally:
                self.process = None


class VideoRecorder:
    """主录制控制器"""
    
    def __init__(self, camera_ip, save_folder="recordings"):
        self.camera_ip = camera_ip
        self.save_folder = save_folder
        
        # 初始化相机控制器
        self.camera_controller = CameraController(ip=camera_ip)
        
        # 初始化FFmpeg录制器（不再在初始化时创建文件夹）
        self.ffmpeg_recorder = FFmpegRecorder(camera_ip, save_folder)
        
        # 测试计数器
        self.test_counter = 0
    
    def record_video(self, duration=10, shutter_value=4, iso_value=4, 
                    use_copy_mode=False, **kwargs):
        """
        录制视频
        
        Args:
            duration: 录制时长(秒)
            shutter_value: 快门值
            iso_value: ISO值
            use_copy_mode: 是否使用流复制模式
            **kwargs: 其他ffmpeg参数
            
        Returns:
            str: 保存的视频文件路径
        """
        print(f"\n{'='*50}")
        print(f"视频录制任务")
        print(f"{'='*50}")
        
        # 设置相机参数
        print("设置相机参数...")
        if not self.camera_controller.set_shutter_for_dual(shutter_value):
            print("曝光设置失败，继续录制...")
        
        if not self.camera_controller.set_iso_for_dual(iso_value):
            print("ISO设置失败，继续录制...")
        
        # 等待参数生效
        print("等待参数生效 (3秒)...")
        time.sleep(3)
        
        # 选择录制模式
        if use_copy_mode:
            return self.ffmpeg_recorder.record_video_copy(
                duration=duration,
                shutter_value=shutter_value,
                iso_value=iso_value,
                test_folder=kwargs.get('test_folder')  # 传递测试文件夹
            )
        else:
            return self.ffmpeg_recorder.record_video(
                duration=duration,
                shutter_value=shutter_value,
                iso_value=iso_value,
                test_folder=kwargs.get('test_folder')  # 传递测试文件夹
            )
    
    def batch_record(self, durations, shutter_values, iso_values, 
                    use_copy_mode=False):
        """
        批量录制多个视频
        
        Args:
            durations: 时长列表
            shutter_values: 快门值列表
            iso_values: ISO值列表
            use_copy_mode: 是否使用流复制模式
            
        Returns:
            list: 保存的文件路径列表
        """
        if len(durations) != len(shutter_values) or len(durations) != len(iso_values):
            print("参数列表长度不一致")
            return []
        
        saved_paths = []
        
        # 修改：创建与单次保存格式一致的批次文件夹，视频文件直接保存在该文件夹中
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 查找最大的批次序号
        batch_index = self._find_next_batch_index()
        
        # 创建批次文件夹，格式为 batch_001_时间
        batch_folder = os.path.join(self.save_folder, f"batch_{batch_index:03d}_{timestamp}")
        if not os.path.exists(batch_folder):
            os.makedirs(batch_folder)
            print(f"📁 创建批次文件夹: {batch_folder}")
        
        for i, (duration, shutter, iso) in enumerate(zip(durations, shutter_values, iso_values)):
            print(f"\n🔢 录制任务 {i+1}/{len(durations)}")
            print(f"参数: 时长={duration}s, 快门={shutter}, ISO={iso}")
            
            # 修改：不再创建子文件夹，直接使用batch_folder作为保存路径
            save_path = self.record_video(
                duration=duration,
                shutter_value=shutter,
                iso_value=iso,
                use_copy_mode=use_copy_mode,
                test_folder=batch_folder  # 直接使用批次文件夹，不创建子文件夹
            )
            
            if save_path:
                saved_paths.append(save_path)
            
            # 任务间等待
            if i < len(durations) - 1:
                print(f"\n等待5秒开始下一个录制...")
                time.sleep(5)
        
        return saved_paths
    
    def _find_next_batch_index(self):
        """查找文件夹中最大的批次序号，返回下一个可用的序号"""
        import re
        
        # 确保保存文件夹存在
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            return 1  # 如果文件夹为空，从001开始
        
        max_index = 0
        
        # 扫描保存文件夹中的所有子文件夹
        for item in os.listdir(self.save_folder):
            item_path = os.path.join(self.save_folder, item)
            
            # 只检查文件夹
            if os.path.isdir(item_path):
                # 使用正则表达式匹配 batch_001_20241216_120000 这样的格式
                match = re.match(r'^batch_(\d{3})_\d{8}_\d{6}$', item)
                if match:
                    current_index = int(match.group(1))
                    max_index = max(max_index, current_index)
        
        # 返回下一个序号
        return max_index + 1
    
    def single_record(self, duration=10, shutter_value=4, iso_value=4, 
                    use_copy_mode=False, test_index=None):
        """
        单次录制，保存到带序号的子文件夹
        
        Args:
            test_index: 测试序号，如果为None则自动查找最大序号+1
        """
        if test_index is None:
            # 自动查找最大的测试序号
            test_index = self._find_next_test_index()
        else:
            self.test_counter = max(self.test_counter, test_index)
        
        # 创建测试子文件夹
        timestamp = datetime.now().strftime("%Y%m%d")
        test_folder = os.path.join(self.save_folder, f"test_{test_index:03d}_{timestamp}")
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        
        print(f"\n📁 保存到子文件夹: {test_folder}")
        
        save_path = self.record_video(
            duration=duration,
            shutter_value=shutter_value,
            iso_value=iso_value,
            use_copy_mode=use_copy_mode,
            test_folder=test_folder
        )
        
        # 更新计数器
        self.test_counter = test_index
        
        return save_path

    def _find_next_test_index(self):
        """查找文件夹中最大的测试序号，返回下一个可用的序号"""
        import re
        
        # 确保保存文件夹存在
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            return 1  # 如果文件夹为空，从001开始
        
        max_index = 0
        
        # 扫描保存文件夹中的所有子文件夹
        for item in os.listdir(self.save_folder):
            item_path = os.path.join(self.save_folder, item)
            
            # 只检查文件夹
            if os.path.isdir(item_path):
                # 使用正则表达式匹配 test_001_20241216 这样的格式
                match = re.match(r'^test_(\d{3})_\d{8}$', item)
                if match:
                    current_index = int(match.group(1))
                    max_index = max(max_index, current_index)
        
        # 返回下一个序号
        return max_index + 1

def main():
    """
    主函数
    """
    # 配置参数
    CAMERA_IP = "192.168.1.160"
    SAVE_FOLDER = "/media/pc/新加卷/YU/cap_videos"
    
    # 录制参数
    RECORD_DURATION = 30  # 录制时长
    SHUTTER_VALUE = 0
    ISO_VALUE = 0
    USE_COPY_MODE = False  # True:流复制(快), False:重新编码(质量好)
    
    # 批量录制参数
    BATCH_RECORD = True
    #BATCH_RECORD = False
    
    DURATIONS = [30, 30, 30]
    SHUTTERS = [4, 19, 0] #曝光时间参数
    ISOS = [32, 202, 0] #感光度参数
    
    print("📹 FFmpeg RTSP视频录制工具")
    print("=" * 50)
    print(f"相机IP: {CAMERA_IP}")
    
    print(f"保存文件夹: {SAVE_FOLDER}")
    print(f"录制模式: {'流复制(快速)' if USE_COPY_MODE else '重新编码(高质量)'}")
    print("=" * 50)
    
    try:
        # 初始化
        recorder = VideoRecorder(CAMERA_IP, SAVE_FOLDER)
        
        # 检查ffmpeg是否可用
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            print("✓ ffmpeg可用")
        except:
            print("✗ ffmpeg未安装或不可用")
            print("请安装ffmpeg: sudo apt install ffmpeg")
            return
        
        # 获取流信息
        print("\n获取视频流信息...")
        stream_info = recorder.ffmpeg_recorder.get_stream_info()
        
        if BATCH_RECORD:
            # 批量录制
            print("\n批量录制模式")
            saved_paths = recorder.batch_record(
                DURATIONS, SHUTTERS, ISOS, 
                use_copy_mode=USE_COPY_MODE
            )
            
            print(f"\n批量录制完成!")
            print(f"成功录制: {len(saved_paths)}/{len(DURATIONS)} 个视频")
            for i, path in enumerate(saved_paths):
                size = os.path.getsize(path) / (1024*1024) if os.path.exists(path) else 0
                print(f"{i+1}. {os.path.basename(path)} ({size:.1f} MB)")
        else:
            # 单次录制，使用带序号的子文件夹
            save_path = recorder.single_record(
                duration=RECORD_DURATION,
                shutter_value=SHUTTER_VALUE,
                iso_value=ISO_VALUE,
                use_copy_mode=USE_COPY_MODE
            )
            
            if save_path:
                print(f"\n✅ 录制完成!")
                print(f"视频文件: {os.path.basename(save_path)}")
                print(f"完整路径: {save_path}")
                
                # 验证时长
                if os.path.exists(save_path):
                    verify_cmd = [
                        'ffprobe',
                        '-v', 'error',
                        '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1',
                        save_path
                    ]
                    try:
                        result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            actual_duration = float(result.stdout.strip())
                            print(f"实际视频时长: {actual_duration:.2f}秒")
                            print(f"目标时长: {RECORD_DURATION}秒")
                            print(f"误差: {abs(actual_duration - RECORD_DURATION):.2f}秒")
                    except:
                        pass
            else:
                print("\n❌ 录制失败!")
        
        print(f"\n所有操作完成!")
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()