import cv2
import os
import time
import datetime
from threading import Thread
from queue import Queue

# 1. 视频流读取线程（保持不变，负责无延迟看画面）
class ThreadedCamera(object):
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.status = False
        self.frame = None

    def update(self):
        while self.running:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(0.01)

    def grab_frame(self):
        return self.frame if self.status else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.capture.release()

# 2. 硬盘后台写入线程（核心升级：防掉帧）
class ImageWriterThread(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue
        self.daemon = True
        self.start()

    def run(self):
        while True:
            # 从队列里拿图片和路径，慢慢存进硬盘
            filepath, frame = self.queue.get()
            if filepath is None: # 收到停止信号
                break
            # 使用最高质量参数保存，保证条纹边缘绝对锐利
            cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            self.queue.task_done()

if __name__ == '__main__':
    # 你的云台 RTSP 视频流地址
    rtsp_url = 'rtsp://admin:abcd1234@192.168.1.19/Streaming/Channels/1'
    save_dir = r"E:\graduation_pics\RS_Underwater"
    
    session_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    session_folder = os.path.join(save_dir, f"test_{session_time}")
    os.makedirs(session_folder, exist_ok=True)

    streamer = ThreadedCamera(rtsp_url)
    print("正在连接相机...")
    time.sleep(2)

    # 创建一个能容纳 1000 张图片的缓冲队列和存图线程
    write_queue = Queue(maxsize=1000)
    writer_thread = ImageWriterThread(write_queue)

    print(f"✅ 连接成功！通信帧将保存在: {session_folder}")
    print(">>> 准备好后，按下 's' 键开始/停止连续 25FPS 满载采集 <<<")
    print(">>> 按下 'q' 键退出 <<<")

    is_recording = False
    img_count = 0

    while True:
        frame = streamer.grab_frame()
        if frame is None:
            continue
            
        display_frame = frame.copy()
        
        # 核心：如果是录制状态，来一帧存一帧，不加任何 sleep！
        if is_recording:
            img_count += 1
            filename = os.path.join(session_folder, f"frame_{img_count:05d}.jpg")
            # 把任务丢进队列就跑，主循环不卡顿
            write_queue.put((filename, frame))
            
            # 在画面上显示实时进度
            cv2.putText(display_frame, f"REC: {img_count} frames", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "Ready. Press 's' to Start", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Communication Capture", display_frame)

        # 键盘监测 (1ms 极速响应)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            is_recording = not is_recording
            if is_recording:
                print("开始 25FPS 满载抓拍！请保持光路稳定...")
            else:
                print(f"抓拍结束。共捕获 {img_count} 张。后台正在拼命保存到硬盘，请稍等几秒...")

    # 收尾工作
    streamer.stop()
    cv2.destroyAllWindows()
    
    # 告诉写硬盘的线程可以下班了
    write_queue.put((None, None)) 
    writer_thread.join()
    print(f" 全部保存完毕！共计 {img_count} 张 100% 质量条纹图。")