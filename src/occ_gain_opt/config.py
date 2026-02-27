"""
配置参数模块
包含相机、LED和优化算法的所有参数
"""


class CameraConfig:
    """相机配置参数"""
    # 模拟增益范围 (根据论文表格2)
    GAIN_MIN = 0.0      # 最小增益 (dB)
    GAIN_MAX = 20.0     # 最大增益 (dB)

    # 曝光时间 (微秒)
    EXPOSURE_MIN = 100
    EXPOSURE_MAX = 10000
    EXPOSURE_STEP = 100

    # 图像参数
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    FPS = 30

    # 饱和值
    SATURATION_VALUE = 255

    # 量化步长 (假设8位ADC)
    QUANTIZATION_STEP = 1.0


class LEDConfig:
    """LED配置参数"""
    # PWM占空比范围 (%)
    DUTY_CYCLE_MIN = 0
    DUTY_CYCLE_MAX = 100

    # 调制频率
    FREQUENCY_MIN = 1   # Hz
    FREQUENCY_MAX = 1000  # Hz

    # 默认占空比
    DEFAULT_DUTY_CYCLE = 50


class ROIStrategy:
    """ROI选择策略"""
    CENTER = "center"           # 中心区域
    MANUAL = "manual"           # 手动选择
    AUTO_BRIGHTNESS = "auto"    # 自动选择最亮区域
    SYNC_BASED = "sync_based"   # 基于同步头的精确数据包ROI


class OptimizationConfig:
    """优化算法配置"""
    # 收敛条件
    TOLERANCE = 1e-3           # 收敛容忍度
    MAX_ITERATIONS = 20        # 最大迭代次数

    # 目标值
    TARGET_GRAY = 255          # 目标灰度值 (饱和点)
    TARGET_MARGIN = 5          # 容忍边界 (避免过饱和)

    # 步长控制
    STEP_SIZE_MIN = 0.1        # 最小步长 (dB)
    STEP_SIZE_MAX = 5.0        # 最大步长 (dB)

    # 安全因子 (防止过饱和)
    SAFETY_FACTOR = 0.95       # 95%的饱和值


class ExperimentConfig:
    """实验配置"""
    # 场景类型
    SCENARIOS = {
        'low_light': '低光照环境',
        'normal': '正常光照',
        'high_light': '高光照环境',
        'varying_light': '变化光照'
    }

    # 测试点数量
    TEST_POINTS = 50

    # 噪声参数
    NOISE_MEAN = 0
    NOISE_STD = 2.0            # 高斯噪声标准差

    # 背景光强范围
    BACKGROUND_MIN = 10
    BACKGROUND_MAX = 200


class PerformanceConfig:
    """性能评估配置"""
    # 评估指标
    METRICS = ['MSE', 'PSNR', 'SSIM', 'SNR']

    # 滤波器类型
    FILTER_TYPES = ['gaussian', 'bilateral', 'median']

    # 窗口大小 (用于局部MSE)
    WINDOW_SIZE = 5


class VisualizationConfig:
    """可视化配置"""
    # 图表尺寸
    FIGURE_WIDTH = 12
    FIGURE_HEIGHT = 8

    # 颜色方案
    COLORS = {
        'original': 'blue',
        'optimized': 'red',
        'target': 'green',
        'background': 'gray'
    }

    # 保存设置
    SAVE_PLOTS = True
    PLOT_DPI = 300
    OUTPUT_DIR = 'results/plots'


class DemodulationConfig:
    """OOK 解调配置"""
    # 数据包参数
    DATA_BITS = 32                      # 数据位长度 (p32)
    SYNC_PATTERN = None                 # 同步头比特模式; None = 自动检测

    # LED 区域检测
    LED_CHANNEL = "green"               # 使用的颜色通道
    COL_MARGIN_RATIO = 0.1              # LED列边界向内收缩比例

    # 信号处理
    SMOOTHING_SIGMA = 1.5               # 行均值高斯平滑 sigma
    THRESHOLD_METHOD = "otsu"           # 二值化方法: "otsu", "midpoint", "percentile"
    PERCENTILE_LOW = 30                 # percentile 方法的低分位数
    PERCENTILE_HIGH = 70               # percentile 方法的高分位数

    # 时钟恢复
    MIN_BIT_PERIOD = 10                 # 最小位周期 (行数)
    MAX_BIT_PERIOD = 80                 # 最大位周期 (行数)

    # 同步头检测
    CORRELATION_THRESHOLD = 0.7         # 相关检测阈值
    MIN_PACKETS = 1                     # 图像中至少需要的完整数据包数

    # 自动检测参数
    AUTO_DETECT_MAX_SYNC_LEN = 16       # 自动检测同步头最大长度 (bits)
    AUTO_DETECT_MIN_SYNC_LEN = 4        # 自动检测同步头最小长度 (bits)
