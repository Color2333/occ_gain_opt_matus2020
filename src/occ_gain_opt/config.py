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
