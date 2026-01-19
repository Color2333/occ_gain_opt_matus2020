"""
调试脚本 - 检查图像模型
"""

from ..data_acquisition import DataAcquisition
from ..config import ROIStrategy


def debug_image_model():
    """调试图像采集模型"""
    data_acq = DataAcquisition()

    led_duty_cycle = 50
    background_light = 50
    led_intensity = (led_duty_cycle / 100.0) * 255

    print("\n调试图像模型")
    print("=" * 60)

    gains = [0, 5, 10, 15, 20]

    print(f"\nLED强度: {led_intensity}")
    print(f"背景光: {background_light}")
    print(f"\n{'增益(dB)':<12} {'线性增益':<12} {'LED信号':<12} {'ROI灰度均值':<12}")
    print("-" * 60)

    for gain_db in gains:
        gain_linear = 10 ** (gain_db / 20.0)
        led_signal = led_intensity * gain_linear

        image = data_acq.capture_image(led_intensity, gain_db, background_light)
        roi_mask = data_acq.select_roi(strategy=ROIStrategy.CENTER)
        stats = data_acq.get_roi_statistics(image, roi_mask)

        print(f"{gain_db:<12.1f} {gain_linear:<12.2f} {led_signal:<12.2f} "
              f"{stats['mean']:<12.2f}")

        if stats['saturated_ratio'] > 0.5:
            print(f"  ⚠ 警告: {stats['saturated_ratio'] * 100:.1f}% 像素饱和")

    print("\n分析:")
    print("  如果灰度值不随增益变化,说明:")
    print("  1. 图像可能立即饱和(达到255)")
    print("  2. 或者LED信号太弱,被背景淹没")
    print("  3. 或者ROI选择有问题")


def main():
    debug_image_model()


if __name__ == "__main__":
    main()
