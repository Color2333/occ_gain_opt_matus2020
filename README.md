# 光相机通信模拟增益优化算法复现

## 论文信息
- **标题**: Experimental Evaluation of an Analog Gain Optimization Algorithm in Optical Camera Communications
- **作者**: Matus 等人, 2020
- **补充**: Ma (2024) 自适应阻尼算法

## 研究概述

OCC（Optical Camera Communications）系统中，LED 发射 OOK 调制信号，相机通过滚动快门拍摄条纹图像接收数据。核心问题是**自动选择相机模拟增益**，使 ROI 灰度值逼近 255 但不超过（避免过饱和/欠曝光）。

## 项目结构

```
uocc-adaptive/
├── src/occ_gain_opt/              主包
│   ├── config.py                  全局配置（CameraParams, DemodulationConfig）
│   ├── demodulation.py            OOK 解调（OOKDemodulator）★ 唯一解调入口
│   ├── experiment_loader.py       ISO-Texp 数据集加载
│   ├── algorithms/                增益优化算法（插件式注册）
│   │   ├── single_shot.py         Matus 单次公式
│   │   ├── adaptive_iter.py       Matus 迭代优化
│   │   ├── adaptive_damping.py    Ma 5 状态自适应阻尼
│   │   └── ber_explore.py         BER 引导探索
│   ├── data_sources/              数据源抽象 + ROI 策略
│   ├── experiments/               实验封装（advisor, batch_demod）
│   ├── hardware/                  海康 ISAPI 相机控制
│   ├── cli.py                     命令行入口 occ-gain-opt
│   ├── realtime.py                实时单步接口
│   ├── simulation.py              仿真
│   └── visualization.py           可视化
│
├── scripts/                       运行脚本
│   ├── demo_demodulation.py       OOK 解调可视化演示
│   ├── batch_demodulate.py        批量解调 + BER 分析 + 出图
│   ├── ber_analysis.py            BER vs 增益曲线分析
│   ├── ber_comparison_plot.py     BER 对比可视化
│   └── validate_algorithms.py     三算法验证
│
├── data/                          发射序列真值
│   ├── Mseq_32_original.csv       32-bit PRBS 真值
│   ├── Mseq_32_with_header.csv    带 header 的完整序列
│   └── Tx_PRBS.py                 序列生成脚本
│
├── ISO-Texp/                      实验数据集（234 张真实图片）
│   ├── bubble/                    气泡干扰实验
│   ├── tap water/                 自来水实验
│   └── turbidity/                 浑浊度实验
│
├── example/                       MATLAB 参考代码 + 示例图片
├── real/                          实时采集脚本 + 参考实现
├── docs/                          文档
├── results/                       输出结果（gitignored）
├── camera_isapi.py                海康相机 ISAPI 控制
└── realtime_experiment_app.py     Streamlit 实验面板
```

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

## 运行

```bash
# 1. OOK 解调演示
python scripts/demo_demodulation.py

# 2. 批量解调 + BER 分析（生成 CSV + 报告 + 图表）
python scripts/batch_demodulate.py

# 3. BER vs 增益曲线
python scripts/ber_analysis.py

# 4. BER 对比可视化
python scripts/ber_comparison_plot.py

# 5. 三算法验证
python scripts/validate_algorithms.py

# CLI 工具
occ-gain-opt --test                              # 快速测试
occ-gain-opt advisor --image x.jpg --iso 35      # 单帧三算法顾问
```

## 解调算法

**唯一入口**: `from occ_gain_opt.demodulation import OOKDemodulator`

数据包结构: Header `[0,1,1,1,1,1,1,0]` + Data `32-bit PRBS` = 40 bits，重复 3 次。

解调步骤:
1. 全图灰度行均值
2. 3 阶多项式拟合二值化（抑制滚动快门亮度渐变）
3. 找两个最长亮游程作为同步头
4. 同步头间均匀采样 34 点，取 `[1:33]` 为 32-bit 数据

**实测结果**: 222/234 张图片 BER=0，整体平均 BER=0.0184。

## 增益优化算法

| 算法 | 公式 | 特点 |
|------|------|------|
| Single-shot | `G_opt = G_curr + 20·log₁₀(Y_target/Y_curr)` | 一步到位 |
| Adaptive Iter | `G_{k+1} = G_k + α·20·log₁₀(Y_target/Y_k)` | 1-4 次迭代收敛 |
| Adaptive Damping | Ma 5 状态机 | 同时调 ISO + 曝光 |

目标灰度: Y_target = 255 × 0.95 = 242.25

## 技术栈

Python 3.8+, NumPy, SciPy, OpenCV, Matplotlib, pandas

## 许可证

仅用于学术研究和教育目的。
