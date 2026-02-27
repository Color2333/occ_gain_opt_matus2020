# 光相机通信模拟增益优化算法复现

## 论文信息
- **标题**: Experimental Evaluation of an Analog Gain Optimization Algorithm in Optical Camera Communications
- **作者**: Matus, 等人
- **年份**: 2020

## 研究概述
本研究提出了一种在光相机通信(OCC)系统中优化相机模拟增益设置的算法。OCC系统利用LED作为发射器,相机作为接收器进行无线通信。

### 核心问题
- **欠饱和/欠曝光**: 图像太暗,难以从背景中区分ROI
- **过饱和**: 图像太亮,达到255的最大强度

### 解决方案
开发一种自动控制算法,为相机选择最佳模拟增益值,使ROI的灰度值尽可能接近255(饱和点),但不超过它。

## 算法核心

### 1. 数据采集 (Data Acquisition)
- 设置LED发射强度和初始相机增益
- 选择ROI (感兴趣区域)
- 提取ROI的灰度值

### 2. 增益优化 (Gain Optimization)
**目标**: 找到最优增益 \( G_{opt} \),使ROI的灰度值最接近255但不超过它

**单次优化算法**:
\[
G_{opt} = G_{curr} \times \frac{Y_{target}}{Y_{curr}}
\]

转换为dB:
\[
G_{opt(dB)} = G_{curr(dB)} + 20 \log_{10}\left(\frac{Y_{target}}{Y_{curr}}\right)
\]

**迭代优化算法**:
\[
G_{k+1} = G_k + \alpha \times 20 \log_{10}\left(\frac{Y_{target}}{Y_k}\right)
\]

其中:
- \( G_{curr} \): 当前增益
- \( Y_{curr} \): 当前测量的灰度值
- \( Y_{target} \): 目标灰度 (通常为 255 × 0.95 = 242.25)
- \( \alpha \): 学习率 (推荐值 0.3-0.7)

### 3. 性能评估 (Performance Evaluation)
使用均方误差(MSE)评估图像质量:

\[
MSE = \frac{1}{MN} \sum_{i=1}^{M} \sum_{j=1}^{N} [I(i,j) - \hat{I}(i,j)]^2
\]

其中:
- \( I(i,j) \): 原始图像像素值
- \( \hat{I}(i,j) \): 去噪后的像素值
- M, N: 图像尺寸

## 项目结构
```
kg/
├── pyproject.toml                     # 标准包配置
├── src/occ_gain_opt/                  # 主包代码
│   ├── __init__.py
│   ├── cli.py                         # 命令行入口
│   ├── config.py                      # 配置参数 (含 DemodulationConfig)
│   ├── data_acquisition.py            # 数据采集 + ROI策略 (CENTER/AUTO/SYNC_BASED)
│   ├── demodulation.py                # OOK 解调模块 (同步头检测 + 数据包定位) ✨
│   ├── gain_optimization.py           # 增益优化算法
│   ├── performance_evaluation.py      # 性能评估模块
│   ├── simulation.py                  # 仿真实验
│   ├── visualization.py               # 可视化工具
│   ├── examples.py                    # 使用示例
│   ├── experiment_loader.py           # 实验数据加载器
│   └── tools/                         # 辅助分析/可视化脚本
├── scripts/                           # 验证脚本
│   ├── verify_experiment.py           # 实验数据分析
│   ├── demo_demodulation.py           # OOK 解调演示 ✨
│   ├── validate_algorithm_on_real_data.py  # 单次优化验证
│   └── validate_iterative_algorithm.py     # 迭代优化验证
├── docs/                              # 文档
├── README.md                          # 项目说明
├── requirements.txt                   # 依赖包
├── .gitignore                         # Git忽略文件
└── results/                           # 结果输出
    ├── demodulation/                   # 解调可视化结果 ✨
    ├── experiment_verification/        # 实验数据验证结果
    ├── algorithm_validation/           # 算法性能验证结果
    └── iterative_validation/           # 迭代优化验证结果
```

## 安装依赖
```bash
pip install -r requirements.txt
pip install -e .
```

## 运行项目

### 仿真模式
```bash
# 运行基本仿真
occ-gain-opt

# 或使用Python直接运行
python -m occ_gain_opt.examples
```

### 完整可复现工作流

按顺序执行以下步骤，即可从原始实验数据复现全部结果：

```bash
# 0. 安装依赖
pip install -r requirements.txt && pip install -e .

# 1. OOK 解调 & 同步头检测 (验证信号解调能力)
python scripts/demo_demodulation.py

# 2. 单次增益优化验证 (使用 sync-based ROI)
python scripts/validate_algorithm_on_real_data.py

# 3. 迭代优化 vs 单次优化对比 (使用 sync-based ROI)
python scripts/validate_iterative_algorithm.py
```

### 实验数据验证

#### 1. OOK 解调演示
从条纹图像中检测同步头并定位完整数据包：
```bash
python scripts/demo_demodulation.py
```

**功能**:
- 检测 LED 列范围 (绿通道 Otsu 分割)
- 提取行均值曲线并二值化
- 检测同步头 (超长全亮段，~8 bit 连续1)
- 以同步头为锚点精确位采样 (周期 ~12.5 行/bit)
- 生成 6 面板可视化：原图+LED列范围、行均值曲线、二值化+采样点、比特序列、同步ROI叠加、统计信息
- 跨多种增益条件测试 (ISO=35/640/3200)

**输出**: [results/demodulation/](results/demodulation/)

#### 2. 算法性能验证
测试算法在真实数据上的预测准确性：
```bash
python scripts/validate_algorithm_on_real_data.py
```

**功能**:
- 使用 **sync-based ROI** 精确覆盖数据包区域 (检测失败自动退回 auto-brightness)
- 从最低增益开始模拟算法优化
- 对比预测值与实际最优值
- 在 6 种实验条件 (bubble/tap water/turbidity × ISO/Texp) 上验证

**输出**: [results/algorithm_validation/](results/algorithm_validation/)

#### 3. 迭代优化验证
对比单次计算和迭代优化两种策略：
```bash
python scripts/validate_iterative_algorithm.py
```

**功能**:
- 使用 **sync-based ROI** 精确覆盖数据包区域
- 实现迭代优化算法 (学习率α=0.5)
- 对比单次优化 vs 迭代优化
- 生成详细的对比图（包含图像对比、收敛曲线、误差分析等）

**输出**: [results/iterative_validation/](results/iterative_validation/)

## 主要功能

### OOK 解调 & 同步头检测
1. **LED 列检测**: 绿通道列均值 + Otsu 分割定位 LED 光源列范围
2. **行均值提取**: 在 LED 列范围内计算行均值 + 高斯平滑
3. **同步头检测**: 识别超长全亮游程 (≥5.5个位周期) 作为同步头
4. **精确位采样**: 以同步头为锚点对齐，消除累积相位漂移
5. **数据包定位**: 两个同步头之间的区域 = 一个完整数据包
6. **Sync-based ROI**: 精确覆盖数据包条纹区域，替代简单的最亮区域检测

**协议参数**: OOK 调制, 同步头 = 8 bit 全1, 数据 = 32 bit 随机序列 (p32), 位周期 ≈ 12.5 行/bit

### 仿真功能
1. **模拟相机响应**: 模拟不同增益设置下的图像采集
2. **增益优化**: 自动寻找最优增益值
3. **性能评估**: 计算MSE/PSNR/SSIM等指标
4. **可视化**: 展示优化过程和结果

## 算法效果图
![算法真实效果对比图](results/plots/algorithm_real_effect.png)

## 复现严格性说明
详见 `docs/REPRODUCTION_FIDELITY.md`。

## 真实数据集测试结果 (Sync-based ROI)

> 以下结果使用 **sync-based ROI** (基于同步头检测的精确数据包区域)，替代了早期的 auto-brightness ROI。

已在 ISO-Texp 真实数据集上执行验证。核心结果如下：

- 平均预测误差（6 组）：**增益误差 27.22 dB**、**灰度误差 155.07**
- 迭代优化在 **tap water/Texp** 条件下明显优于单次优化（改善率 16.8% vs 8.6%）
- 收敛通常只需 **1-4 次迭代**

**按实验汇总（目标灰度≈242.25）**：

| 实验 | 初始灰度 | 迭代优化灰度 / 误差 | 单次优化灰度 / 误差 | 更优方法 |
|---|---:|---:|---:|---|
| bubble/ISO | 65.96 | 56.43 / 185.82 | 56.43 / 185.82 | 相同 |
| bubble/Texp | 0.05 | 80.63 / 161.62 | 80.63 / 161.62 | 相同 |
| tap water/ISO | 86.05 | 86.05 / 156.20 | 86.05 / 156.20 | 相同 |
| tap water/Texp | 86.88 | 112.95 / 129.30 | 100.28 / 141.97 | **迭代** |
| turbidity/ISO | 0.05 | 116.72 / 125.53 | 116.72 / 125.53 | 相同 |
| turbidity/Texp | 81.98 | 82.98 / 159.27 | 82.98 / 159.27 | 相同 |

结果文件：
- `results/algorithm_validation/validation_report.txt`
- `results/algorithm_validation/algorithm_validation_summary.png`
- `results/iterative_validation/iterative_validation_report.txt`
- `results/iterative_validation/iterative_vs_single_summary.png`

### 验证功能
1. **OOK 解调器**: [demodulation.py](src/occ_gain_opt/demodulation.py)
   - 完整流水线: LED列检测 → 行均值 → 二值化 → 同步头检测 → 位采样 → 数据包提取
   - 支持 `OOKDemodulator.demodulate(image)` / `get_packet_roi_mask(image)` / `get_signal_quality(image)`
   - 输出 `DemodulationResult` 数据类，含行曲线、比特序列、同步位置、ROI掩码、置信度等

2. **ROI 策略**: [data_acquisition.py](src/occ_gain_opt/data_acquisition.py)
   - `CENTER`: 图像中心固定区域
   - `AUTO_BRIGHTNESS`: 自动检测最亮区域
   - `SYNC_BASED`: 基于同步头的精确数据包 ROI (新增)
   - 验证脚本中自动 fallback: SYNC_BASED → AUTO_BRIGHTNESS

3. **单次优化验证**: 基于论文公式的一次性预测
4. **迭代优化验证**: 渐进式优化策略 (α=0.5, 1-4次收敛)

## 验证结果总结

### 实验数据集
- **Bubble**: 气泡实验 (72张ISO + 42张Texp)
- **Tap Water**: 自来水实验 (42张ISO + 24张Texp)
- **Turbidity**: 浑浊度实验 (18张ISO + 36张Texp)
- **总计**: 234张实验图片

### 主要发现

1. **Sync-based ROI 有效性** ✅
   - 同步头检测在 ISO=35 ~ ISO=3200 全增益范围内稳定工作 (置信度 1.000)
   - 精确定位数据包区域，替代简单的最亮区域检测
   - 协议参数自动恢复: 同步头=8bit全1, 位周期≈12.5行/bit

2. **迭代 vs 单次** (Sync-based ROI)
   | 实验 | 迭代优化改善 | 单次优化改善 | 更优方法 |
   |-----|------------|------------|---------|
   | Bubble/ISO | -5.4% | -5.4% | 相同 |
   | Bubble/Texp | 33.3% | 33.3% | 相同 |
   | Tap Water/ISO | 0.0% | 0.0% | 相同 |
   | **Tap Water/Texp** | **16.8%** | 8.6% | **迭代** |
   | Turbidity/ISO | 48.2% | 48.2% | 相同 |
   | Turbidity/Texp | 0.6% | 0.6% | 相同 |

3. **推荐策略**
   - 使用 sync-based ROI 进行精确数据包定位
   - 迭代优化在多数场景下与单次相当，在 tap water/Texp 条件下明显更优

### 详细报告
- [算法性能分析](results/algorithm_validation/ALGORITHM_ANALYSIS.md)
- [迭代优化对比](results/iterative_validation/ITERATIVE_SUMMARY.md)

## 可视化示例

### 迭代优化对比图
每个实验生成包含8个子图的详细对比：
1. 初始图像 (ROI高亮)
2. 迭代优化结果
3. 单次优化结果
4. ROI灰度分布直方图
5. 迭代收敛曲线
6. 方法对比柱状图
7. 灰度误差分析
8. 详细统计信息

示例：[bubble_ISO_comparison.png](results/iterative_validation/bubble_ISO_comparison.png)

## 文档
- [仿真算法文档](docs/)
- [实验验证报告](results/README.md)
- [迭代优化总结](results/iterative_validation/ITERATIVE_SUMMARY.md)

## 技术栈
- **Python 3.8+**
- **NumPy**: 数值计算
- **SciPy**: 高斯平滑 (gaussian_filter1d)
- **OpenCV**: 图像处理、Otsu 分割、轮廓检测
- **Matplotlib**: 可视化（支持中文显示, macOS Hiragino Sans GB）
- **实验数据**: ISO-Texp/ (234张真实实验图片)

## 未来改进方向
1. **混合策略**: 先单次预测，再迭代微调
2. **饱和检测**: 实时监控，避免过优化
3. **场景识别**: 根据初始状态自动选择方法
4. **自适应学习率**: 根据误差大小动态调整α

## 许可证
本项目仅用于学术研究和教育目的。

## 致谢
- 感谢原作者Matus等人提出的优化算法
- 实验数据来源于ISO-Texp实验数据集

---
**最后更新**: 2026-02-27
**验证状态**: ✅ 已完成真实实验数据验证 (含 sync-based ROI)
