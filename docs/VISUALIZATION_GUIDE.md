# 可视化图表说明

## 已生成的所有图表 (8个)

### 📊 原始实验图表 (5个)

1. **experiment_1_gain_sweep.png** (276 KB)
   - 内容: 不同背景光强下的增益-灰度响应曲线
   - 用途: 分析增益对灰度值的影响

2. **experiment_2_algorithm_comparison.png** (351 KB)
   - 内容: 基础算法 vs 自适应算法性能比较
   - 用途: 对比两种算法的优化得分、迭代次数、灰度误差

3. **experiment_3_noise_robustness.png** (161 KB)
   - 内容: 不同噪声水平下的算法鲁棒性
   - 用途: 分析噪声对精度的影响

4. **experiment_4_convergence_analysis.png** (392 KB)
   - 内容: 从不同初始增益的收敛特性分析
   - 用途: 研究算法收敛速度和增益调整幅度

5. **summary_report.png** (338 KB)
   - 内容: 综合实验报告
   - 用途: 整体概览所有实验结果

---

## 🎨 增强可视化图表 (3个) - 推荐!

### 1. enhanced_comprehensive.png (1.9 MB) ⭐⭐⭐⭐⭐

**综合效果展示图** - 一张图看懂所有关键信息

包含内容:
- ✅ 增益-灰度响应曲线 (带数据表格)
- ✅ 优化过程收敛动画 (双y轴)
- ✅ 4个不同增益的实拍图像 (0/7/14/20dB)
- ✅ 不同条件下的优化效果对比柱状图
- ✅ ROI灰度值分布直方图对比

**特点**:
- 信息密度高,一目了然
- 带详细数据标注
- 图像直观对比
- 适合报告和展示

---

### 2. gain_comparison_grid.png (4.0 MB) ⭐⭐⭐⭐⭐

**增益对比网格图** - 6种增益的完整对比

展示内容:
- ✅ 6个子图 (0/4/8/12/16/20 dB)
- ✅ 每个子图包含:
  - 实际图像
  - ROI统计信息 (均值、标准差、最小/最大、饱和比)
  - ROI红色边框标注
  - 颜色条

**特点**:
- 图像清晰,高分辨率 (300 DPI)
- 统计信息完整
- 便于对比不同增益的效果
- 文件较大但质量很高

---

### 3. optimization_process_detail.png (1.9 MB) ⭐⭐⭐⭐⭐

**优化过程详情图** - 完整展示优化算法如何工作

包含内容:
- ✅ 增益变化曲线 (每次迭代标注数值)
- ✅ 灰度值变化曲线 (带目标值参考线)
- ✅ 增益-灰度散点关系图 (颜色映射迭代次数)
- ✅ 3次迭代的实际图像对比

**特点**:
- 动态展示优化过程
- 每步都有详细标注
- 便于理解算法原理
- 适合教学和演示

---

## 📖 如何查看这些图表

### 方法1: 命令行
```bash
# macOS
open results/plots/enhanced_comprehensive.png

# Linux
xdg-open results/plots/enhanced_comprehensive.png

# Windows
start results/plots/enhanced_comprehensive.png
```

### 方法2: 在文件管理器中
直接打开 `results/plots/` 目录,双击查看PNG文件

### 方法3: 用Python查看
```python
from IPython.display import Image
Image(filename='results/plots/enhanced_comprehensive.png')
```

---

## 🎯 图表使用建议

### 用于论文/报告
推荐使用:
1. **enhanced_comprehensive.png** - 综合展示 (图1)
2. **gain_comparison_grid.png** - 增益对比 (图2)
3. **optimization_process_detail.png** - 算法原理 (图3)

### 用于演示/PPT
推荐使用:
1. **enhanced_comprehensive.png** - 一页展示所有信息
2. **experiment_1_gain_sweep.png** - 增益响应曲线
3. **experiment_2_algorithm_comparison.png** - 算法比较

### 用于教学
推荐使用:
1. **optimization_process_detail.png** - 算法工作原理
2. **gain_comparison_grid.png** - 增益效果对比
3. **experiment_4_convergence_analysis.png** - 收敛特性

---

## 📊 图表关键信息解读

### 增益响应曲线

**理想效果**:
```
0dB  → 低灰度值 (约50-70)
10dB → 中灰度值 (约100-140)
20dB → 高灰度值 (约200-250)
```

**我们当前的实现**:
```
0dB  → 65.2  ✓ 正常
10dB → 99.1  ✓ 正常
20dB → 206.4 ✓ 良好 (距离目标255还有差距)
```

### 优化过程

**收敛速度**:
- 平均2-3次迭代 ✓ 优秀
- 每次增益调整明显 ✓ 正常

**稳定性**:
- 对不同初始增益都能收敛 ✓ 稳定
- 对噪声不敏感 ✓ 鲁棒

---

## 📈 数据可视化亮点

### 1. 增益响应曲线 (enhanced_comprehensive.png)
- ✅ 清晰展示0-20dB的灰度值变化
- ✅ 标注最大值点和目标值线
- ✅ 附带数据表格,便于查读
- ✅ 蓝色填充增强视觉效果

### 2. 图像对比 (gain_comparison_grid.png)
- ✅ 6种增益的完整对比
- ✅ 每张图都有详细统计
- ✅ ROI红框清晰可见
- ✅ 高分辨率,适合放大查看

### 3. 优化动画 (optimization_process_detail.png)
- ✅ 双曲线展示增益和灰度变化
- ✅ 散点图显示优化轨迹
- ✅ 实际图像展示每步效果
- ✅ 数值标注详细

---

## 💡 下一步可视化建议

如果还需要其他可视化,可以考虑:

1. **3D可视化**: 增益×LED强度×背景光的三维图
2. **动画**: 优化过程的GIF动画
3. **交互式图表**: Plotly可交互图表
4. **热力图**: ROI像素级分布热力图
5. **视频**: 动态优化过程视频

---

## 🎨 图表规格

| 图表 | 分辨率 | DPI | 文件大小 | 颜色 |
|------|--------|-----|----------|------|
| enhanced_comprehensive | ~4470×2950 | 300 | 1.9 MB | 彩色 |
| gain_comparison_grid | 高清 | 300 | 4.0 MB | 灰度 |
| optimization_process_detail | ~4470×2950 | 300 | 1.9 MB | 彩色 |
| 原始图表5个 | 高清 | 300 | ~1.5 MB | 彩色 |

**所有图表都适合用于打印和演示!**

---

## 📝 总结

✅ **已生成8个高质量可视化图表**
- 5个原始实验图表
- 3个增强可视化图表
- 总大小约10 MB
- 分辨率300 DPI,适合出版

✅ **图表信息丰富,直观易懂**
- 增益响应清晰可见
- 优化过程完整展示
- 实际图像对比直观
- 数据标注详细完整

✅ **适用于多种场景**
- 学术论文
- 技术报告
- 演示PPT
- 教学材料

---

**查看路径**: `results/plots/`

**推荐优先查看**:
1. enhanced_comprehensive.png (综合概览)
2. gain_comparison_grid.png (增益对比)
3. optimization_process_detail.png (优化过程)
