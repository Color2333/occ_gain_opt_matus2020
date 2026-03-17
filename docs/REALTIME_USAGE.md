# 真实实验装置下的实时增益调整

在真实 OCC 实验装置下，算法可以**只负责根据当前帧计算“下一帧应设置的参数”**，由你在外部完成：**采集 → 设置相机参数 → 再采集** 的循环，从而实现实时调整。

## 算法返回的可调参数

当前接口返回的主要是**模拟增益 (dB)**，供你写入相机：

| 参数 | 含义 | 典型用法 |
|------|------|----------|
| `gain_db` | 建议的相机增益 (dB) | 直接设置到相机的 ISO/增益 控制（若相机用线性增益，需按厂商文档换算） |

若你的相机同时支持**曝光时间 (Exposure)**，本仓库目前未自动输出曝光建议，你可以在外层根据 `gain_db` 或 `mean_gray` 自行做简单策略（例如固定增益只调曝光，或先调曝光再调增益）。

### gain_db 对应哪个相机参数？

算法里的 **gain_db** 是论文中的**模拟增益 (analog gain)**，单位是 **dB**。在真实相机上对应什么，取决于你用的接口：

| 相机/接口 | 对应关系 | 用法 |
|-----------|----------|------|
| **增益控制，单位已是 dB** | 一一对应 | 直接把 `gain_db` 设给相机（注意项目里限制在 0～20 dB，可按你相机范围再 clamp）。 |
| **增益为线性倍率** | \(G_{linear} = 10^{G_{dB}/20}\) | 先换算成线性再设置，例如 OpenCV 部分设备：`cap.set(cv2.CAP_PROP_GAIN, 10**(gain_db/20))`（具体以相机文档为准）。 |
| **只有 ISO** | ISO 与线性增益成正比 | 选一个基准 ISO（如 100），则 `ISO ≈ 100 × 10^(gain_db/20)`，再取相机支持的离散档位（如 100, 200, 400, 800…）里最接近的值。ISO 每加倍约 +6 dB。 |
| **只有曝光时间** | 曝光与线性增益成正比 | 选一个基准曝光 \(T_0\)，则 `T = T_0 × 10^(gain_db/20)`，再按相机步长取整。 |
| **同时有 ISO 和曝光（如本仓库 ISO-Texp）** | 等效增益 = (Texp/基准Texp) × (ISO/基准ISO) | 本仓库用「等效增益 dB」统一描述：`gain_db = 20×log10((Texp/T0)×(ISO/ISO0))`。反过来：给定 `gain_db`，你可固定其一（如固定 ISO）只调另一个，或按策略分配两者。 |

**小结**：  
- 若相机有**模拟增益 / 总增益 (dB)** 控制，**gain_db 就对应这个参数**。  
- 若只有 **ISO**，则对应 **ISO**（通过上面的公式或查表换算）。  
- 若只有 **曝光时间**，则对应 **曝光时间**。  
- 若两者都有，则 gain_db 对应的是 **ISO 与曝光共同决定的等效增益**，你需要把 gain_db 拆成“调 ISO”或“调曝光”或两者配合（和本仓库 `experiment_loader.ExperimentImage.calculate_equivalent_gain` 的逆过程）。

## 使用方式

### 方式一：单步函数（自己写循环）

每次拍一帧后，把**当前增益**和**当前图像**传给算法，得到**下一帧要用的增益**，再设回相机：

```python
from occ_gain_opt.realtime import compute_next_gain

# 假设你已有：当前增益 current_gain_db、当前帧 image（numpy BGR 或灰度）
params = compute_next_gain(
    current_gain_db,
    image,
    roi_strategy="sync_based",   # 或 "auto" / "center"
    use_adaptive=True,
    learning_rate=0.5,
    tolerance_gray=5.0,
)

# 把算法建议的增益设到相机
next_gain_db = params["gain_db"]
# your_camera.set_gain(next_gain_db)   # 按你的相机 API 实现

# 判断是否可停止
if params["converged"]:
    print("已收敛，当前 ROI 灰度接近目标")
else:
    print(f"建议下一帧增益: {next_gain_db:.2f} dB, 当前灰度: {params['mean_gray']:.1f}")
```

下一轮循环时：用 `next_gain_db` 作为新的 `current_gain_db`，再拍一帧，重复上述过程，直到 `params["converged"]` 或达到你设定的最大迭代次数。

### 方式二：控制器类（带状态）

若希望少写一点循环逻辑，可以用 `RealtimeGainController`，内部会记录当前增益和迭代次数：

```python
from occ_gain_opt.realtime import RealtimeGainController

controller = RealtimeGainController(
    initial_gain_db=0.0,
    roi_strategy="sync_based",
    use_adaptive=True,
    learning_rate=0.5,
    tolerance_gray=5.0,
    max_iterations=10,
)

# 在真实装置循环里：每次拍一帧 image
for _ in range(10):
    # image = your_camera.capture()   # 相机当前已是 controller.current_gain_db
    # params = controller.step(image)
    # your_camera.set_gain(params["gain_db"])
    # if params["converged"]:
    #     break
    params = controller.step(image)
    next_gain_db = params["gain_db"]
    if params["converged"]:
        break
```

新一次实验前可调用 `controller.reset(initial_gain_db=0.0)` 重置状态。

## 与真实相机的对接要点

1. **增益单位**  
   算法给出的是 **dB**。若相机 API 使用线性增益或 ISO 档位，需要你按相机文档做换算（例如 `linear = 10^(gain_db/20)` 或查表）。

2. **谁负责“采集”**  
   本仓库**不包含**相机驱动。你需要：
   - 用 OpenCV、PyAV、v4l2、厂商 SDK 等先得到当前帧 `image`（numpy 数组）；
   - 把当前使用的增益（dB 或你换算后的值）传给 `compute_next_gain` 或 `RealtimeGainController`。

3. **ROI 策略**  
   - **sync_based**：依赖 OOK 同步头，适合已有条纹/数据包场景，精度高。  
   - **auto**：自动找最亮区域，不依赖协议。  
   - **center**：固定中心区域，实现最简单。

4. **收敛与迭代**  
   - `tolerance_gray` 越小，收敛越“严格”，迭代可能更多。  
   - 真实装置中 1–4 次迭代内收敛较常见；若始终不收敛，可适当增大 `tolerance_gray` 或检查 ROI 是否对准 LED/数据区。

## 小结

- **可以实现**：算法只做“根据当前帧 + 当前增益 → 返回下一增益”，你在外部循环里**实时**：拍图 → 设参数 → 再拍图。  
- **返回的可调参数**：主要是 **`gain_db`**（已限制在配置的增益范围内）。  
- **扩展**：若你需要同时调节曝光时间，可在得到 `gain_db` 与 `mean_gray` 后，在外层按自己的策略计算曝光并设置到相机。
