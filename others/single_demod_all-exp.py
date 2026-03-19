"""
002-解调数据集图片，保存误码信息，同步头起始位置，采样率
【所有图片共用同一个标签文件】
"""

import numpy as np
from PIL import Image
import pandas as pd
import os
import csv

# -------------------- 同步头检测函数 --------------------
def find_sync(rr, head_len=8, max_head_len=100, max_len_diff=5, last_header_len=None):
    runs = []
    p = 0
    while p < len(rr):
        if rr[p] == 1:
            q = p
            while q < len(rr) and rr[q] == 1:
                q += 1
            length = q - p
            if head_len <= length <= max_head_len:
                runs.append((p, q - 1, length))
            p = q
        else:
            p += 1

    if not runs:
        raise ValueError("未检测到任何有效同步段")

    runs_sorted = sorted(runs, key=lambda x: x[2], reverse=True)

    if last_header_len is not None:
        filtered_runs = [r for r in runs_sorted if abs(r[2] - last_header_len) <= max_len_diff]
        if filtered_runs:
            runs_sorted = filtered_runs

    header1_start, header1_end, len1 = runs_sorted[0]
    header2_start, header2_end, len2 = None, None, None

    for run in runs_sorted[1:]:
        if abs(run[2] - len1) <= max_len_diff:
            header2_start, header2_end, len2 = run
            break

    if header2_start is not None:
        payload_start = min(header1_end, header2_end) + 1
        equ = round(abs(((len1 + len2) / (head_len * 2))))
        sync_header_start = min(header1_start, header2_start)
    else:
        payload_start = header1_end + 1
        equ = round(len1 / head_len)
        sync_header_start = header1_start

    return equ, payload_start, sync_header_start


# -------------------- 数据恢复函数 --------------------
def recover_data(rr, payload_start, equ_len):
    p = payload_start
    res = []
    for i in range(payload_start, len(rr) - 1):
        if rr[i + 1] != rr[i]:
            q = i + 1
            width = q - p
            cnt = round(width / equ_len)
            res.extend([rr[i]] * cnt)
            p = q
    return np.array(res)


# -------------------- 误码率评估 --------------------
def evaluate(tx, rx):
    tx = tx[:len(rx)]
    num_errors = np.sum(tx != rx)
    ber = num_errors / len(tx)
    return num_errors, ber


# -------------------- 阈值计算：三阶多项式拟合 --------------------
def polyfit_threshold(y, degree=3):
    x = np.arange(1, len(y) + 1)
    coeffs = np.polyfit(x, y, degree)
    return np.polyval(coeffs, x)


# -------------------- 主流程 --------------------
if __name__ == "__main__":

    # ================== 路径设置 ==================
    #image_dir = r"/media/pc/Seagate Basic/OCC_Exp/rolling/Green LED/water 3m/shiyan/0827/turbidity/Mg_1.5_bubble_1_2_2/35800_35_p64_Mg_1.5_bubble_1_2_2/uieb"
    label_csv_path =r"C:\Users\lenovo\Desktop\data\Mseq_16_original.csv"  
    dir = r"C:\Users\lenovo\Downloads"
    #image_dir= os.path.join(dir, "uieb_weight16_0")
    image_dir = r"C:\Users\lenovo\Downloads"
    csv_path = os.path.join(dir, "uieb_weight16_0.csv") # Overexposed_Len51_0.csv/Attenuated_Len51_0.csv/Blurred_Len51_0.csv

    df = pd.read_csv(label_csv_path, skiprows=5)
    input_bits = df.iloc[:, 1].to_numpy()
    print(f"✅ 已加载统一标签，比特长度 = {len(input_bits)}")

    # ================== 图片列表 ==================
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    all_bers = []
    results_for_csv = []

    # ================== 主循环 ==================
    for image_file in image_files:
        print(f"\n================== 正在处理图像: {image_file} ==================")

        img_gray = Image.open(os.path.join(image_dir, image_file)).convert('L')
        img = np.array(img_gray, dtype=np.float64)

        column = np.mean(img, axis=1)
        mean = np.mean(column)
        std = np.std(column)
        y = (column - mean) / std

        threshold = polyfit_threshold(y, degree=3)
        yy = y - threshold
        rr = (yy > 0).astype(int)

        try:
            equ, payload_start, sync_start = find_sync(rr)
        except ValueError as e:
            print(f"❌ 同步失败: {e}")
            results_for_csv.append([image_file, "同步失败", "", "", "", ""])
            continue

        res = recover_data(rr, payload_start, equ)
        res = res[1:len(input_bits) + 1]

        if len(res) == len(input_bits):
            num, ber = evaluate(input_bits, res)
            all_bers.append(ber)
            results_for_csv.append([image_file, "OK", sync_start, equ, num, ber])
            print(f"✅ BER = {ber:.6f} | Errors = {num}")
        else:
            print("⚠️ 恢复数据长度不足")
            results_for_csv.append([image_file, "数据长度不足", sync_start, equ, "", ""])

    # ================== 统计与保存 ==================
    success_count = len(all_bers)

    with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Status", "Sync_Start", "NPPS", "Errors", "BER"])
        writer.writerows(results_for_csv)

        if success_count:
            avg_ber = np.mean(all_bers)
            writer.writerow([])
            writer.writerow(["AVG_BER", "", "", "", "", avg_ber])
            writer.writerow(["成功解调数量", "", "", "", "", success_count])

    print("\n====================================")
    print(f"✅ 成功解调数量: {success_count} / {len(image_files)}")
    if success_count:
        print(f"✅ 平均 BER: {np.mean(all_bers):.6f}")
    print(f"📄 结果已保存到: {csv_path}")
    print("====================================")
