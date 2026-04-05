import numpy as np
import matplotlib.pyplot as plt


def generate_mseq_with_header(baseband_out_length, fbconnection, header):
    s1 = len(fbconnection)  # 移位寄存器数量
    register1 = np.zeros(s1, dtype=int)  # 初始化移位寄存器
    register1[-1] = 1  # 设置最后一个寄存器的初值为1

    mseq = np.zeros(baseband_out_length, dtype=int)  # 初始化输出序列

    # 生成PRBS序列
    for i in range(baseband_out_length):
        newregister1 = np.zeros(s1, dtype=int)
        newregister1[0] = np.mod(np.sum(fbconnection * register1), 2)  # 新的寄存器状态
        newregister1[1:] = register1[:-1]  # 移位
        register1 = newregister1
        mseq[i] = register1[0]

    # 保存原始的 M 序列
    save_to_csv(f'Mseq_{baseband_out_length}_original.csv', mseq)

    # 在序列前添加数据头
    mseq_with_header = np.concatenate((header, mseq))  # 加上数据头

    # 重复三次
    mseq_with_header = np.tile(mseq_with_header, 3)  # 将序列重复三次

    # 保存带数据头并重复的 M 序列
    save_to_csv(f'Mseq_{baseband_out_length}_with_header.csv', mseq_with_header)

    return mseq_with_header, mseq  # 返回带数据头并重复的序列及原始序列


def save_to_csv(filename, sequence):
    # 保存生成的序列到 CSV 文件
    with open(filename, 'w') as f:
        f.write('data length,200\n')  # AWG波形长度16384
        f.write('frequency,160\n')  # 定义输出频率
        f.write('amp,2\n')  # 定义幅值
        f.write('offset,0\n')
        f.write('phase,0\n')
        f.write('xpos,value\n')  # CSV文件头部

        index = 1
        # 写入数据序列
        for i in range(len(sequence)):
            f.write(f'{index},{sequence[i]}\n')
            index += 1

    print(f'{filename} 已生成')


# 定义数据头（假设数据头是一个10位的二进制头部）
header = np.array([0, 1, 1, 1, 1, 1, 1, 0])  # 示例数据头（可以更改）
# header = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0])  # 示例数据头（可以更改）
#header = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
# 定义伪随机序列的反馈连接
#fbconnection1 = np.array([0, 1, 0, 0, 0, 1, 1])  # PRBS的 0823
fbconnection1 = np.array([0, 1, 0, 0, 0, 1, 1])

# 调用该函数生成不同长度的伪随机序列并添加头部，重复三次
baseband_out_length1 = 20
baseband_out_length2 = 32
baseband_out_length3 = 50
baseband_out_length4 = 64
baseband_out_length5 = 96
"""
baseband_out_length2 = 64
baseband_out_length3 = 96
baseband_out_length4 = 128
baseband_out_length5 = 160
"""
# 使用伪随机数生成不同长度的序列，并添加头部和重复
mseq1_with_header, mseq1 = generate_mseq_with_header(baseband_out_length1, fbconnection1, header)
mseq2_with_header, mseq2 = generate_mseq_with_header(baseband_out_length2, fbconnection1, header)
mseq3_with_header, mseq3 = generate_mseq_with_header(baseband_out_length3, fbconnection1, header)
mseq4_with_header, mseq4 = generate_mseq_with_header(baseband_out_length4, fbconnection1, header)
mseq5_with_header, mseq5 = generate_mseq_with_header(baseband_out_length5, fbconnection1, header)
# 可视化结果
plt.figure(figsize=(10, 6))

plt.subplot(5, 1, 1)
plt.step(np.arange(len(mseq1_with_header)), mseq1_with_header, where='post')
plt.title('M-sequence 1 with Header and Repeated')

plt.subplot(5, 1, 2)
plt.step(np.arange(len(mseq2_with_header)), mseq2_with_header, where='post')
plt.title('M-sequence 2 with Header and Repeated')

plt.subplot(5, 1, 3)
plt.step(np.arange(len(mseq3_with_header)), mseq3_with_header, where='post')
plt.title('M-sequence 3 with Header and Repeated')

plt.subplot(5, 1, 4)
plt.step(np.arange(len(mseq4_with_header)), mseq4_with_header, where='post')
plt.title('M-sequence 4 with Header and Repeated')

plt.subplot(5, 1, 5)
plt.step(np.arange(len(mseq5_with_header)), mseq5_with_header, where='post')
plt.title('M-sequence 5 with Header and Repeated')

plt.tight_layout()
plt.show()
