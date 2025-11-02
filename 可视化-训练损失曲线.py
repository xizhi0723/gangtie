import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

losses = [0.2810, 0.1697, 0.1350, 0.1183, 0.1071, 0.1018, 0.0953, 0.0936, 0.0895, 0.0895, 0.0830, 0.0823, 0.0822, 0.0809, 0.0783, 0.0765, 0.0769, 0.0744, 0.0756, 0.0725, 0.0720, 0.0708, 0.0709, 0.0689, 0.0682, 0.0692, 0.0699, 0.0651, 0.0648, 0.0659]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体
plt.rcParams['axes.unicode_minus'] = False
plt.plot(losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("训练损失曲线")
plt.legend()
plt.show()