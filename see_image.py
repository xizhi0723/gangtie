import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import jaccard_score

# 加载预测结果和真实标签
pred = np.load(r"D:\desktop\result\000763.npy")  # 预测结果
mask = np.array(Image.open(r"D:\desktop\NEU_Seg-main\annotations\test\000763.png"))  # 真实标签

# 定义颜色映射函数
def label_to_color(mask):
    # 定义颜色映射
    color_map = {
        0: [0, 0, 0],  # 背景：黑色
        1: [255, 0, 0],  # 缺陷类别 1（夹杂物）：红色
        2: [0, 255, 0],  # 缺陷类别 2（补丁）：绿色
        3: [0, 0, 255]  # 缺陷类别 3（划痕）：蓝色
    }

    # 创建彩色图像
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            color_mask[i, j] = color_map[mask[i, j]]
    return color_mask

# 将预测结果转换为彩色图像
color_mask = label_to_color(pred)

# 加载原始图像
image = np.array(Image.open(r"D:\desktop\NEU_Seg-main\images\test\000763.jpg"))

# 可视化
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(15, 5))

# 原始图像
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("原始图像")

# 真实标签
plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="jet")
plt.title("真实标签")

# 预测结果（彩色）
plt.subplot(1, 3, 3)
plt.imshow(color_mask)
plt.title("预测结果（彩色）")

plt.tight_layout()
plt.show()

# 计算 IoU
iou = jaccard_score(mask.flatten(), pred.flatten(), average="macro")
print(f"IoU: {iou:.4f}")