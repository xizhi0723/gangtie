import os
import zipfile
import numpy as np
import io
import sys
import json
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_iou(pred, gt, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        gt_cls = (gt == cls)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union

        ious.append(iou)
    return ious


def extract_zip_to_memory(zip_path, is_gt=False):
    file_data = {}
    with open(zip_path, 'rb') as f:
        with zipfile.ZipFile(io.BytesIO(f.read())) as zip_ref:
            for file_name in zip_ref.namelist():
                base_name = os.path.splitext(os.path.basename(file_name))[0]

                if is_gt and file_name.endswith('.png'):
                    with zip_ref.open(file_name) as file:
                        img = Image.open(io.BytesIO(file.read()))
                        file_data[base_name] = np.array(img)
                elif not is_gt and file_name.endswith('.npy'):
                    with zip_ref.open(file_name) as file:
                        data = np.load(io.BytesIO(file.read()), allow_pickle=True)
                        file_data[base_name] = data
    return file_data


def check_class_distribution(data, num_classes, name):
    print(f"\n{name} 类别分布:")
    for cls in range(num_classes):
        count = np.sum([np.sum(arr == cls) for arr in data.values()])
        print(f"类别 {cls}: {count} 像素")


def seg(pred_data, gt_data, num_classes):
    common_files = set(pred_data.keys()) & set(gt_data.keys())
    assert common_files, "没有匹配的文件"

    all_ious = []
    for file_name in sorted(common_files):
        pred = pred_data[file_name]
        gt = gt_data[file_name]

        if pred.shape != gt.shape:
            raise ValueError(f"文件 {file_name} 形状不匹配: 预测 {pred.shape} vs 真实 {gt.shape}")

        pred = pred.astype(np.int32)
        gt = gt.astype(np.int32)

        ious = calculate_iou(pred, gt, num_classes)
        all_ious.append(ious)
        print(f"文件 {file_name} 的 IoU: {ious}")

    return np.mean(all_ious, axis=0)


def plot_iou_results(mean_ious, class_names):
    # 绘制 IoU 结果的柱状图
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(8, 6))

    # 设置每个类别的颜色
    color_map = {'夹杂物': '#FFCCCC', '补丁': '#CCFFCC', '划痕': '#CCE5FF'}  # 红色、绿色、蓝色
    colors = [color_map[class_name] for class_name in class_names]

    bars = plt.bar(class_names, mean_ious, color=colors)
    plt.title("各类别的平均 IoU", fontsize=16)
    plt.xlabel("类别", fontsize=14)
    plt.ylabel("IoU", fontsize=14)
    plt.ylim(0, 1.0)  # IoU 范围是 [0, 1]

    # 为每个柱子添加值的标签
    for bar, iou in zip(bars, mean_ious):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{iou:.2f}', ha='center', va='bottom', fontsize=12)

    plt.show()

def eval():
    pred_zip = r"D:\desktop\result.zip"
    gt_zip = r"D:\desktop\test_annotations.zip"

    pred_data = extract_zip_to_memory(pred_zip, is_gt=False)
    gt_data = extract_zip_to_memory(gt_zip, is_gt=True)

    print(f"预测文件数: {len(pred_data)} 真实文件数: {len(gt_data)}")
    print("示例匹配文件:", list(pred_data.keys())[:3])

    check_class_distribution(pred_data, num_classes=3, name="预测数据")
    check_class_distribution(gt_data, num_classes=3, name="真实标签")

    mean_ious = seg(pred_data, gt_data, num_classes=3)
    score = np.sum(mean_ious) * 100

    # 可视化 IoU 结果
    class_names = ["夹杂物", "补丁", "划痕"]
    plot_iou_results(mean_ious, class_names)

    return {
        "score": round(score, 2),
        "errorMsg": "success",
        "code": 0,
        "data": [{"score": round(score, 2)}]
    }


if __name__ == '__main__':
    result = eval()
    print(json.dumps(result))