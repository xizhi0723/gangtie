import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import zipfile
import albumentations as alb


# 配置参数
config = {
    "train_img_dir": r"D:\desktop\NEU_Seg-main\images\training",  # 训练集图像路径
    "train_mask_dir": r"D:\desktop\NEU_Seg-main\annotations\training",  # 训练集标签路径
    "test_img_dir": r"D:\desktop\NEU_Seg-main\images\test",  # 测试集图像路径
    "model_save_path":r"D:\desktop\model.pth",  # 模型保存路径
    "batch_size": 4,  # 批大小
    "num_epochs": 30,  # 训练轮数
    "learning_rate": 0.001,  # 学习率
    "img_size": (200, 200),  # 输入图像尺寸
    "num_classes": 4,  # 分类类别数（0-3）
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择
}


# 数据集定义
class SteelDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # 读取图像
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")

        # 生成标签文件路径
        mask_name = os.path.splitext(self.img_names[idx])[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 检查标签文件是否存在
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file {mask_path} not found!")

        # 读取标签
        mask = Image.open(mask_path).convert("L")  # 转为灰度图

        # 转换为numpy数组处理
        image = np.array(image)
        mask = np.array(mask)

        # 处理标签：将像素值转换为类别标签
        mask = mask.astype(np.int64)

        # 应用数据增强
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        # 转为Tensor并归一化
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask).long()

        return image, mask


# U-Net模型定义
class DoubleConv(nn.Module):
    """(卷积 => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=4):
        super(UNet, self).__init__()
        # 下采样路径
        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        # 上采样路径
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        # 最终卷积层
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 下采样
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x2 = self.down2(x2)
        x3 = self.pool(x2)
        x3 = self.down3(x3)
        x4 = self.pool(x3)
        x4 = self.down4(x4)

        # 上采样
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)

        # 输出
        logits = self.out_conv(x)
        return logits


# 训练函数
def train_model():
    # 初始化模型
    model = UNet(n_channels=3, n_classes=config["num_classes"])
    model.to(config["device"])

    # 数据增强
    transform = alb.Compose([
        alb.Resize(*config["img_size"]),
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.RandomRotate90(p=0.5),
    ])

    # 数据集和数据加载器
    dataset = SteelDataset(config["train_img_dir"],
                           config["train_mask_dir"],
                           transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=config["batch_size"],
                            shuffle=True)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # 训练循环
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0

        for images, masks in dataloader:
            images = images.to(config["device"])
            masks = masks.to(config["device"])

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Loss: {running_loss / len(dataloader):.4f}")

    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(config["model_save_path"]), exist_ok=True)

    # 保存模型参数
    torch.save(model.state_dict(), config["model_save_path"])
    print("Training completed. Model saved to", config["model_save_path"])


# 预测函数
def predict_and_save():
    # 初始化模型
    model = UNet(n_channels=3, n_classes=config["num_classes"])

    # 确保模型文件存在
    if not os.path.exists(config["model_save_path"]):
        raise FileNotFoundError(f"Model file {config['model_save_path']} not found!")

    # 加载模型参数
    model.load_state_dict(torch.load(config["model_save_path"], map_location=config["device"]))
    model.to(config["device"])
    model.eval()

    # 处理测试集
    test_images = os.listdir(config["test_img_dir"])

    # 存储预测结果
    all_preds = []

    with torch.no_grad():
        for img_name in test_images:
            # 读取图像
            img_path = os.path.join(config["test_img_dir"], img_name)
            image = Image.open(img_path).convert("RGB")

            # 预处理
            transform = transforms.Compose([
                transforms.Resize(config["img_size"]),
                transforms.ToTensor()
            ])
            image_tensor = transform(image).unsqueeze(0).to(config["device"])

            # 预测
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            # 保存为npy文件
            np.save(f"result/{os.path.splitext(img_name)[0]}.npy", pred)

    # 打包结果
    with zipfile.ZipFile("result.zip", "w") as zipf:
        for root, dirs, files in os.walk("result"):
            for file in files:
                zipf.write(os.path.join(root, file), file)
    print("Prediction results saved to result.zip")


# 主程序
if __name__ == "__main__":
    # 创建结果目录
    os.makedirs("result", exist_ok=True)

    # 训练模型
    train_model()

    # 进行预测并保存结果
    predict_and_save()