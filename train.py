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
from albumentations.pytorch import ToTensorV2 # <--- MODIFICATION
from sklearn.model_selection import train_test_split # <--- MODIFICATION
import collections

# 配置参数
config = {
    "all_img_dir": "/mnt/workspace/data/images/training",
    "all_mask_dir": "/mnt/workspace/data/annotations/training",
    "test_img_dir": "/mnt/workspace/data/images/test",
    "model_save_path":"/mnt/workspace/data/best_model.pth", # <--- MODIFICATION: 保存最佳模型
    "batch_size": 4,
    "num_epochs": 30, # <--- MODIFICATION: 增加训练轮数
    "learning_rate": 0.001,
    "img_size": (256, 256), # <--- MODIFICATION: 增加图像尺寸
    "num_classes": 4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# 数据集定义
class SteelDataset(Dataset):
    # <--- MODIFICATION: 接收文件名列表而不是目录
    def __init__(self, img_dir, mask_dir, img_names, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_names = img_names

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        
        mask_name = os.path.splitext(self.img_names[idx])[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file {mask_path} not found!")
        mask = Image.open(mask_path).convert("L")

        image = np.array(image)
        mask = np.array(mask)
        mask = mask.astype(np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        # <--- MODIFICATION: ToTensorV2 会处理通道和类型，不需要手动转换
        return image, mask

# U-Net模型定义 (无需修改)
class DoubleConv(nn.Module):
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
        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)
    def forward(self, x):
        x1 = self.down1(x); x2 = self.pool(x1); x2 = self.down2(x2)
        x3 = self.pool(x2); x3 = self.down3(x3); x4 = self.pool(x3)
        x4 = self.down4(x4); x = self.up1(x4); x = torch.cat([x, x3], dim=1)
        x = self.conv1(x); x = self.up2(x); x = torch.cat([x, x2], dim=1)
        x = self.conv2(x); x = self.up3(x); x = torch.cat([x, x1], dim=1)
        x = self.conv3(x); logits = self.out_conv(x)
        return logits
        
# 训练函数 (重大修改)
# 训练函数 (已修复数据类型问题)
def train_model(train_loader, val_loader):
    model = UNet(n_channels=3, n_classes=config["num_classes"])
    model.to(config["device"])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10 

    for epoch in range(config["num_epochs"]):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(config["device"])
            # <--- MODIFICATION: 修正数据类型
            masks = masks.to(config["device"], dtype=torch.long) 

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(config["device"])
                # <--- MODIFICATION: 修正数据类型
                masks = masks.to(config["device"], dtype=torch.long)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)

        # --- 保存最佳模型并实现早停 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(config["model_save_path"]), exist_ok=True)
            torch.save(model.state_dict(), config["model_save_path"])
            print(f"Validation loss decreased. Saving model to {config['model_save_path']}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"No improvement in validation loss for {patience} epochs. Early stopping.")
                break

    print("Training completed.")

# 预测函数 (基本不变，但加载模型路径需要注意)
def predict_and_save():
    model = UNet(n_channels=3, n_classes=config["num_classes"])
    if not os.path.exists(config["model_save_path"]):
        raise FileNotFoundError(f"Model file {config['model_save_path']} not found! Please train the model first.")
        
    model.load_state_dict(torch.load(config["model_save_path"], map_location=config["device"]))
    model.to(config["device"])
    model.eval()

    test_images = os.listdir(config["test_img_dir"])
    
    # <--- MODIFICATION: 预测时也需要归一化
    transform = alb.Compose([
        alb.Resize(*config["img_size"]),
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    with torch.no_grad():
        for img_name in test_images:
            img_path = os.path.join(config["test_img_dir"], img_name)
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
            
            augmented = transform(image=image)
            image_tensor = augmented['image'].unsqueeze(0).to(config["device"])

            output = model(image_tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # 这里需要注意，预测出的mask大小是config["img_size"]，如果提交要求原图尺寸，需要缩放回去
            # pred_resized = cv2.resize(pred, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            
            np.save(f"result/{os.path.splitext(img_name)[0]}.npy", pred.astype(np.uint8))

    with zipfile.ZipFile("result.zip", "w") as zipf:
        for file in os.listdir("result"):
            if file.endswith(".npy"):
                zipf.write(os.path.join("result", file), file)
    print("Prediction results saved to result.zip")


# 主程序
if __name__ == "__main__":
    os.makedirs("result", exist_ok=True)
    
    # <--- MODIFICATION: 数据准备
    all_img_names = os.listdir(config["all_img_dir"])
    train_names, val_names = train_test_split(all_img_names, test_size=0.2, random_state=42)

    # <--- MODIFICATION: 定义训练和验证的数据增强
    train_transform = alb.Compose([
        alb.Resize(*config["img_size"]),
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.RandomRotate90(p=0.5),
        alb.RandomBrightnessContrast(p=0.2),
        alb.GaussNoise(p=0.2),
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = alb.Compose([
        alb.Resize(*config["img_size"]),
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # <--- MODIFICATION: 创建数据集和加载器
    train_dataset = SteelDataset(config["all_img_dir"], config["all_mask_dir"], train_names, transform=train_transform)
    val_dataset = SteelDataset(config["all_img_dir"], config["all_mask_dir"], val_names, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # 训练模型
    train_model(train_loader, val_loader)

    # 进行预测并保存结果
    predict_and_save()
