#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于你原始代码的保守优化版：
- 保持 UNet 结构不变（尽量与原始 266 分方案一致）
- 使用 class-weighted CrossEntropy + DiceLoss（小权重）
- 轻量增强，验证集与 TTA 推理
- 兼容 albumentations ToTensorV2 的 mask 类型
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import zipfile
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import random

# ========== 配置参数（保留与你原来一致或很接近） ==========
config = {
    "train_img_dir": "/mnt/workspace/data/images/training",
    "train_mask_dir": "/mnt/workspace/data/annotations/training",
    "test_img_dir": "/mnt/workspace/data/images/test",
    "model_save_path": "/mnt/workspace/data/model.pth",
    "batch_size": 4,
    "num_epochs": 30,
    "learning_rate": 0.001,
    "img_size": (200, 200),
    "num_classes": 4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "val_ratio": 0.1,
    "num_workers": 4,   # CPU: 改为 0
    "patience": 10,     # early stopping patience 放宽
    "dice_weight": 0.6, # Dice 在总损失中的权重（经验值）
    "seed": 42
}

os.makedirs("result", exist_ok=True)

# ========== 固定随机种子 ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(config["seed"])

# ========== Dataset（兼容 ToTensorV2 返回 mask 为 Tensor 的情况） ==========
class SteelDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))

        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file {mask_path} not found: {mask_path}")
        mask = np.array(Image.open(mask_path).convert("L")).astype(np.int64)

        # 如果 mask 有特殊编码（例如 255 表示缺陷），在这里做映射（根据你数据调整）
        # mask[mask == 255] = 1

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 兼容 image/mask 可能为 np.ndarray 或 torch.Tensor
        if isinstance(image, np.ndarray):
            image = transforms.ToTensor()(image)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        elif isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask

# ========== 保留你原始 UNet 与 DoubleConv（不改结构） ==========
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
        x1 = self.down1(x)
        x2 = self.pool(x1); x2 = self.down2(x2)
        x3 = self.pool(x2); x3 = self.down3(x3)
        x4 = self.pool(x3); x4 = self.down4(x4)

        x = self.up1(x4); x = torch.cat([x, x3], dim=1); x = self.conv1(x)
        x = self.up2(x); x = torch.cat([x, x2], dim=1); x = self.conv2(x)
        x = self.up3(x); x = torch.cat([x, x1], dim=1); x = self.conv3(x)

        logits = self.out_conv(x)
        return logits

# ========== 损失：Weighted CE + Dice（Dice 权重较小） ==========
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        N, C, H, W = probs.shape
        with torch.no_grad():
            target_onehot = torch.zeros_like(probs)
            target_onehot.scatter_(1, targets.unsqueeze(1), 1)
        dims = (0, 2, 3)
        inter = torch.sum(probs * target_onehot, dims)
        union = torch.sum(probs + target_onehot, dims)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        return loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, dice_w=0.6, ce_w=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss()
        self.dice_w = dice_w
        self.ce_w = ce_w

    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.ce_w * loss_ce + self.dice_w * loss_dice

# ========== 指标：每类 IoU ==========
def compute_iou_per_class(pred, target, num_classes):
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        inter = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(inter / union)
    return ious

# ========== 数据增强（保守） ==========
train_transform = A.Compose([
    A.Resize(*config["img_size"]),
    A.Affine(translate_percent=0.03, scale=(0.95, 1.05), rotate=10, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.3, brightness_limit=0.15, contrast_limit=0.15),
    A.Normalize(),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

val_transform = A.Compose([
    A.Resize(*config["img_size"]),
    A.Normalize(),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

# ========== 计算类别权重（像素级别） ==========
def compute_class_weights(dataset, num_classes, sample_limit=None):
    counts = np.zeros(num_classes, dtype=np.float64)
    n = len(dataset)
    if sample_limit is None or sample_limit >= n:
        indices = range(n)
    else:
        rng = np.random.default_rng(config["seed"])
        indices = rng.choice(n, size=sample_limit, replace=False)
    for i in tqdm(indices, desc="Counting pixels for class weights"):
        img, mask = dataset[i]
        # mask may be tensor
        if isinstance(mask, torch.Tensor):
            m = mask.cpu().numpy().ravel()
        else:
            m = np.array(mask).ravel()
        for c in range(num_classes):
            counts[c] += np.sum(m == c)
    counts = np.maximum(counts, 1.0)
    freqs = counts / counts.sum()
    # inverse freq normalized
    weights = (1.0 / freqs)
    weights = weights / np.sum(weights) * num_classes
    return torch.tensor(weights, dtype=torch.float32)

# ========== 训练函数 ==========
def train_model():
    # 数据集与划分
    full_dataset = SteelDataset(config["train_img_dir"], config["train_mask_dir"], transform=train_transform)
    total = len(full_dataset)
    val_size = int(total * config["val_ratio"])
    train_size = total - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # set val transform
    val_dataset.dataset.transform = val_transform

    # 计算类别权重（只在 train 子集上统计，sample_limit 可加快）
    print("计算类别权重（像素级别）...")
    weights = compute_class_weights(train_dataset, config["num_classes"], sample_limit=1000).to(config["device"])
    print("class weights:", weights.cpu().numpy())

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=config["num_workers"], pin_memory=True)

    # 模型、损失、优化器
    model = UNet(n_channels=3, n_classes=config["num_classes"]).to(config["device"])
    criterion = CombinedLoss(weight=weights, dice_w=config["dice_weight"], ce_w=1.0)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for images, masks in pbar:
            images = images.to(config["device"])
            masks = masks.to(config["device"])

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(train_loss=running_loss / (pbar.n + 1))

        # 验证
        model.eval()
        val_loss = 0.0
        iou_list = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(config["device"])
                masks = masks.to(config["device"])
                outputs = model(images)
                loss_v = criterion(outputs, masks)
                val_loss += float(loss_v.item())

                pred = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
                gt = masks.squeeze().cpu().numpy()
                ious = compute_iou_per_class(pred, gt, config["num_classes"])
                iou_list.append(ious)

        val_loss /= len(val_loader)
        iou_arr = np.array(iou_list, dtype=np.float32)
        mean_iou_per_class = np.nanmean(iou_arr, axis=0)
        mean_iou = float(np.nanmean(mean_iou_per_class))

        print(f"Epoch {epoch+1}: TrainLoss={running_loss/len(train_loader):.4f} ValLoss={val_loss:.4f} MeanIoU={mean_iou:.4f}")
        print("Per-class IoU:", mean_iou_per_class)

        scheduler.step(val_loss)

        # 保存最佳模型并早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config["model_save_path"])
            print("Saved best model.")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training finished.")

# ========== 预测（含简单 TTA 翻转平均） ==========
def predict_and_save():
    model = UNet(n_channels=3, n_classes=config["num_classes"]).to(config["device"])
    if not os.path.exists(config["model_save_path"]):
        raise FileNotFoundError(f"Model file {config['model_save_path']} not found!")
    model.load_state_dict(torch.load(config["model_save_path"], map_location=config["device"]))
    model.eval()

    preprocess = A.Compose([A.Resize(*config["img_size"]), A.Normalize(), ToTensorV2()])
    test_images = sorted([f for f in os.listdir(config["test_img_dir"]) if f.lower().endswith(('.png','.jpg','.jpeg'))])

    with torch.no_grad():
        for img_name in tqdm(test_images, desc="Predicting"):
            img_path = os.path.join(config["test_img_dir"], img_name)
            orig = Image.open(img_path).convert("RGB")
            orig_w, orig_h = orig.size
            arr = np.array(orig)

            aug = preprocess(image=arr)
            img_tensor = aug["image"].unsqueeze(0).to(config["device"])

            preds = torch.softmax(model(img_tensor), dim=1)
            # HF
            hf = torch.flip(img_tensor, dims=[3])
            out_hf = model(hf)
            out_hf = torch.flip(out_hf, dims=[3])
            preds += torch.softmax(out_hf, dim=1)
            # VF
            vf = torch.flip(img_tensor, dims=[2])
            out_vf = model(vf)
            out_vf = torch.flip(out_vf, dims=[2])
            preds += torch.softmax(out_vf, dim=1)

            preds = preds / 3.0
            pred_mask = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            pred_pil = Image.fromarray(pred_mask)
            pred_pil = pred_pil.resize((orig_w, orig_h), resample=Image.NEAREST)
            np.save(f"result/{os.path.splitext(img_name)[0]}.npy", np.array(pred_pil))

    # 打包结果
    with zipfile.ZipFile("result.zip", "w") as zipf:
        for root, dirs, files in os.walk("result"):
            for file in files:
                zipf.write(os.path.join(root, file), file)
    print("Prediction results saved to result.zip")

# ========== 主程序 ==========
if __name__ == "__main__":
    os.makedirs("result", exist_ok=True)
    train_model()
    predict_and_save()

