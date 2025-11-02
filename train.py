#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete optimized train+predict script for 200x200 steel-defect segmentation.
- Fixes albumentations ToTensorV2 mask type issue.
- Uses WeightedRandomSampler to upsample images containing rare classes (small patches).
- Uses Focal + Dice loss, TTA for inference, validation & early stopping.
"""

import os
import numpy as np
from PIL import Image
import zipfile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

# ----------------- 配置项（如需在 CPU 上运行，请调整 num_workers=0, batch_size=1） -----------------
config = {
    "train_img_dir": "/mnt/workspace/data/images/training",
    "train_mask_dir": "/mnt/workspace/data/annotations/training",
    "test_img_dir": "/mnt/workspace/data/images/test",
    "model_save_path": "/mnt/workspace/data/model_best.pth",
    "result_dir": "result",
    "batch_size": 4,            # CPU: 1 或 2
    "num_epochs": 40,
    "learning_rate": 1e-3,
    "img_size": (200, 200),     # 原始图像为 200x200，保持原始尺寸
    "num_classes": 4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "val_ratio": 0.1,
    "num_workers": 4,           # CPU: 0
    "patience": 6,              # 早停
    "pixel_count_sample_limit": None  # None: 全量统计；或设置为整数（如500）以加快统计（采样）
}
os.makedirs(config["result_dir"], exist_ok=True)

# ----------------- Dataset -----------------
class SteelDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # 筛选图片文件并排序保证稳定性
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
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        mask = np.array(Image.open(mask_path).convert("L")).astype(np.int64)

        # TODO: 若你的 mask 使用 255/128 等特殊编码，请在这里把它们映射到 0..C-1，例如：
        # mask[mask == 255] = 1

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # 可能是 torch.Tensor (ToTensorV2) 或 np.ndarray
            mask = augmented["mask"]     # 可能是 torch.Tensor (ToTensorV2) 或 np.ndarray

        # 兼容上面两种情况
        if isinstance(image, np.ndarray):
            image = transforms.ToTensor()(image)  # 如果没有 ToTensorV2，则用 torchvision 转
        # else: image 已经是 tensor（ToTensorV2），保持不变

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        elif isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask

# ----------------- UNet (可调 base channels) -----------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=4, base_c=64):
        super().__init__()
        self.inc = DoubleConv(n_channels, base_c)
        self.down1 = DoubleConv(base_c, base_c*2)
        self.down2 = DoubleConv(base_c*2, base_c*4)
        self.down3 = DoubleConv(base_c*4, base_c*8)
        self.pool = nn.MaxPool2d(2)

        self.up1t = nn.ConvTranspose2d(base_c*8, base_c*4, kernel_size=2, stride=2)
        self.up1 = DoubleConv(base_c*8, base_c*4)
        self.up2t = nn.ConvTranspose2d(base_c*4, base_c*2, kernel_size=2, stride=2)
        self.up2 = DoubleConv(base_c*4, base_c*2)
        self.up3t = nn.ConvTranspose2d(base_c*2, base_c, kernel_size=2, stride=2)
        self.up3 = DoubleConv(base_c*2, base_c)

        self.outc = nn.Conv2d(base_c, n_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.pool(x1); x2 = self.down1(x2)
        x3 = self.pool(x2); x3 = self.down2(x3)
        x4 = self.pool(x3); x4 = self.down3(x4)

        x = self.up1t(x4)
        x = torch.cat([x, x3], dim=1); x = self.up1(x); x = self.dropout(x)
        x = self.up2t(x)
        x = torch.cat([x, x2], dim=1); x = self.up2(x); x = self.dropout(x)
        x = self.up3t(x)
        x = torch.cat([x, x1], dim=1); x = self.up3(x)
        logits = self.outc(x)
        return logits

# ----------------- 损失函数：Focal + Dice -----------------
class FocalLoss(nn.Module):
    """
    稳健版 Focal Loss：
    - 若 self.ignore_index 为 None 则不传入 ignore_index（避免某些 torch 版本报错）
    - 自动把 weight 转到 logits 的 device
    - 支持 reduction='mean'/'sum'/None
    """
    def __init__(self, gamma=2.0, weight=None, reduction='mean', ignore_index=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # 可以是 None, list, np.array 或 torch.Tensor
        self.reduction = reduction
        self.ignore_index = ignore_index  # None 或 int

    def forward(self, logits, targets):
        # logits: (N, C, H, W); targets: (N, H, W)
        # 统一 weight 到 logits device（若有）
        weight = None
        if self.weight is not None:
            if isinstance(self.weight, torch.Tensor):
                weight = self.weight.to(logits.device)
            else:
                weight = torch.tensor(self.weight, dtype=torch.float32, device=logits.device)

        # 调用 cross_entropy：只有当 ignore_index 不是 None 时才传这个参数
        if self.ignore_index is None:
            ce = nn.functional.cross_entropy(logits, targets, weight=weight, reduction='none')
        else:
            ce = nn.functional.cross_entropy(logits, targets, weight=weight,
                                              reduction='none', ignore_index=int(self.ignore_index))

        # 计算 focal 项
        pt = torch.exp(-ce)                 # pt = exp(-CE)
        loss = ((1 - pt) ** self.gamma) * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 返回逐元素损失（shape 与 ce 相同）

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

class ComboLoss(nn.Module):
    def __init__(self, class_weight=None, alpha=1.0, beta=1.0):
        super().__init__()
        self.focal = FocalLoss(weight=class_weight)
        self.dice = DiceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        return self.alpha * self.focal(logits, targets) + self.beta * self.dice(logits, targets)

# ----------------- 指标：每类 IoU -----------------
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

# ----------------- 数据增强（消除警告：使用 Affine） -----------------
train_transform = A.Compose([
    A.Resize(*config["img_size"]),
    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=15, p=0.6),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ElasticTransform(p=0.2, alpha=1, sigma=50),  # 不使用 alpha_affine（避免警告）
    A.Normalize(),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

val_transform = A.Compose([
    A.Resize(*config["img_size"]),
    A.Normalize(),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

# ----------------- 像素统计（可采样） -----------------
def compute_class_pixel_freq(dataset, num_classes, sample_limit=None):
    """
    dataset: iterable dataset (returns image, mask)
    sample_limit: None 或 int — 如果为 int 则随机采样该数量样本用于统计（加速）
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    n = len(dataset)
    if sample_limit is None or sample_limit >= n:
        indices = range(n)
    else:
        # 随机采样索引（固定随机种子保证可重复性）
        rng = np.random.default_rng(0)
        indices = rng.choice(n, size=sample_limit, replace=False)

    for i in tqdm(indices, desc="Counting pixels"):
        img, mask = dataset[i]
        # mask 可能是 tensor 或 ndarray；把它变为 numpy
        if isinstance(mask, torch.Tensor):
            m = mask.cpu().numpy().ravel()
        else:
            m = np.array(mask).ravel()
        for c in range(num_classes):
            counts[c] += np.sum(m == c)
    counts = np.maximum(counts, 1.0)  # 防止 0
    freqs = counts / counts.sum()
    return freqs

# ----------------- 构造 WeightedRandomSampler（图级权重） -----------------
def make_sampler_for_subset(subset, num_classes, sample_limit=None):
    """
    subset: a Subset or list-like dataset (indexable)
    返回 WeightedRandomSampler：重采样含稀有类的图片
    """
    n = len(subset)
    # 为速度，在 subset 上做 pixel freq 采样统计（可设置 sample_limit）
    freqs = compute_class_pixel_freq(subset, num_classes, sample_limit=sample_limit)
    inv_freq = 1.0 / freqs  # 类越稀有，权越大

    weights = np.zeros(n, dtype=np.float32)
    for i in tqdm(range(n), desc="Make per-image weights"):
        _, mask = subset[i]
        if isinstance(mask, torch.Tensor):
            m = mask.cpu().numpy()
        else:
            m = np.array(mask)
        present = np.zeros(num_classes, dtype=np.float32)
        for c in range(num_classes):
            if np.any(m == c):
                present[c] = 1.0
        w = float((present * inv_freq).sum())
        if w == 0.0:
            w = 1.0
        weights[i] = w
    # WeightedRandomSampler 不要求权重归一化
    sampler = WeightedRandomSampler(weights=weights, num_samples=n, replacement=True)
    return sampler

# ----------------- 训练主流程 -----------------
def train_model():
    # 全量 dataset（带 train_transform，之后 split）
    full_dataset = SteelDataset(config["train_img_dir"], config["train_mask_dir"], transform=train_transform)
    n_total = len(full_dataset)
    print(f"Total training images: {n_total}")
    # 划分 train/val
    val_size = int(n_total * config["val_ratio"])
    train_size = n_total - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # 把 val_dataset 的 transform 替换为 val_transform
    val_dataset.dataset.transform = val_transform

    # 计算像素级类频率（以训练子集为主）
    print("Compute class pixel freq on training subset (can sample to speed up)...")
    sample_limit = config["pixel_count_sample_limit"]  # None 或 int
    # compute_class_pixel_freq 需要索引访问（我们的 Subset 支持）
    train_subset_for_count = train_dataset
    freqs = compute_class_pixel_freq(train_subset_for_count, config["num_classes"], sample_limit=sample_limit)
    class_weights = torch.tensor(1.0 / freqs, dtype=torch.float32).to(config["device"])
    print("Class freqs:", freqs)
    print("Class weights (inverse freq):", class_weights.cpu().numpy())

    # 构造 sampler（可选：如果想不用 sampler，直接把 sampler=None）
    print("Constructing WeightedRandomSampler to upsample images containing rare classes...")
    sampler = make_sampler_for_subset(train_dataset, config["num_classes"], sample_limit=sample_limit)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=sampler,
                              num_workers=config["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=config["num_workers"], pin_memory=True)

    # 模型 / 损失 / 优化器 / 学习率调度
    model = UNet(n_channels=3, n_classes=config["num_classes"], base_c=64).to(config["device"])
    criterion = ComboLoss(class_weight=class_weights, alpha=1.0, beta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
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
            pbar.set_postfix({'train_loss': running_loss / (pbar.n + 1)})

        # 验证
        model.eval()
        val_loss = 0.0
        iou_list = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(config["device"])
                masks = masks.to(config["device"])
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

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

        # 保存与早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config["model_save_path"])
            print(f"Saved best model to {config['model_save_path']}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training finished.")

# ----------------- 推理（TTA） -----------------
def predict_and_save():
    model = UNet(n_channels=3, n_classes=config["num_classes"], base_c=64).to(config["device"])
    if not os.path.exists(config["model_save_path"]):
        raise FileNotFoundError(f"Model file not found: {config['model_save_path']}")
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

            # 原预测 + 翻转 TTA（HF, VF）
            preds = torch.softmax(model(img_tensor), dim=1)
            hf = torch.flip(img_tensor, dims=[3])
            out_hf = model(hf)
            out_hf = torch.flip(out_hf, dims=[3])
            preds += torch.softmax(out_hf, dim=1)
            vf = torch.flip(img_tensor, dims=[2])
            out_vf = model(vf)
            out_vf = torch.flip(out_vf, dims=[2])
            preds += torch.softmax(out_vf, dim=1)
            preds = preds / 3.0

            pred_mask = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_pil = Image.fromarray(pred_mask)
            pred_pil = pred_pil.resize((orig_w, orig_h), resample=Image.NEAREST)
            np.save(os.path.join(config["result_dir"], os.path.splitext(img_name)[0] + ".npy"), np.array(pred_pil))

    # 打包结果
    with zipfile.ZipFile("result.zip", "w") as zf:
        for root, _, files in os.walk(config["result_dir"]):
            for f in files:
                zf.write(os.path.join(root, f), arcname=f)
    print("Saved predictions to result.zip")

# ----------------- 主流程 -----------------
if __name__ == "__main__":
    # 训练并预测
    train_model()
    predict_and_save()
