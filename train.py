import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import zipfile
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import cv2 # <--- 引入OpenCV用于后处理

# --- 配置参数 ---
config = {
    "all_img_dir": "/mnt/workspace/data/images/training",
    "all_mask_dir": "/mnt/workspace/data/annotations/training",
    "test_img_dir": "/mnt/workspace/data/images/test",
    "model_save_path":"/mnt/workspace/data/best_model.pth",
    "batch_size": 2,               # <--- 物理批大小，因为图像尺寸增大
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "img_size": (320, 320),        # <--- NEW: 增大图像尺寸
    "num_classes": 4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# --- 数据集与Dice Loss定义 (无变化) ---
class SteelDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_names, transform=None): self.img_dir = img_dir; self.mask_dir = mask_dir; self.transform = transform; self.img_names = img_names
    def __len__(self): return len(self.img_names)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx]); image = Image.open(img_path).convert("RGB")
        mask_name = os.path.splitext(self.img_names[idx])[0] + ".png"; mask_path = os.path.join(self.mask_dir, mask_name); mask = Image.open(mask_path).convert("L")
        image = np.array(image); mask = np.array(mask)
        if self.transform: augmented = self.transform(image=image, mask=mask); image = augmented["image"]; mask = augmented["mask"]
        return image, mask
class DiceLoss(nn.Module):
    def __init__(self, n_classes): super(DiceLoss, self).__init__(); self.n_classes = n_classes
    def _one_hot_encoder(self, input_tensor):
        tensor_list = [];
        for i in range(self.n_classes): temp_prob = input_tensor == i; tensor_list.append(temp_prob.unsqueeze(1))
        return torch.cat(tensor_list, dim=1)
    def _dice_loss(self, score, target):
        target = target.float(); smooth = 1e-5; intersect = torch.sum(score * target); y_sum = torch.sum(target * target); z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth); return 1 - loss
    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax: inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None: weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        loss = 0.0
        for i in range(0, self.n_classes): dice = self._dice_loss(inputs[:, i], target[:, i]); loss += dice * weight[i]
        return loss / self.n_classes

# --- 训练函数 (集成梯度累积) ---
def train_model(train_loader, val_loader):
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=config["num_classes"])
    model.to(config["device"])
    criterion_ce = nn.CrossEntropyLoss(); criterion_dice = DiceLoss(n_classes=config["num_classes"])
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"], eta_min=1e-6)
    
    # <--- NEW: 再次引入梯度累积以支持大尺寸图像 ---
    accumulation_steps = 2 # 虚拟批大小 = 2 * 2 = 4
    
    best_val_loss = float('inf'); epochs_no_improve = 0; patience = 15
    for epoch in range(config["num_epochs"]):
        model.train(); running_loss = 0.0; optimizer.zero_grad()
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(config["device"]); masks = masks.to(config["device"], dtype=torch.long)
            outputs = model(images)
            loss_ce = criterion_ce(outputs, masks); loss_dice = criterion_dice(outputs, masks)
            loss = 0.2 * loss_ce + 0.8 * loss_dice
            loss = loss / accumulation_steps
            loss.backward()
            running_loss += loss.item() * accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                optimizer.step(); optimizer.zero_grad()
        scheduler.step()
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(config["device"]); masks = masks.to(config["device"], dtype=torch.long)
                outputs = model(images); loss_ce = criterion_ce(outputs, masks); loss_dice = criterion_dice(outputs, masks)
                loss = 0.2 * loss_ce + 0.8 * loss_dice; val_loss += loss.item()
        val_loss /= len(val_loader)
        train_loss_epoch = running_loss / len(train_loader.dataset) * config["batch_size"]
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss; os.makedirs(os.path.dirname(config["model_save_path"]), exist_ok=True)
            torch.save(model.state_dict(), config["model_save_path"]); print(f"Validation loss decreased. Saving model...")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience: print(f"Early stopping."); break
    print("Training completed.")

# --- 预测函数 (集成后处理) ---
def predict_and_save():
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=config["num_classes"])
    model.load_state_dict(torch.load(config["model_save_path"], map_location=config["device"]))
    model.to(config["device"]); model.eval()
    transform = alb.Compose([alb.Resize(*config["img_size"]), alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
    test_images = os.listdir(config["test_img_dir"])
    with torch.no_grad():
        for img_name in test_images:
            img_path = os.path.join(config["test_img_dir"], img_name); image = np.array(Image.open(img_path).convert("RGB"))
            images_to_predict = [image, np.fliplr(image).copy(), np.flipud(image).copy(), np.flipud(np.fliplr(image)).copy()]
            predictions = []
            for i, img in enumerate(images_to_predict):
                image_tensor = transform(image=img)['image'].unsqueeze(0).to(config["device"]); output = model(image_tensor)
                if i == 1: output = torch.flip(output, dims=[3])
                if i == 2: output = torch.flip(output, dims=[2])
                if i == 3: output = torch.flip(torch.flip(output, dims=[3]), dims=[2])
                predictions.append(torch.softmax(output, dim=1))
            final_output = torch.mean(torch.stack(predictions, dim=0), dim=0)
            pred_mask = torch.argmax(final_output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            # <--- NEW: 后处理，移除小噪点 ---
            # 遍历每个缺陷类别（从1到3，因为0是背景）
            for class_id in range(1, config["num_classes"]):
                # 创建一个只包含当前类别像素的二值图
                class_mask = (pred_mask == class_id).astype(np.uint8)
                
                # 连通域分析
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, 4, cv2.CV_32S)
                
                # 遍历除背景外的所有连通域
                for i in range(1, num_labels):
                    # 如果连通域的面积小于阈值（例如20像素），则在原始预测图中将其移除
                    if stats[i, cv2.CC_STAT_AREA] < 20:
                        pred_mask[labels == i] = 0 # 归为背景
            
            np.save(f"result/{os.path.splitext(img_name)[0]}.npy", pred_mask)

    with zipfile.ZipFile("result.zip", "w") as zipf:
        for file in os.listdir("result"):
            if file.endswith(".npy"): zipf.write(os.path.join("result", file), file)
    print("Prediction with Post-Processing results saved to result.zip")

# --- 主程序 (无变化) ---
if __name__ == "__main__":
    os.makedirs("result", exist_ok=True)
    all_img_names = os.listdir(config["all_img_dir"])
    train_names, val_names = train_test_split(all_img_names, test_size=0.2, random_state=42)
    train_transform = alb.Compose([alb.Resize(*config["img_size"]), alb.HorizontalFlip(p=0.5), alb.VerticalFlip(p=0.5), alb.RandomRotate90(p=0.5), alb.RandomBrightnessContrast(p=0.2), alb.GaussNoise(p=0.2), alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
    val_transform = alb.Compose([alb.Resize(*config["img_size"]), alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
    train_dataset = SteelDataset(config["all_img_dir"], config["all_mask_dir"], train_names, transform=train_transform)
    val_dataset = SteelDataset(config["all_img_dir"], config["all_mask_dir"], val_names, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    train_model(train_loader, val_loader)
    predict_and_save()
