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
from sklearn.model_selection import KFold # <--- 引入KFold
import segmentation_models_pytorch as smp
import gc # 引入垃圾回收

# --- 最终配置 ---
config = {
    "all_img_dir": "/mnt/workspace/data/images/training",
    "all_mask_dir": "/mnt/workspace/data/annotations/training",
    "test_img_dir": "/mnt/workspace/data/images/test",
    "model_save_dir":"/mnt/workspace/data/kfold_models/", # <--- 保存多个模型的目录
    "batch_size": 4,
    "num_epochs": 150,
    "learning_rate": 1e-4,
    "img_size": (256, 256),
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

# --- 训练函数 (基本无变化，仅修改保存路径) ---
def train_model(train_loader, val_loader, fold):
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=config["num_classes"])
    model.to(config["device"])
    criterion_ce = nn.CrossEntropyLoss(); criterion_dice = DiceLoss(n_classes=config["num_classes"])
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"], eta_min=1e-6)
    best_val_loss = float('inf'); epochs_no_improve = 0; patience = 20
    
    model_save_path = os.path.join(config["model_save_dir"], f"best_model_fold_{fold}.pth")

    for epoch in range(config["num_epochs"]):
        model.train(); running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(config["device"]); masks = masks.to(config["device"], dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(images); loss_ce = criterion_ce(outputs, masks); loss_dice = criterion_dice(outputs, masks)
            loss = 0.2 * loss_ce + 0.8 * loss_dice
            loss.backward(); optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(config["device"]); masks = masks.to(config["device"], dtype=torch.long)
                outputs = model(images); loss_ce = criterion_ce(outputs, masks); loss_dice = criterion_dice(outputs, masks)
                loss = 0.2 * loss_ce + 0.8 * loss_dice; val_loss += loss.item()
        val_loss /= len(val_loader)
        train_loss_epoch = running_loss / len(train_loader)
        print(f"Fold {fold} - Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path); print(f"Validation loss decreased. Saving model for fold {fold}...")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience: print(f"Early stopping for fold {fold}."); break
    print(f"Training for fold {fold} completed.")
    # 释放显存
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

# --- 预测函数 (大改，集成5个模型) ---
def predict_and_save():
    models = []
    for fold in range(5): # 假设我们跑了5折
        model_path = os.path.join(config["model_save_dir"], f"best_model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model for fold {fold} not found. Skipping.")
            continue
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=config["num_classes"])
        model.load_state_dict(torch.load(model_path, map_location=config["device"]))
        model.to(config["device"]); model.eval()
        models.append(model)
        print(f"Loaded model from fold {fold}")

    if not models:
        print("No models found for prediction. Exiting.")
        return

    transform = alb.Compose([alb.Resize(*config["img_size"]), alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
    test_images = os.listdir(config["test_img_dir"])
    with torch.no_grad():
        for img_name in test_images:
            img_path = os.path.join(config["test_img_dir"], img_name); image = np.array(Image.open(img_path).convert("RGB"))
            
            # 对每个模型都进行4向TTA
            all_fold_predictions = []
            for model in models:
                images_to_predict = [image, np.fliplr(image).copy(), np.flipud(image).copy(), np.flipud(np.fliplr(image)).copy()]
                tta_predictions = []
                for i, img in enumerate(images_to_predict):
                    image_tensor = transform(image=img)['image'].unsqueeze(0).to(config["device"]); output = model(image_tensor)
                    if i == 1: output = torch.flip(output, dims=[3])
                    if i == 2: output = torch.flip(output, dims=[2])
                    if i == 3: output = torch.flip(torch.flip(output, dims=[3]), dims=[2])
                    tta_predictions.append(torch.softmax(output, dim=1))
                # 平均当前模型的4次TTA结果
                model_pred = torch.mean(torch.stack(tta_predictions, dim=0), dim=0)
                all_fold_predictions.append(model_pred)

            # 平均所有模型的预测结果
            final_output = torch.mean(torch.stack(all_fold_predictions, dim=0), dim=0)
            pred_mask = torch.argmax(final_output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            np.save(f"result/{os.path.splitext(img_name)[0]}.npy", pred_mask)

    with zipfile.ZipFile("result.zip", "w") as zipf:
        for file in os.listdir("result"):
            if file.endswith(".npy"): zipf.write(os.path.join("result", file), file)
    print("Ensembled prediction results saved to result.zip")

# --- 主程序 (大改，引入K-Fold循环) ---
if __name__ == "__main__":
    os.makedirs("result", exist_ok=True)
    os.makedirs(config["model_save_dir"], exist_ok=True) # 创建保存模型的目录
    
    all_img_names = np.array(os.listdir(config["all_img_dir"]))
    
    # --- K-Fold 设置 ---
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # --- 训练循环 ---
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_img_names)):
        print(f"========== Starting Fold {fold} ==========")
        
        # --- 为当前折叠准备数据 ---
        train_names = all_img_names[train_idx]
        val_names = all_img_names[val_idx]
        
        train_transform = alb.Compose([alb.Resize(*config["img_size"]), alb.HorizontalFlip(p=0.5), alb.VerticalFlip(p=0.5), alb.RandomRotate90(p=0.5), alb.RandomBrightnessContrast(p=0.2), alb.GaussNoise(p=0.2), alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
        val_transform = alb.Compose([alb.Resize(*config["img_size"]), alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
        
        train_dataset = SteelDataset(config["all_img_dir"], config["all_mask_dir"], train_names, transform=train_transform)
        val_dataset = SteelDataset(config["all_img_dir"], config["all_mask_dir"], val_names, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        
        train_model(train_loader, val_loader, fold)

    # --- 预测 ---
    print("========== Starting Prediction with Ensembled Models ==========")
    predict_and_save()
