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

# --- 配置参数 ---
config = {
    "all_img_dir": "/mnt/workspace/data/images/training",
    "all_mask_dir": "/mnt/workspace/data/annotations/training",
    "test_img_dir": "/mnt/workspace/data/images/test",
    "model_save_path":"/mnt/workspace/data/best_model.pth",
    "batch_size": 4,
    "num_epochs": 100,             # <--- MODIFICATION: 给予充足的训练时间
    "learning_rate": 0.001,
    "img_size": (256, 256),
    "num_classes": 4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# --- 数据集定义 ---
class SteelDataset(Dataset):
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
        mask = Image.open(mask_path).convert("L")

        image = np.array(image)
        mask = np.array(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        return image, mask

# --- Dice Loss 定义 ---
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        return torch.cat(tensor_list, dim=1)

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes

# --- 训练函数 ---
def train_model(train_loader, val_loader):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=config["num_classes"],
    )
    model.to(config["device"])
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=config["num_classes"])
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.1, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 15 # <--- MODIFICATION: 更有耐心地等待模型收敛

    for epoch in range(config["num_epochs"]):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(config["device"])
            masks = masks.to(config["device"], dtype=torch.long) 

            optimizer.zero_grad()
            outputs = model(images)
            
            loss_ce = criterion_ce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            
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
                masks = masks.to(config["device"], dtype=torch.long)
                outputs = model(images)
                
                loss_ce = criterion_ce(outputs, masks)
                loss_dice = criterion_dice(outputs, masks)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                
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


# --- 预测函数 ---
def predict_and_save():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=config["num_classes"],
    )
    
    if not os.path.exists(config["model_save_path"]):
        raise FileNotFoundError(f"Model file {config['model_save_path']} not found!")
        
    model.load_state_dict(torch.load(config["model_save_path"], map_location=config["device"]))
    model.to(config["device"])
    model.eval()

    test_images = os.listdir(config["test_img_dir"])
    
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
            
            np.save(f"result/{os.path.splitext(img_name)[0]}.npy", pred.astype(np.uint8))

    with zipfile.ZipFile("result.zip", "w") as zipf:
        for file in os.listdir("result"):
            if file.endswith(".npy"):
                zipf.write(os.path.join("result", file), file)
    print("Prediction results saved to result.zip")


# --- 主程序 ---
if __name__ == "__main__":
    os.makedirs("result", exist_ok=True)
    
    all_img_names = os.listdir(config["all_img_dir"])
    train_names, val_names = train_test_split(all_img_names, test_size=0.2, random_state=42)

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

    train_dataset = SteelDataset(config["all_img_dir"], config["all_mask_dir"], train_names, transform=train_transform)
    val_dataset = SteelDataset(config["all_img_dir"], config["all_mask_dir"], val_names, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    train_model(train_loader, val_loader)

    predict_and_save()
