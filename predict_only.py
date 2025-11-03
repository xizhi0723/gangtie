import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import zipfile
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# --- 使用和训练时完全相同的配置 ---
config = {
    "test_img_dir": "/mnt/workspace/data/images/test",
    "model_save_dir":"/mnt/workspace/data/kfold_models/",
    "img_size": (256, 256),
    "num_classes": 4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# --- 预测函数 (它会自动加载所有存在的模型) ---
def predict_and_save():
    models = []
    # --- 关键改动：我们知道总共有5折 ---
    num_total_folds = 5 
    for fold in range(num_total_folds):
        model_path = os.path.join(config["model_save_dir"], f"best_model_fold_{fold}.pth")
        
        # 如果模型文件存在，我们就加载它
        if os.path.exists(model_path):
            print(f"Found and loading model for fold {fold}...")
            model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=config["num_classes"])
            model.load_state_dict(torch.load(model_path, map_location=config["device"]))
            model.to(config["device"])
            model.eval()
            models.append(model)
        else:
            print(f"Warning: Model for fold {fold} not found. Skipping.")

    if not models:
        print("No models found for prediction. Exiting.")
        return
        
    print(f"\nStarting ensembled prediction using {len(models)} models.")

    transform = alb.Compose([
        alb.Resize(*config["img_size"]),
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    test_images = os.listdir(config["test_img_dir"])

    with torch.no_grad():
        for img_name in test_images:
            img_path = os.path.join(config["test_img_dir"], img_name)
            image = np.array(Image.open(img_path).convert("RGB"))
            
            all_fold_predictions = []
            for model in models:
                images_to_predict = [image, np.fliplr(image).copy(), np.flipud(image).copy(), np.flipud(np.fliplr(image)).copy()]
                tta_predictions = []
                for i, img in enumerate(images_to_predict):
                    image_tensor = transform(image=img)['image'].unsqueeze(0).to(config["device"])
                    output = model(image_tensor)
                    if i == 1: output = torch.flip(output, dims=[3])
                    if i == 2: output = torch.flip(output, dims=[2])
                    if i == 3: output = torch.flip(torch.flip(output, dims=[3]), dims=[2])
                    tta_predictions.append(torch.softmax(output, dim=1))
                
                model_pred = torch.mean(torch.stack(tta_predictions, dim=0), dim=0)
                all_fold_predictions.append(model_pred)

            final_output = torch.mean(torch.stack(all_fold_predictions, dim=0), dim=0)
            pred_mask = torch.argmax(final_output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            np.save(f"result/{os.path.splitext(img_name)[0]}.npy", pred_mask)

    with zipfile.ZipFile("result.zip", "w") as zipf:
        for file in os.listdir("result"):
            if file.endswith(".npy"):
                zipf.write(os.path.join("result", file), file)
    print("Ensembled prediction results saved to result.zip")

# --- 主程序 ---
if __name__ == "__main__":
    os.makedirs("result", exist_ok=True)
    predict_and_save()
