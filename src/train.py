import os
import glob
import torch
import wandb
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

# Importing your custom modules
from transforms import get_train_transforms, get_val_transforms
from model import get_model
from train_utils import get_loss_and_metrics, get_optimizer

def train():
    # 1. Setup Device and Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join("data", "Task09_Spleen")
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

    data_dicts = [{"image": i, "label": l} for i, l in zip(train_images, train_labels)]
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]

    # 2. Loaders
    train_ds = CacheDataset(data=train_files, transform=get_train_transforms(), cache_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=get_val_transforms(), cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    # 3. Initialize Components
    model = get_model(device)
    loss_function, dice_metric = get_loss_and_metrics()
    optimizer = get_optimizer(model.parameters())
    
    # Updated Scaler
    scaler = torch.amp.GradScaler('cuda') 

    # Loading & Best Metric Logic
    model_path = "outputs/best_model.pth"
    best_metric = -1 # Default
    
    if os.path.exists(model_path):
        print(f"--- 🧠 Loading existing knowledge from {model_path} ---")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        best_metric = -1
    else:
        print("--- 🆕 No previous model found. Starting from scratch. ---")

    # 4. Initialize Weights & Biases (optional: set WANDB_API_KEY to enable logging)
    if os.environ.get("WANDB_API_KEY"):
        wandb.login()
        wandb.init(project="spleen-segmentation", name="3d-unet-v1", resume="allow")
    else:
        wandb.init(project="spleen-segmentation", mode="disabled")
        print("--- ⚠️ W&B non configuré. Entraînement en mode local uniquement. ---")

    # 5. Training Loop
    max_epochs = 500
    # REMOVED: best_metric = -1 was here, causing the reset bug.
    
    print(f"Starting Training on {torch.cuda.get_device_name(0)}...")

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()

            # Updated Autocast
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        wandb.log({"train_loss": avg_loss, "epoch": epoch})
        print(f"Epoch {epoch+1}/{max_epochs} - Avg Loss: {avg_loss:.4f}")

        # 6. Validation every 2 epochs
        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
                    
                    post_pred = AsDiscrete(argmax=True, to_onehot=2)
                    post_label = AsDiscrete(to_onehot=2)
                    
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                
                wandb.log({"val_dice": metric, "epoch": epoch})
                
                if metric > best_metric:
                    print(f"--- 📈 Improvement detected: {best_metric:.4f} -> {metric:.4f} ---")
                    best_metric = metric
                    os.makedirs("outputs", exist_ok=True)
                    torch.save(model.state_dict(), model_path)
                    print(f"New Best Model Saved!")

    wandb.finish()

if __name__ == "__main__":
    train()