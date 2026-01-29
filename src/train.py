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

    # 2. Loaders (Using CacheDataset for speed)
    train_ds = CacheDataset(data=train_files, transform=get_train_transforms(), cache_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

    val_ds = CacheDataset(data=val_files, transform=get_val_transforms(), cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    # 3. Initialize Factory Components
    model = get_model(device)
    loss_function, dice_metric = get_loss_and_metrics()
    optimizer = get_optimizer(model.parameters())
    scaler = torch.cuda.amp.GradScaler() # For Mixed Precision (Speed)

    # 4. Initialize Weights & Biases
    wandb.init(project="spleen-segmentation", name="3d-unet-resunits-v1")

    # 5. Training Loop
    max_epochs = 100
    best_metric = -1
    
    print(f"Starting Training on {torch.cuda.get_device_name(0)}...")

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with torch.cuda.amp.autocast():
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
                    
                    # Prepare for metric
                    post_pred = AsDiscrete(argmax=True, to_onehot=2)
                    post_label = AsDiscrete(to_onehot=2)
                    
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                
                wandb.log({"val_dice": metric, "epoch": epoch})
                
                if metric > best_metric:
                    best_metric = metric
                    os.makedirs("outputs", exist_ok=True)
                    torch.save(model.state_dict(), "outputs/best_model.pth")
                    print(f"New Best Model! Dice: {metric:.4f}")

    wandb.finish()

if __name__ == "__main__":
    train()