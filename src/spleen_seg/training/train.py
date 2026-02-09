"""Training loop for spleen segmentation."""

import os
import glob
import torch
import wandb
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

from ..config import DATA_DIR, MODEL_PATH, ROI_SIZE, SW_BATCH_SIZE, VAL_SPLIT, NUM_CLASSES
from ..model import get_model
from ..data import get_train_transforms, get_val_transforms
from .losses import get_loss_and_metrics, get_optimizer


def train():
    """Run training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": i, "label": l} for i, l in zip(train_images, train_labels)]
    train_files, val_files = data_dicts[:-VAL_SPLIT], data_dicts[-VAL_SPLIT:]

    train_ds = CacheDataset(data=train_files, transform=get_train_transforms(), cache_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=get_val_transforms(), cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    model = get_model(device)
    loss_function, dice_metric = get_loss_and_metrics()
    optimizer = get_optimizer(model.parameters())
    scaler = torch.amp.GradScaler("cuda")

    best_metric = -1
    max_epochs = 500

    if os.path.exists(MODEL_PATH):
        print(f"Loading checkpoint from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    else:
        print("No checkpoint found. Training from scratch.")

    if os.environ.get("WANDB_API_KEY"):
        wandb.login()
        wandb.init(project="spleen-segmentation", name="3d-unet-v1", resume="allow")
    else:
        wandb.init(project="spleen-segmentation", mode="disabled")
        print("W&B not configured. Training in local mode only.")

    print(f"Starting training on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}...")

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        wandb.log({"train_loss": avg_loss, "epoch": epoch})
        print(f"Epoch {epoch + 1}/{max_epochs} - Avg Loss: {avg_loss:.4f}")

        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    val_outputs = sliding_window_inference(
                        val_inputs, ROI_SIZE, SW_BATCH_SIZE, model
                    )
                    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
                    post_label = AsDiscrete(to_onehot=NUM_CLASSES)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                wandb.log({"val_dice": metric, "epoch": epoch})

                if metric > best_metric:
                    print(f"Validation Dice improved: {best_metric:.4f} -> {metric:.4f}")
                    best_metric = metric
                    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                    torch.save(model.state_dict(), MODEL_PATH)
                    print("Best model saved.")

    wandb.finish()
