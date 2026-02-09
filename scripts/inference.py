#!/usr/bin/env python3
"""Entry point for PyTorch inference on test images."""

import os
import glob
import torch
import nibabel as nib
import numpy as np
from monai.data import Dataset, DataLoader
from monai.transforms import decollate_batch, AsDiscrete
from monai.inferers import sliding_window_inference

from spleen_seg.config import DATA_DIR, MODEL_PATH, PREDICTIONS_DIR, ROI_SIZE, SW_BATCH_SIZE, NUM_CLASSES
from spleen_seg.data.transforms import get_val_transforms
from spleen_seg.inference.pytorch import create_pytorch_predictor


def _get_test_transforms():
    """Test transforms (image only, no label)."""
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        Spacingd,
        ScaleIntensityRanged,
        EnsureTyped,
    )
    from spleen_seg.config import HU_MIN, HU_MAX, SPACING

    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=SPACING, mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"]),
    ])


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTs", "*.nii.gz")))
    test_data = [{"image": img} for img in test_images]
    test_ds = Dataset(data=test_data, transform=_get_test_transforms())
    test_loader = DataLoader(test_ds, batch_size=1)

    model = create_pytorch_predictor(MODEL_PATH, device)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)

    print(f"Running inference on {len(test_images)} images...")
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            inputs = batch_data["image"].to(device)
            outputs = sliding_window_inference(inputs, ROI_SIZE, SW_BATCH_SIZE, model)
            outputs = [post_pred(o) for o in decollate_batch(outputs)]
            seg_data = outputs[0].detach().cpu().numpy()[0]
            image_name = os.path.basename(test_images[i])
            save_path = os.path.join(PREDICTIONS_DIR, f"seg_{image_name}")
            affine = nib.load(test_images[i]).affine
            nib.save(nib.Nifti1Image(seg_data.astype(np.float32), affine), save_path)
            print(f"Saved: {save_path}")


if __name__ == "__main__":
    run_inference()
