import os
import torch
import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Orientationd, Spacingd, ScaleIntensityRanged, 
    Invertd, AsDiscrete, SaveImaged, EnsureTyped
)
from monai.data import Dataset, DataLoader, decollate_batch
from model import get_model # Importing your 3D UNet
import glob

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Define the same transforms used during validation
    test_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"]),
    ])

    # 2. Load the test images (we use the imagesTs folder)
    test_images = sorted(glob.glob(os.path.join("data", "Task09_Spleen", "imagesTs", "*.nii.gz")))
    test_data = [{"image": img} for img in test_images]
    test_ds = Dataset(data=test_data, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1)

    # 3. Load your trained model
    model = get_model(device)
    model.load_state_dict(torch.load("outputs/best_model.pth"))
    model.eval()

    # 4. Create an output directory
    output_dir = "outputs/predictions"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running inference on {len(test_images)} images...")

    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            inputs = batch_data["image"].to(device)
            
            # Use the same sliding window logic
            outputs = sliding_window_inference(inputs, (96, 96, 96), 4, model)
            
            # Convert probabilities to a discrete mask (0 or 1)
            outputs = [AsDiscrete(argmax=True)(i) for i in decollate_batch(outputs)]
            
            # Save the result as a NIfTI file
            image_name = os.path.basename(test_images[i])
            save_path = os.path.join(output_dir, f"seg_{image_name}")
            
            # Convert back to numpy to save
            seg_data = outputs[0].detach().cpu().numpy()[0] # Take first channel
            affine = nib.load(test_images[i]).affine
            nib.save(nib.Nifti1Image(seg_data.astype(np.float32), affine), save_path)
            
            print(f"Saved prediction for image {i}: {save_path}")

if __name__ == "__main__":
    run_inference()