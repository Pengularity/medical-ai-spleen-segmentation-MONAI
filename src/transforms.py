"""
Train/validation transforms for Task09 Spleen (CT).
HU reference: soft tissue ~[-57, 164]; air ~-1000; bone ~+300 to +1000.
"""
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    EnsureTyped,
)


def get_train_transforms():
    return Compose([
        # 1. Load image and label
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # 2. Resample to a consistent voxel size (1.5mm x 1.5mm x 2.0mm)
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        
        # 3. Standardize orientation to RAS (Right, Anterior, Superior)
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # 4. CT intensity window (HU) for soft tissue
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        
        # 5. Crop to foreground
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # 6. Random 96³ crop (balanced pos/neg)
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        EnsureTyped(keys=["image", "label"]),
    ])

def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])