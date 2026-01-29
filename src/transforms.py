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
        
        # 4. Intensity Windowing (The HU logic we discussed)
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        
        # 5. Remove useless black space
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        # 6. Randomly crop a 96x96x96 cube for training
        # This ensures we get a balance of organ (pos) and background (neg)
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1, neg=1,
            num_samples=4, # Your 3090 can handle more, but we'll start here
            image_key="image",
            image_threshold=0,
        ),
        
        # 7. Convert to Tensor for the GPU
        EnsureTyped(keys=["image", "label"]),
    ])

def get_val_transforms():
    # Validation doesn't need random cropping
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
