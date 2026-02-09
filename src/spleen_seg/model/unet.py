"""3D UNet for spleen segmentation."""

import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm

from ..config import UNET_CHANNELS, UNET_STRIDES, UNET_NUM_RES_UNITS, NUM_CLASSES


def get_model(device=None):
    """Returns a 3D UNet configured for spleen segmentation."""
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=NUM_CLASSES,
        channels=UNET_CHANNELS,
        strides=UNET_STRIDES,
        num_res_units=UNET_NUM_RES_UNITS,
        norm=Norm.BATCH,
    )
    if device is not None:
        model = model.to(device)
    return model
