from monai.networks.nets import UNet
from monai.networks.layers import Norm
import torch

def get_model(device):
    """
    Returns a 3D UNet model configured for spleen segmentation.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,  # [Background, Spleen]
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,  # residual connections
        norm=Norm.BATCH,
    ).to(device)
    
    return model