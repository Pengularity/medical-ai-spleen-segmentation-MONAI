from monai.networks.nets import UNet
from monai.networks.layers import Norm
import torch

def get_model(device):
    """
    Returns a 3D UNet model configured for spleen segmentation.
    """
    model = UNet(
        spatial_dims=3,          # We are working with 3D volumes
        in_channels=1,           # One input (the CT scan intensities)
        out_channels=2,          # Two outputs: [Background, Spleen]
        channels=(16, 32, 64, 128, 256), # Number of filters per layer
        strides=(2, 2, 2, 2),    # Downsampling steps
        num_res_units=2,         # Adds residual connections for better learning
        norm=Norm.BATCH,         # Normalizes data to speed up training
    ).to(device)
    
    return model