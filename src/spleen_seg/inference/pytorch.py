"""PyTorch inference backend."""

import torch

from ..config import ROI_SIZE, SW_BATCH_SIZE
from ..model import get_model


def create_pytorch_predictor(model_path, device=None):
    """Create a PyTorch predictor callable for spleen segmentation (model)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model
