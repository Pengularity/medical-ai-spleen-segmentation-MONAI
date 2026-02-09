"""Loss functions and optimizer for spleen segmentation."""

import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric


def get_loss_and_metrics():
    """Return Dice loss and Dice metric for training/validation."""
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    return loss_function, dice_metric


def get_optimizer(model_params, learning_rate=1e-4):
    """Return Adam optimizer."""
    return torch.optim.Adam(model_params, learning_rate)
