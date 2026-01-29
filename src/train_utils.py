from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import torch

def get_loss_and_metrics():
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    return loss_function, dice_metric

def get_optimizer(model_params, learning_rate=1e-4):
    return torch.optim.Adam(model_params, learning_rate)