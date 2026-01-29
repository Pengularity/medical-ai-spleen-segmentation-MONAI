from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import torch

def get_loss_and_metrics():
    # DiceLoss handles the class imbalance
    # to_onehot_y=True: converts labels to binary
    # softmax=True: ensures the model outputs probabilities
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    
    # DiceMetric is used for final evaluation during validation
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    return loss_function, dice_metric

def get_optimizer(model_params, learning_rate=1e-4):
    # Adam is a robust optimizer that adjusts the learning rate automatically
    return torch.optim.Adam(model_params, learning_rate)