"""Centralized configuration for spleen segmentation pipeline."""

import os

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "Task09_Spleen")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Model paths
MODEL_PATH = os.path.join(OUTPUTS_DIR, "best_model.pth")
ONNX_PATH = os.path.join(OUTPUTS_DIR, "model_spleen.onnx")
TENSORRT_ENGINE_PATH = os.path.join(OUTPUTS_DIR, "model_spleen_fp16.engine")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")

# Model architecture
SPATIAL_SIZE = (96, 96, 96)
ROI_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 4
NUM_CLASSES = 2

# UNet config
UNET_CHANNELS = (16, 32, 64, 128, 256)
UNET_STRIDES = (2, 2, 2, 2)
UNET_NUM_RES_UNITS = 2

# Data
HU_MIN = -57
HU_MAX = 164
SPACING = (1.5, 1.5, 2.0)
VAL_SPLIT = 9
