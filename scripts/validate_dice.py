#!/usr/bin/env python3
"""Validate Dice score across PyTorch, ONNX Runtime, and TensorRT backends."""

import os
import sys
import glob
import argparse

import torch
import numpy as np
import onnxruntime as ort
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from spleen_seg.config import (
    DATA_DIR,
    MODEL_PATH,
    ONNX_PATH,
    ROI_SIZE,
    SW_BATCH_SIZE,
    NUM_CLASSES,
    TENSORRT_ENGINE_PATH,
)
from spleen_seg.data.transforms import get_val_transforms
from spleen_seg.inference.pytorch import create_pytorch_predictor
from spleen_seg.inference.onnx import create_onnx_predictor
from spleen_seg.inference.tensorrt import create_tensorrt_predictor


def get_val_files():
    images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]
    return data_dicts[-9:]


def compute_dice_pytorch(device):
    val_files = get_val_files()
    val_ds = CacheDataset(data=val_files, transform=get_val_transforms(), cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    model = create_pytorch_predictor(MODEL_PATH, device)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)
            val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, SW_BATCH_SIZE, model)
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            dice_metric(y_pred=val_outputs, y=val_labels)
    return dice_metric.aggregate().item()


def compute_dice_onnx(device, use_tensorrt=False):
    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"] if use_tensorrt else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in providers if p in ort.get_available_providers()]
    if not providers:
        providers = ["CPUExecutionProvider"]
    predictor = create_onnx_predictor(ONNX_PATH, use_tensorrt)

    val_files = get_val_files()
    val_ds = CacheDataset(data=val_files, transform=get_val_transforms(), cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)
            val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, SW_BATCH_SIZE, predictor)
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            dice_metric(y_pred=val_outputs, y=val_labels)
    return dice_metric.aggregate().item()


def compute_dice_tensorrt_engine(device, engine_path):
    predictor = create_tensorrt_predictor(engine_path)
    val_files = get_val_files()
    val_ds = CacheDataset(data=val_files, transform=get_val_transforms(), cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)
            val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, SW_BATCH_SIZE, predictor)
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            dice_metric(y_pred=val_outputs, y=val_labels)
    return dice_metric.aggregate().item()


def main():
    parser = argparse.ArgumentParser(description="Validate Dice across backends")
    parser.add_argument("--engine", default=TENSORRT_ENGINE_PATH, help="Path to TensorRT .engine file")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(ONNX_PATH):
        print(f"Error: ONNX not found at {ONNX_PATH}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    results = {}

    print("Computing Dice - PyTorch...")
    try:
        results["PyTorch"] = compute_dice_pytorch(device)
        print(f"  Dice: {results['PyTorch']:.4f}\n")
    except Exception as e:
        print(f"  Error: {e}\n")
        results["PyTorch"] = None

    print("Computing Dice - ONNX Runtime...")
    try:
        results["ONNX Runtime"] = compute_dice_onnx(device, use_tensorrt=False)
        print(f"  Dice: {results['ONNX Runtime']:.4f}\n")
    except Exception as e:
        print(f"  Error: {e}\n")
        results["ONNX Runtime"] = None

    print("Computing Dice - TensorRT (.engine)...")
    if os.path.exists(args.engine):
        try:
            results["TensorRT (.engine)"] = compute_dice_tensorrt_engine(device, args.engine)
            print(f"  Dice: {results['TensorRT (.engine)']:.4f}\n")
        except Exception as e:
            print(f"  Error: {e} (requires cupy: pip install cupy-cuda12x)\n")
            results["TensorRT (.engine)"] = None
    else:
        print(f"  Skipped: engine not found at {args.engine}\n")
        results["TensorRT (.engine)"] = None

    print("=" * 50)
    print("Dice Score Summary")
    print("=" * 50)
    for backend, dice in results.items():
        print(f"  {backend:25s}: {dice:.4f}" if dice is not None else f"  {backend:25s}: N/A")
    print()


if __name__ == "__main__":
    main()
