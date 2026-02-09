#!/usr/bin/env python3
"""Visualize a sample from the spleen dataset."""

import os
import glob
import nibabel as nib
import matplotlib.pyplot as plt

from spleen_seg.config import DATA_DIR, OUTPUTS_DIR


def visualize_sample():
    images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))

    if not images or not labels:
        print("Error: No images/labels found. Download the dataset first.")
        return

    img = nib.load(images[0]).get_fdata()
    label = nib.load(labels[0]).get_fdata()
    print(f"Image shape: {img.shape}, Label shape: {label.shape}")

    slice_idx = img.shape[2] // 2
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"CT Scan at Slice {slice_idx}")
    plt.imshow(img[:, :, slice_idx], cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Label (Spleen)")
    plt.imshow(label[:, :, slice_idx], cmap="gray")
    plt.axis("off")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUTS_DIR, "sample_exploration.png")
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    visualize_sample()
