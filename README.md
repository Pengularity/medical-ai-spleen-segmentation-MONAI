# 3D Medical Segmentation - Spleen (MONAI)

## 🩺 Project Overview
This project implements a 3D segmentation pipeline for medical imaging using **MONAI** and **PyTorch**. The goal is to automate the segmentation of the spleen from abdominal CT scans.

## 🚀 Technical Stack
- **Architecture:** 3D UNet with Residual Units
- **Loss Function:** Dice Loss (optimized for class imbalance)
- **Hardware:** NVIDIA RTX 3090 (24GB VRAM)
- **Monitoring:** Weights & Biases (W&B)
- **Optimization:** Automatic Mixed Precision (AMP)

## 📈 Performance
- **Target Dice Score:** 0.90+
- **Current Training:** 600 Epochs convergence run

## 🛠️ How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Download data: `python src/download_data.py`
3. Train model: `python src/train.py`