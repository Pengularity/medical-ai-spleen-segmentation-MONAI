# 3D Spleen Segmentation with MONAI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.5+-8888ff.svg)](https://monai.io/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end 3D semantic segmentation of the spleen from abdominal CT scans using **MONAI** and **PyTorch**. This project implements a training and inference pipeline on the [Medical Segmentation Decathlon](https://medicaldecathlon.com/) Task09 Spleen dataset.

---

## Features

| Component | Description |
|-----------|-------------|
| **Architecture** | 3D UNet with residual units (MONAI) |
| **Loss** | Dice Loss (class-imbalance friendly) |
| **Optimization** | Adam, Automatic Mixed Precision (AMP) |
| **Inference** | Sliding-window 3D prediction |
| **Monitoring** | [Weights & Biases](https://wandb.ai/) (optional) |
| **Post-processing** | Resample predictions to original image geometry |

---

## Results & Performance

### Training metrics (Weights & Biases)

Training and validation curves logged with W&B:

![W&B results](results/features/w&b-results.png)

**Final Dice Metric:** `0.93539`
**Inference Time:** < 1s on GPU (RTX 3090).

### Spleen segmentation preview

Animated 3D view of a predicted spleen mask from the test set:

![Spleen segmentation GIF](results/features/monai-spleen-gif.gif)

---

## Dataset

- **Source:** [Medical Segmentation Decathlon – Task09 Spleen](https://drive.google.com/file/d/1jzeB1kcFLWqxND9bRx7T4b65DNlBKnQN/view)
- **Modality:** CT (3D volumes)
- **Labels:** Binary mask (background / spleen)
- **Split:** Training/validation (script split); test set in `imagesTs/`

The download script fetches the dataset from the official MONAI/MSD mirror.

---

## Project Structure

```
medical-ai-spleen-segmentation-MONAI/
├── src/
│   ├── download_data.py   # Download Task09_Spleen
│   ├── explore_data.py   # Visualize samples (slices + save PNG)
│   ├── transforms.py     # Train/val data transforms (spacing, HU, crops)
│   ├── model.py         # 3D UNet (MONAI) definition
│   ├── train_utils.py   # Loss, metrics, optimizer
│   ├── train.py         # Training loop + W&B + checkpointing
│   ├── inference.py     # Sliding-window inference on imagesTs
│   └── post_process.py  # Resample prediction to original space
├── data/                 # Created by download_data.py
│   └── Task09_Spleen/
│       ├── imagesTr/    # Training images
│       ├── labelsTr/    # Training labels
│       └── imagesTs/    # Test images
├── outputs/              # Created at runtime
│   ├── best_model.pth   # Best checkpoint (val Dice)
│   └── predictions/    # NIfTI segmentations from inference
├── requirements.txt
├── check_gpu.py         # PyTorch + MONAI + CUDA check
└── README.md
```

---

## Installation

### 1. Clone and enter the project

```bash
git clone https://github.com/<username>/medical-ai-spleen-segmentation-MONAI.git
cd medical-ai-spleen-segmentation-MONAI
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` includes PyTorch with CUDA 12.1. For CPU-only or another CUDA version, install PyTorch from [pytorch.org](https://pytorch.org/) first, then `pip install -r requirements.txt`.

### 4. Verify GPU and stack

```bash
python check_gpu.py
```

---

## Usage

All commands below are from the project root. Run training/inference from the project root so that `src` is on `PYTHONPATH`, or use `python -m` as shown.

### 1. Download the dataset

```bash
python src/download_data.py
```

Data is extracted to `data/Task09_Spleen/`.

### 2. Explore the data (optional)

```bash
python src/explore_data.py
```

Writes a sample slice figure to `outputs/sample_exploration.png`.

### 3. Train the model

```bash
python src/train.py
```

- **Outputs:** `outputs/best_model.pth` (best model by validation Dice)  
- **Config:** 500 epochs, batch size 2, sliding-window validation every 2 epochs.  
- **W&B:** If the `WANDB_API_KEY` environment variable is set (or you ran `wandb login`), training metrics are sent to Weights & Biases. Otherwise, W&B runs in **disabled** mode and a warning is printed: *« W&B non configuré. Entraînement en mode local uniquement. »* — training continues normally without remote logging.  
  To enable W&B: `export WANDB_API_KEY=your_key` (or add it to a `.env` file and load it before running).

### 4. Run inference

```bash
python src/inference.py
```

Reads from `data/Task09_Spleen/imagesTs/`, uses `outputs/best_model.pth`, and writes NIfTI masks to `outputs/predictions/seg_*.nii.gz`.

### 5. Post-process a prediction (optional)

To resample a prediction back to the original image geometry and header:

```python
from post_process import resample_to_original

resample_to_original(
    "data/Task09_Spleen/imagesTs/spleen_7.nii.gz",
    "outputs/predictions/seg_spleen_7.nii.gz",
    "outputs/predictions/fixed_seg_spleen_7.nii.gz"
)
```

Or run the example in `post_process.py`:

```bash
python src/post_process.py
```

---

## Configuration Summary

| Item | Value |
|------|--------|
| Input size (train) | 96×96×96 (random crop) |
| Spacing | (1.5, 1.5, 2.0) mm |
| HU window | [-57, 164] (soft tissue) |
| Sliding window | 96×96×96, overlap 4 |
| Optimizer | Adam, lr=1e-4 |
| Checkpoint | Best validation Dice |

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [MONAI](https://monai.io/) for medical imaging utilities and 3D UNet
- [Medical Segmentation Decathlon](https://medicaldecathlon.com/) for the Task09 Spleen dataset
- [Weights & Biases](https://wandb.ai/) for experiment tracking
