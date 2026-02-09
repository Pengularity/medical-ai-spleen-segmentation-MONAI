# 3D Spleen Segmentation with MONAI

A robust Deep Learning pipeline designed for clinical-grade spleen segmentation, bridging the gap between PyTorch research and C++ production environments

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.5+-8888ff.svg)](https://monai.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-24+-2496ed.svg)](https://www.docker.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.16+-00599C.svg)](https://onnxruntime.ai/)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.6+-76b900.svg)](https://developer.nvidia.com/tensorrt)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://isocpp.org/)
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
| **Export** | ONNX (Opset 12), TensorRT FP16 |
| **API** | [FastAPI](https://fastapi.tiangolo.com/) REST API for NIfTI upload → ONNX inference |
| **Deployment** | Docker image (Python 3.11-slim, Uvicorn on port 8000) |
| **Monitoring** | [Weights & Biases](https://wandb.ai/) (optional) |
| **Post-processing** | Resample predictions to original image geometry |

---

## Results & Performance

### ⚡ Performance Benchmarks

Measured on NVIDIA RTX 3090 using `trtexec` (C++ backend) to isolate model performance from Python overhead.

| Framework | Precision | Dice Score | Latency (GPU) | Throughput | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch** | FP32 | 0.9399 | ~5.16 ms* | ~194 qps | 1× |
| **TensorRT** | **FP16** | **0.9400** | **1.85 ms** | **541 qps** | **~2.8× (Python)** / **~35× (Pure C++)** |

*\*PyTorch latency includes Python runtime overhead. Pure TensorRT latency reveals the true hardware limit.*

### Training metrics (Weights & Biases)

Training and validation curves logged with W&B:

![W&B results](results/features/w&b-results.png)

**Final Dice Metric:** `0.93539`

### Spleen segmentation preview

Animated 3D view of a predicted spleen mask from the test set:

![Spleen segmentation GIF](results/features/monai-spleen-gif.gif)

---

## Dataset

- **Source:** [Medical Segmentation Decathlon – Task09 Spleen](https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar)
- **Modality:** CT (3D volumes)
- **Labels:** Binary mask (background / spleen)
- **Split:** Training/validation (script split); test set in `imagesTs/`

The download script fetches the dataset from the official MONAI/MSD mirror.

---

## Project Structure

```
medical-ai-spleen-segmentation-MONAI/
├── src/spleen_seg/         # Main package
│   ├── __init__.py
│   ├── config.py           # Centralized config
│   ├── data/
│   │   ├── dataset.py, download.py, transforms.py
│   ├── model/
│   │   └── unet.py
│   ├── training/
│   │   ├── train.py, losses.py
│   └── inference/
│       ├── pytorch.py, onnx.py, tensorrt.py, post_process.py
├── api/
│   └── app.py              # FastAPI: POST /predict
├── scripts/                # Entry points
│   ├── train.py, inference.py, explore_data.py
│   ├── export_onnx.py, export_tensorrt.sh
│   ├── validate_onnx.py, validate_dice.py, benchmark.py
│   └── check_gpu.py
├── cpp/
│   └── inference.cpp       # C++ ONNX Runtime inference
├── data/                   # Created by download script
│   └── Task09_Spleen/
├── results/                # Curated results (screenshots, GIFs)
├── outputs/                # Runtime: best_model.pth, model_spleen.onnx, etc.
├── Dockerfile
├── pyproject.toml, requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone and enter the project

```bash
git clone https://github.com/pengularity/medical-ai-spleen-segmentation-MONAI.git
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
pip install -e .
# Or: pip install -r requirements.txt
```

> **Note:** `requirements.txt` includes PyTorch with CUDA 12.1. For CPU-only or another CUDA version, install PyTorch from [pytorch.org](https://pytorch.org/) first, then `pip install -r requirements.txt`.

### 4. Verify GPU and stack

```bash
python scripts/check_gpu.py
```

---

## Usage

All commands below are from the project root. Install the package with `pip install -e .` so that `spleen_seg` is available.

### 1. Download the dataset

```bash
python -c "from spleen_seg.data.download import download_dataset; download_dataset()"
```

Data is extracted to `data/Task09_Spleen/`.

### 2. Explore the data (optional)

```bash
python scripts/explore_data.py
```

Writes a sample slice figure to `outputs/sample_exploration.png`.

### 3. Train the model

```bash
python scripts/train.py
```

- **Outputs:** `outputs/best_model.pth` (best model by validation Dice)  
- **Config:** 500 epochs, batch size 2, sliding-window validation every 2 epochs.  
- **W&B:** If the `WANDB_API_KEY` environment variable is set (or you ran `wandb login`), training metrics are sent to Weights & Biases. Otherwise, W&B runs in **disabled** mode and a warning is printed: *« W&B not configured. Training in local mode only »* — training continues normally without remote logging.  
  To enable W&B: `export WANDB_API_KEY=your_key` (or add it to a `.env` file and load it before running).

### 4. Run inference

```bash
python scripts/inference.py
```

Reads from `data/Task09_Spleen/imagesTs/`, uses `outputs/best_model.pth`, and writes NIfTI masks to `outputs/predictions/seg_*.nii.gz`.

### 5. Post-process a prediction (optional)

To resample a prediction back to the original image geometry and header:

```bash
python scripts/post_process.py \
  data/Task09_Spleen/imagesTs/spleen_7.nii.gz \
  outputs/predictions/seg_spleen_7.nii.gz \
  outputs/predictions/fixed_seg_spleen_7.nii.gz
```

Or in Python: `from spleen_seg.inference.post_process import resample_to_original`

### 6. Export to ONNX (optional)

After training, export the model for use with ONNX Runtime, TensorRT, etc.:

```bash
python scripts/export_onnx.py
```

Writes `outputs/model_spleen.onnx` (Opset 12, input/output names `input` / `output`). Requires `outputs/best_model.pth`.

### 7. Validate ONNX vs PyTorch (optional)

Check that the ONNX model matches PyTorch predictions (max difference &lt; 0.01):

```bash
python scripts/validate_onnx.py
```

Requires `outputs/best_model.pth` and `outputs/model_spleen.onnx`.

### 8. Export to TensorRT (optional)

Build a TensorRT FP16 engine for maximum inference speed. Use NVIDIA Docker (trtexec is included):

```bash
docker run --gpus all -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/tensorrt:24.01-py3 trtexec \
  --onnx=outputs/model_spleen.onnx \
  --saveEngine=outputs/model_spleen_fp16.engine \
  --fp16 \
  --minShapes=input:1x1x96x96x96 \
  --optShapes=input:1x1x96x96x96 \
  --maxShapes=input:1x1x96x96x96
```

First build can take ~10 minutes (kernel profiling). Add `--timingCacheFile=outputs/tensorrt.cache` to speed up future builds.

### 9. Validate Dice across backends (optional)

Compute Dice score on the validation set for PyTorch, ONNX Runtime, and TensorRT (.engine):

```bash
python scripts/validate_dice.py
# With custom engine path:
python scripts/validate_dice.py --engine outputs/model_spleen_fp16.engine
```

Requires `outputs/best_model.pth`, `outputs/model_spleen.onnx`. For TensorRT: `outputs/model_spleen_fp16.engine` and `pip install cupy-cuda12x`.

### 10. Benchmark (optional)

Python benchmark (PyTorch, ONNX Runtime). TensorRT pure performance: use `trtexec` (see step 8).

```bash
python scripts/benchmark.py
```

### 12. C++ inference (optional)

A minimal C++ inference example using the ONNX Runtime C++ API is in `cpp/inference.cpp`. It expects `outputs/model_spleen.onnx` and an ONNX Runtime SDK (e.g. [onnxruntime-linux-x64-gpu-1.16.3](https://github.com/microsoft/onnxruntime/releases) for GPU).

**Build** (from project root, adjust paths to your ONNX Runtime install):

```bash
g++ cpp/inference.cpp -o inference_cpp \
    -I onnxruntime-linux-x64-gpu-1.16.3/include \
    -L onnxruntime-linux-x64-gpu-1.16.3/lib \
    -lonnxruntime -std=c++17
```

**Run**:

```bash
LD_LIBRARY_PATH=$(pwd)/onnxruntime-linux-x64-gpu-1.16.3/lib ./inference_cpp
```

For CPU-only, use the CPU ONNX Runtime package and remove the CUDA provider from `inference.cpp` (or use the CPU build and do not call `AppendExecutionProvider_CUDA`).

### 13. Run the FastAPI API (optional)

A REST API in `api/app.py` exposes **POST /predict**: upload a NIfTI (`.nii.gz`) CT volume; the app preprocesses it (LoadImage, HU window [-57, 164], resize to 96×96×96), runs ONNX inference, and returns JSON with `filename`, `detected_spleen_voxels`, `has_spleen`, and `inference_engine`.

**Prerequisite:** `outputs/model_spleen.onnx` must exist (run `python scripts/export_onnx.py` first).

**Local run** (from project root, with `pip install -e .`):

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/docs for Swagger UI and try **POST /predict** with a NIfTI file.

**Docker run:**

Build (ensure `outputs/model_spleen.onnx` exists; run `python scripts/export_onnx.py` first; the Dockerfile copies it into the image):

```bash
docker build -t spleen-seg-api .
docker run -p 8000:8000 spleen-seg-api
```

API is available at http://localhost:8000. Use **POST /predict** with a NIfTI file as form body.

---

## Deployment & Optimization

To bridge the gap between research and production, this project includes:

| Component | Description |
|-----------|-------------|
| **ONNX Export** | PyTorch → ONNX (Opset 12) for interoperability (`scripts/export_onnx.py`). |
| **TensorRT Export** | ONNX → FP16 engine via `trtexec` for maximum GPU performance (~2.8× faster). |
| **Numerical Validation** | `scripts/validate_onnx.py` and `scripts/validate_dice.py` ensure Dice consistency across backends. |
| **FastAPI + Docker** | REST API (`api/app.py`) for NIfTI upload and ONNX inference. |
| **C++ Inference** | Low-latency prototype using **ONNX Runtime C++ API** (`cpp/inference.cpp`). |

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
- [MONAI Spleen Segmentation Tutorial](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb) for the tutorial inspiration
- [Weights & Biases](https://wandb.ai/) for experiment tracking
- [ONNX Runtime](https://onnxruntime.ai/) for the inference pipeline and ONNX validation
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) for GPU optimization
