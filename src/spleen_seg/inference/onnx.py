"""ONNX Runtime inference backend."""

import numpy as np
import onnxruntime as ort
import torch

from ..config import ROI_SIZE, SW_BATCH_SIZE


def create_onnx_predictor(onnx_path, use_tensorrt=False):
    """Create an ONNX Runtime predictor compatible with sliding_window_inference."""
    if use_tensorrt:
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    def predictor(x):
        # x: torch.Tensor (B, 1, D, H, W)
        inp = x.cpu().numpy().astype(np.float32)
        out = session.run(None, {input_name: inp})[0]
        return torch.from_numpy(out).to(x.device)

    return predictor
