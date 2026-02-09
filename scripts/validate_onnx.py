#!/usr/bin/env python3
"""Compare PyTorch and ONNX outputs for consistency."""

import os
import torch
import numpy as np
import onnxruntime as ort

from spleen_seg.config import MODEL_PATH, ONNX_PATH, SPATIAL_SIZE
from spleen_seg.model import get_model


def to_numpy(t):
    return t.detach().cpu().numpy() if t.requires_grad else t.cpu().numpy()


def validate():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ONNX_PATH):
        print("Error: Model or ONNX file not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 1, *SPATIAL_SIZE).to(device)
    with torch.no_grad():
        torch_out = model(dummy_input)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in providers if p in ort.get_available_providers()]
    if not providers:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(ONNX_PATH, providers=providers)
    ort_inputs = {session.get_inputs()[0].name: to_numpy(dummy_input)}
    onnx_out = session.run(None, ort_inputs)[0]

    diff = np.abs(to_numpy(torch_out) - onnx_out)
    is_ok = np.allclose(to_numpy(torch_out), onnx_out, rtol=1e-3, atol=1e-2)
    print(f"Max difference: {np.max(diff):.6f}")
    print("Outputs are consistent!" if is_ok else "WARNING: Outputs differ.")


if __name__ == "__main__":
    validate()
