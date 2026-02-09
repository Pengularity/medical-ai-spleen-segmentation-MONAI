#!/usr/bin/env python3
"""Export PyTorch model to ONNX."""

import os
import torch

from spleen_seg.config import MODEL_PATH, ONNX_PATH, SPATIAL_SIZE
from spleen_seg.model import get_model


def export_to_onnx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 1, *SPATIAL_SIZE).to(device)
    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)

    print(f"Exporting to {ONNX_PATH}...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported to {ONNX_PATH}")


if __name__ == "__main__":
    export_to_onnx()
