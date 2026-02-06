import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import os

MODEL_PATH = "outputs/best_model.pth"
ONNX_PATH = "outputs/model_spleen.onnx"
SPATIAL_SIZE = (96, 96, 96)

def export_to_onnx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval() # set model to evaluation mode (no dropout, batch norm)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Model not found at {MODEL_PATH}")
        return

    # Create a dummy input tensor for Tracing
    dummy_input = torch.randn(1, 1, *SPATIAL_SIZE).to(device)

    # Export the model to ONNX
    print(f"Exporting model to {ONNX_PATH}...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Model exported to {ONNX_PATH}")
    print("this file can now used with C++, TensorRT, etc.")

if __name__ == "__main__":
    export_to_onnx()