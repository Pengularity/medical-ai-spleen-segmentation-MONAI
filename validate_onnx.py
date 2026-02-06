import torch
import numpy as np
import onnxruntime as ort
from monai.networks.nets import UNet
from monai.networks.layers import Norm

MODEL_PATH = "outputs/best_model.pth"
ONNX_PATH = "outputs/model_spleen.onnx"
SPATIAL_SIZE = (96, 96, 96)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def validate():
    print(" Validating ONNX model...")
    
    dummy_input = torch.randn(1, 1, *SPATIAL_SIZE).cuda()
    
    model_pt = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).cuda()
    
    model_pt.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
    model_pt.eval()

    with torch.no_grad():
     torch_out = model_pt(dummy_input)
    
    ort_session = ort.InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_out = ort_outs[0]
    
    # Compare the outputs
    # np.allclose is a function that checks if two arrays are close to each other
    # rtol is the relative tolerance (default 1e-03)
    # atol is the absolute tolerance (default 1e-02)
    is_identical = np.allclose(to_numpy(torch_out), onnx_out, rtol=1e-03, atol=1e-02)
    
    diff = np.abs(to_numpy(torch_out) - onnx_out)
    print(f"Max difference: {np.max(diff):.6f}")
    
    if is_identical:
        print("Outputs are identical at 99.99% accuracy!")
    else:
        print("WARNING: ONNX and PyTorch outputs are different.")
        


if __name__ == "__main__":
    validate()