"""FastAPI application for spleen segmentation."""

import os
import shutil
import tempfile
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensityRange,
    Resize,
    EnsureType,
)

from spleen_seg.config import ONNX_PATH, HU_MIN, HU_MAX, SPATIAL_SIZE

app = FastAPI(
    title="Spleen Segmentation API",
    description="API for segmenting spleen from CT scans using a 3D UNet model",
    version="1.0.0",
)

ort_session = None

preprocessing = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensityRange(a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
    Resize(SPATIAL_SIZE),
    EnsureType(),
])


@app.on_event("startup")
def load_model():
    global ort_session
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(f"Model file not found at {ONNX_PATH}")
    print(f"Loading model from {ONNX_PATH}...")
    providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(ONNX_PATH, providers=providers)
    print("Model loaded successfully")


@app.post("/predict")
async def predict_spleen(file: UploadFile = File(...)):
    if not ort_session:
        raise HTTPException(status_code=500, detail="Model not loaded")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        input_data = preprocessing(tmp_path)
        input_tensor = input_data.unsqueeze(0).numpy()
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: input_tensor})
        raw_output = outputs[0]
        mask = np.argmax(raw_output, axis=1).astype(np.uint8)
        spleen_voxels = int(np.sum(mask))
        return JSONResponse(content={
            "filename": file.filename,
            "detected_spleen_voxels": spleen_voxels,
            "has_spleen": spleen_voxels > 100,
            "inference_engine": "ONNX Runtime",
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(tmp_path)
