#!/bin/bash
# Build TensorRT FP16 engine from ONNX model.
# Requires: trtexec (from TensorRT), outputs/model_spleen.onnx

ONNX_PATH="${1:-outputs/model_spleen.onnx}"
ENGINE_PATH="${2:-outputs/model_spleen_fp16.engine}"

if [ ! -f "$ONNX_PATH" ]; then
    echo "Error: ONNX file not found: $ONNX_PATH"
    exit 1
fi

trtexec --onnx="$ONNX_PATH" --saveEngine="$ENGINE_PATH" --fp16
echo "Saved engine to: $ENGINE_PATH"
