#!/usr/bin/env python3
"""Benchmark PyTorch and ONNX Runtime inference latency."""

import os
import sys
import time
import argparse
import numpy as np
import torch
import onnxruntime as ort

from spleen_seg.config import MODEL_PATH, ONNX_PATH, SPATIAL_SIZE
from spleen_seg.model import get_model

WARMUP = 100
ITERATIONS = 500


def benchmark_pytorch(device):
    model = get_model(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    x = torch.randn(1, 1, *SPATIAL_SIZE, dtype=torch.float32).to(device)

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(ITERATIONS):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def benchmark_onnx(use_tensorrt=False):
    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"] if use_tensorrt else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in providers if p in ort.get_available_providers()]
    if not providers:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(ONNX_PATH, providers=providers)
    input_name = session.get_inputs()[0].name
    x = np.random.randn(1, 1, *SPATIAL_SIZE).astype(np.float32)

    for _ in range(WARMUP):
        _ = session.run(None, {input_name: x})
    latencies = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: x})
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def format_stats(latencies):
    arr = np.array(latencies)
    return {
        "mean_ms": np.mean(arr),
        "median_ms": np.median(arr),
        "p99_ms": np.percentile(arr, 99),
        "throughput_qps": 1000.0 / np.mean(arr),
    }


def print_table(rows, headers):
    col_widths = [max(len(str(h)), 4) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    fmt = "|" + "|".join(f" {{:<{w}}} " for w in col_widths) + "|"
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch and ONNX Runtime")
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    args = parser.parse_args()
    global ITERATIONS
    ITERATIONS = args.iterations

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(ONNX_PATH):
        print(f"Error: ONNX not found at {ONNX_PATH}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Iterations: {ITERATIONS} (warmup: {WARMUP})\n")
    results = []

    print("Benchmarking PyTorch...")
    try:
        lat = benchmark_pytorch(device)
        s = format_stats(lat)
        results.append(("PyTorch", s))
        print(f"  mean={s['mean_ms']:.2f} ms, throughput={s['throughput_qps']:.0f} qps\n")
    except Exception as e:
        print(f"  Error: {e}\n")

    print("Benchmarking ONNX Runtime (CUDA)...")
    try:
        lat = benchmark_onnx(use_tensorrt=False)
        s = format_stats(lat)
        results.append(("ONNX Runtime", s))
        print(f"  mean={s['mean_ms']:.2f} ms, throughput={s['throughput_qps']:.0f} qps\n")
    except Exception as e:
        print(f"  Error: {e}\n")

    print("=" * 70)
    print("Benchmark Summary (single patch 1x1x96x96x96)")
    print("=" * 70)
    headers = ["Backend", "Mean (ms)", "Median (ms)", "P99 (ms)", "Throughput (qps)"]
    rows = [[n, f"{s['mean_ms']:.2f}", f"{s['median_ms']:.2f}", f"{s['p99_ms']:.2f}", f"{s['throughput_qps']:.0f}"] for n, s in results]
    print_table(rows, headers)
    print("\nTensorRT pure: use trtexec (see README).")


if __name__ == "__main__":
    main()
