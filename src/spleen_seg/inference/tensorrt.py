"""TensorRT native .engine inference."""

import os
import numpy as np

_CUDA_BACKEND = None


def _detect_cuda_backend():
    global _CUDA_BACKEND
    if _CUDA_BACKEND is not None:
        return _CUDA_BACKEND
    try:
        import cupy as cp
        _ = cp.array([1.0])
        _CUDA_BACKEND = "cupy"
        return _CUDA_BACKEND
    except (ImportError, Exception):
        pass
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        _CUDA_BACKEND = "pycuda"
        return _CUDA_BACKEND
    except (ImportError, Exception):
        pass
    _CUDA_BACKEND = False
    return _CUDA_BACKEND


def create_tensorrt_predictor(engine_path, device_id=0):
    """Create a callable predictor for TensorRT .engine inference."""
    import tensorrt as trt
    import torch

    backend = _detect_cuda_backend()
    if not backend:
        raise RuntimeError(
            "TensorRT .engine requires CuPy or pycuda. Install: pip install cupy-cuda12x"
        )

    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine file not found: {engine_path}")

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    input_shape = (1, 1, 96, 96, 96)
    output_shape = (1, 2, 96, 96, 96)
    input_size = int(np.prod(input_shape))
    output_size = int(np.prod(output_shape))

    if backend == "cupy":
        import cupy as cp
        stream = cp.cuda.Stream(non_blocking=False)

        def _run_one(ctx, inp_arr, cp_mod):
            inp_gpu = cp_mod.ascontiguousarray(cp_mod.asarray(inp_arr))
            out_gpu = cp_mod.empty(output_shape, dtype=cp_mod.float32)
            bindings = [int(inp_gpu.data.ptr), int(out_gpu.data.ptr)]
            ctx.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)
            stream.synchronize()
            return cp_mod.asnumpy(out_gpu)

        def predictor_torch(x):
            inp = x.cpu().numpy().astype(np.float32)
            if inp.shape[0] > 1:
                outputs = [_run_one(context, inp[i : i + 1], cp) for i in range(inp.shape[0])]
                out = np.concatenate(outputs, axis=0)
            else:
                out = _run_one(context, inp, cp)
            return torch.from_numpy(out).to(x.device)

        return predictor_torch

    else:
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        d_input = cuda.mem_alloc(input_size * 4)
        d_output = cuda.mem_alloc(output_size * 4)
        strm = cuda.Stream()

        def _run_one(ctx, inp_flat, cuda_mod):
            cuda_mod.memcpy_htod_async(d_input, inp_flat, strm)
            ctx.execute_async_v2(
                bindings=[int(d_input), int(d_output)], stream_handle=strm.handle
            )
            h_out = np.empty(output_size, dtype=np.float32)
            cuda_mod.memcpy_dtoh_async(h_out, d_output, strm)
            strm.synchronize()
            return h_out.reshape(output_shape)

        def predictor_torch(x):
            inp = x.cpu().numpy().astype(np.float32)
            if inp.shape[0] > 1:
                outputs = [
                    _run_one(context, inp[i : i + 1].ravel(), cuda)
                    for i in range(inp.shape[0])
                ]
                out = np.concatenate(outputs, axis=0)
            else:
                out = _run_one(context, inp.ravel(), cuda)
            return torch.from_numpy(out).to(x.device)

        return predictor_torch
