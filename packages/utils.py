#!/usr/bin/env python3
"""
Inference backends for the object-detection agent.

A backend wraps a trained detector and exposes:

- ``net_h``, ``net_w``: the input resolution the network was exported with;
- ``in_dtype``: the numpy dtype the input tensor must have (float32, or float16 for a model exported with ``half=True``);
- ``infer(x)``: run the network on a preprocessed ``[1, 3, net_h, net_w]`` tensor and return the raw output for the
  (single) image, i.e. a ``[4 + C, anchors]`` tensor for a YOLO model exported with ``nms=False``.

Only an ONNX Runtime backend is provided. It asks for the TensorRT provider first, then CUDA, then the CPU, and uses
the first one that loads: TensorRT or CUDA on a Jetson whose image provides the JetPack GPU build of ONNX Runtime
(``dt-machine-learning-base-environment``), CUDA on an x86_64 laptop with ``onnxruntime-gpu``, the CPU everywhere else.
Other backends can be added to ``BACKENDS``.
"""
import ctypes
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Type, Union

import numpy as np


class ONNXBackend:
    """ONNX Runtime inference backend."""

    def __init__(self, onnx_path: Union[str, Path]):
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found (did you download your trained model?): {onnx_path}")

        sess_opts = ort.SessionOptions()
        # NOTE: we deliberately do not set `intra_op_num_threads`: by default ONNX Runtime uses all the physical
        #       CPU cores, which is what we want on a CPU-only robot (e.g., a Jetson Nano).

        # TensorRT -> CUDA -> CPU: ONNX Runtime uses the first provider that loads
        available = ort.get_available_providers()
        providers: List[Any] = []
        if "TensorrtExecutionProvider" in available and _tensorrt_loadable():
            # building a TensorRT engine takes minutes on a Jetson Nano: cache it next to the model
            providers.append(("TensorrtExecutionProvider", {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": _engine_cache_dir(onnx_path),
                "trt_fp16_enable": True,
                "trt_max_workspace_size": 1 << 30,
            }))
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self._sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
        inp = self._sess.get_inputs()[0]
        self._input_name = inp.name
        self.net_h = int(inp.shape[2])
        self.net_w = int(inp.shape[3])
        self.in_dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
        print(f"ONNX backend ready  |  model: {onnx_path.name}  |  providers: {self._sess.get_providers()}")

    def infer(self, x: np.ndarray) -> np.ndarray:
        """Run inference on a preprocessed [1, 3, H, W] tensor. Returns the raw output tensor of the (only) image."""
        return self._sess.run(None, {self._input_name: x})[0][0]


def _tensorrt_loadable() -> bool:
    """Whether the TensorRT runtime library can be loaded. The PyPI ``onnxruntime-gpu`` lists the TensorRT provider
    even where TensorRT is not installed (most laptops); asking for it there only prints a long error before the
    fallback. On the Jetson the library is mounted from the host."""
    for name in ("libnvinfer.so.8", "libnvinfer.so.10", "libnvinfer.so"):
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


def _engine_cache_dir(onnx_path: Path) -> str:
    """Where TensorRT caches its engines: ``TRT_ENGINE_CACHE_DIR`` when set (point it at a mounted directory to keep the
    engines across container restarts), else next to the model, else the temporary directory."""
    candidates = [Path(os.environ["TRT_ENGINE_CACHE_DIR"])] if os.environ.get("TRT_ENGINE_CACHE_DIR") else []
    candidates += [onnx_path.parent / "trt_cache", Path(tempfile.gettempdir()) / "trt_cache"]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            if os.access(candidate, os.W_OK):
                return str(candidate)
        except OSError:
            continue
    return tempfile.gettempdir()


# Registry of the available inference backends, keyed by the value of `BACKEND` in `solution/config.py`.
BACKENDS: Dict[str, Type] = {
    "onnx": ONNXBackend,
}
