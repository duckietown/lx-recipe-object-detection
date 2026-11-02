#!/usr/bin/env python3
"""
Inference backends for the object-detection agent.

A backend wraps a trained detector and exposes:

- ``net_h``, ``net_w``: the input resolution the network was exported with;
- ``in_dtype``: the numpy dtype the input tensor must have (float32, or float16 for a model exported with ``half=True``);
- ``infer(x)``: run the network on a preprocessed ``[1, 3, net_h, net_w]`` tensor and return the raw output for the
  (single) image, i.e. a ``[4 + C, anchors]`` tensor for a YOLO model exported with ``nms=False``.

Only an ONNX Runtime backend is provided: it runs on the CPU on the robot and on the GPU on an x86_64 laptop where
``onnxruntime-gpu`` is installed. A GPU backend for the Jetson can be added to ``BACKENDS`` (see DTSW-8526).
"""
from pathlib import Path
from typing import Dict, Type, Union

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

        # CUDA is used when available (onnxruntime-gpu on an x86_64 laptop), the CPU otherwise
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        providers = [p for p in wanted if p in available] or ["CPUExecutionProvider"]

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


# Registry of the available inference backends, keyed by the value of `BACKEND` in `solution/config.py`.
# A GPU backend (e.g., TensorRT on the Jetson) can be plugged in here.
BACKENDS: Dict[str, Type] = {
    "onnx": ONNXBackend,
}
