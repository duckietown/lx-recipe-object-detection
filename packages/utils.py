#!/usr/bin/env python3

import numpy as np


class ONNXBackend:
    """ONNX Runtime inference backend."""

    def __init__(self, onnx_path):
        import onnxruntime as ort
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        inp = self._sess.get_inputs()[0]
        self._input_name = inp.name
        self.net_h   = inp.shape[2]
        self.net_w   = inp.shape[3]
        self.in_dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
        print(f"ONNX backend ready  |  providers: {self._sess.get_providers()}")

    def infer(self, x: np.ndarray) -> np.ndarray:
        """Run inference. Returns raw tensor [4+C, anchors]."""
        return self._sess.run(None, {self._input_name: x})[0][0]


class TRTBackend:
    """TensorRT inference backend. Builds the engine from ONNX on first run."""

    def __init__(self, onnx_path, trt_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        self._cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)

        if not trt_path.exists():
            if not onnx_path.exists():
                raise FileNotFoundError(
                    f"Neither TRT engine ({trt_path}) nor ONNX model ({onnx_path}) found."
                )
            _build_engine_from_onnx(onnx_path, trt_path, logger)

        runtime = trt.Runtime(logger)
        with open(str(trt_path), "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()

        input_shape  = self._engine.get_binding_shape(0)
        self.net_h   = input_shape[2]
        self.net_w   = input_shape[3]
        self.in_dtype = trt.nptype(self._engine.get_binding_dtype(0))

        self._inputs, self._outputs, self._bindings = [], [], []
        self._stream = cuda.Stream()
        for i in range(self._engine.num_bindings):
            shape    = self._engine.get_binding_shape(i)
            size     = trt.volume(shape)
            dtype    = trt.nptype(self._engine.get_binding_dtype(i))
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem  = cuda.mem_alloc(host_mem.nbytes)
            self._bindings.append(int(dev_mem))
            bucket   = self._inputs if self._engine.binding_is_input(i) else self._outputs
            bucket.append({"host": host_mem, "device": dev_mem})

        print("TRT backend ready")

    def infer(self, x: np.ndarray) -> np.ndarray:
        """Run inference. Returns raw tensor [4+C, anchors]."""
        cuda = self._cuda
        np.copyto(self._inputs[0]["host"], x.ravel())
        cuda.memcpy_htod_async(self._inputs[0]["device"], self._inputs[0]["host"], self._stream)
        self._context.execute_async_v2(bindings=self._bindings, stream_handle=self._stream.handle)
        cuda.memcpy_dtoh_async(self._outputs[0]["host"], self._outputs[0]["device"], self._stream)
        self._stream.synchronize()
        out_shape = self._engine.get_binding_shape(self._engine.num_bindings - 1)
        return self._outputs[0]["host"].reshape(out_shape)[0]


def _build_engine_from_onnx(onnx_path, trt_path, logger):
    import tensorrt as trt
    print(f"Building TensorRT engine from {onnx_path} — this may take a few minutes...")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)
    with open(str(onnx_path), "rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errors}")
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1 GiB
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    with open(str(trt_path), "wb") as f:
        f.write(serialized)
    print(f"Engine saved to {trt_path}")


def jetson_stats() -> dict:
    """Read Jetson GPU/RAM stats for logging. Returns n/a for unavailable fields."""
    import pycuda.driver as cuda
    stats = {}
    try:
        with open("/sys/devices/gpu.0/load") as f:
            stats["GR3D"] = f"{int(f.read().strip()) / 10:.0f}%"
    except OSError:
        stats["GR3D"] = "n/a"
    try:
        free, total = cuda.mem_get_info()
        used = (total - free) // 1024 // 1024
        stats["GPU_MEM"] = f"{used}/{total // 1024 // 1024}MB"
    except Exception:
        stats["GPU_MEM"] = "n/a"
    try:
        with open("/proc/meminfo") as f:
            meminfo = {k: int(v.split()[0]) for k, v in
                       (line.split(":", 1) for line in f if ":" in line)}
        total = meminfo["MemTotal"] // 1024
        used  = (meminfo["MemTotal"] - meminfo["MemAvailable"]) // 1024
        stats["RAM"] = f"{used}/{total}MB"
    except OSError:
        stats["RAM"] = "n/a"
    return stats
