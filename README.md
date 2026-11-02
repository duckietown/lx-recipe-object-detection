# LX Recipe: Object Detection

Recipe for the [lx-object-detection](https://github.com/duckietown/lx-object-detection) learning experience and its
solution repository. The LX ("meat") repositories provide the notebooks and the `solution` package that students work
on; this recipe provides the Docker images, the agent that runs on the robot and the Colab training notebook.

## Layout

- `Dockerfile`: the workbench image, i.e., what `dts code build` / `dts code workbench` run on the robot (or on the
  laptop with `--local`). It is based on `dt-commons` and only needs `onnxruntime` + `lib-dt-computer-vision` on top of
  it. The `packages/` and `assets/` directories of the recipe and of the LX are merged into the image.
- `Dockerfile.vscode`: the editor image (`dts code editor`). On top of the workbench dependencies it installs what the
  notebooks need to label (SAM3), train (ultralytics / torch) and export (ONNX) a model, see `dependencies-vscode.txt`.
  It installs on both amd64 and arm64 editors; on amd64 `dts code editor --gpus all` gives the notebooks a local GPU.
- `packages/agent.py`: the DTPS node. It subscribes to the camera (jpeg, intrinsics, homography), runs the student's
  `MLModel` (`solution/model.py` in the LX) on every rectified frame and publishes wheel commands, the LED pattern
  (brake lights on when stopped) and a debug image with the detections drawn (`object_detector_image/jpeg`).
  With `--mode data_collection` (launcher `data_collection`) it only saves frames to `/data/data_collection`.
- `packages/utils.py`: the inference backends. `ONNXBackend` wraps ONNX Runtime; `BACKENDS` maps the `BACKEND` value
  set in `solution/config.py` to a backend class so that other backends (e.g., TensorRT on the Jetson) can be added.
- `assets/colab/dt_object_detection_training.ipynb`: the Colab notebook (SAM3 auto-labelling, YOLO11n training, ONNX
  export). The LX's `notebooks/03-Training/local-training.ipynb` does the same inside the editor image.
- `launchers/`: `default` (agent) and `data_collection`.

## Model format

The agent expects a YOLO model exported with `imgsz=(480, 640)`, `opset=15`, `nms=False`, `half=False`,
`simplify=True`, `dynamic=False`. The raw output is a `[4 + C, anchors]` tensor and the students implement the
non-maximum suppression themselves (`MLModel._apply_nms`). Keep the export in fp32: that is what the CPU on the robot
runs, and the CUDA provider accepts fp32 as well.

## Running on the robot (CPU)

`dts code build -R ROBOT` builds the image on the robot itself (slow, but it works) and `dts code workbench -R ROBOT`
runs the agent on the robot. Inference runs on the CPU through ONNX Runtime, using all the cores. Measured cost: about
60 ms per frame on a single fast arm64 core (fp32 YOLO11n at 480x640), so expect low single-digit frames per second
on a Jetson Nano. The laptop (`--local`) therefore stays the recommended default for speed.

GPU inference on the robot (CUDA / TensorRT providers for the Jetson) is tracked separately in DTSW-8526; the
`BACKENDS` registry in `packages/utils.py` is where such a backend plugs in.

## Building the images by hand

From the recipe directory, with the LX (or the solution) checked out next to it:

```
docker buildx build --load -f Dockerfile --build-context recipe=. --build-context meat=../lx-object-detection \
    --build-arg ARCH=arm64v8 --build-arg DISTRO=ente --build-arg DOCKER_REGISTRY=docker.io \
    --build-arg BASE_TAG=ente-arm64v8 -t lx-object-detection:test .
docker buildx build --load -f Dockerfile.vscode --build-context recipe=. --build-context meat=../lx-object-detection \
    --build-arg ARCH=arm64v8 --build-arg DISTRO=ente --build-arg DOCKER_REGISTRY=docker.io \
    --build-arg BASE_TAG=ente-arm64v8 -t lx-object-detection:vscode .
```
