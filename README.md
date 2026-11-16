# LX Recipe: Object Detection

Recipe for the [lx-object-detection](https://github.com/duckietown/lx-object-detection) learning experience and its
solution repository. The LX ("meat") repositories provide the notebooks and the `solution` package that students work
on; this recipe provides the Docker images, the agent that runs on the robot and the Colab training notebook.

## Layout

- `Dockerfile`: the workbench image, i.e., what `dts code build` / `dts code workbench` run on the robot (or on the
  laptop with `--local`). It is based on `dt-commons` (build argument `BASE_IMAGE`) and only needs ONNX Runtime +
  `lib-dt-computer-vision` on top of it; ONNX Runtime is installed only when the base image does not provide it. The
  `packages/` and `assets/` directories of the recipe and of the LX are merged into the image.
- `Dockerfile.vscode`: the editor image (`dts code editor`). On top of the workbench dependencies it installs what the
  notebooks need to label (SAM3), train (ultralytics / torch) and export (ONNX) a model, see `dependencies-vscode.txt`.
  It installs on both amd64 and arm64 editors; on amd64 `dts code editor --gpus all` gives the notebooks a local GPU.
- `packages/agent.py`: the DTPS node. It subscribes to the camera (jpeg, intrinsics, homography), runs the student's
  `MLModel` (`solution/model.py` in the LX) on every rectified frame and publishes wheel commands, the LED pattern
  (brake lights on when stopped) and a debug image with the detections drawn (`object_detector_image/jpeg`).
  With `--mode data_collection` (launcher `data_collection`) it only saves frames to `/data/data_collection`.
- `packages/utils.py`: the inference backends. `ONNXBackend` wraps ONNX Runtime and uses the first execution provider
  that loads among TensorRT, CUDA and CPU; `BACKENDS` maps the `BACKEND` value set in `solution/config.py` to a backend
  class so that other backends can be added.
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

## Running on the robot's GPU

`dt-machine-learning-base-environment` provides the JetPack 4.6 GPU build of ONNX Runtime on the robot (Python 3.8, with
the TensorRT and CUDA providers; CUDA and TensorRT are mounted from the host) and `onnxruntime-gpu` on amd64. Building
the workbench image on it (`ARG BASE_IMAGE=dt-machine-learning-base-environment` in the `Dockerfile`) gives the agent
the Jetson's GPU with no code change. The recipe builds on both bases: the Python-version and Ubuntu-release specific
dependencies are in `dependencies-py3.txt` (markers) and `dependencies-apt.noble.txt`.

It becomes the default once that base image is published with Python 3.8 on arm64 (DTSW-7100); the arm64 image
published today is an older Python 3.6 build. To try it on a robot before then, build the base on the robot from the
DTSW-7100 branch of `dt-machine-learning-base-environment`, then the LX with a local copy of this recipe whose
`BASE_IMAGE` is changed:

```
dts devel build -H ROBOT                         # in dt-machine-learning-base-environment (DTSW-7100 branch)
dts code build -R ROBOT --no-pull --recipe ../lx-recipe-object-detection    # in the LX, recipe with BASE_IMAGE changed
```

(`--no-pull` keeps the base image built on the robot instead of pulling the published one.)

The first time the TensorRT provider sees a model it builds an engine, which takes minutes on a Jetson Nano. The
engine is cached in `trt_cache/` next to the model, inside the container, so it is rebuilt when the container is
recreated; set `TRT_ENGINE_CACHE_DIR` to a mounted directory to keep it.

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
