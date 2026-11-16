#!/usr/bin/env python3
import cv2
import signal
import asyncio
import time
from asyncio import AbstractEventLoop
import argparse
import numpy as np
from typing import Optional

from dt_computer_vision.camera import CameraModel
from dt_computer_vision.camera.types import Pixel
from dtps import context, ContextConfig, DTPSContext
from dt_node_utils.node import Node
from dt_node_utils import NodeType
from dtps_http import RawData
from dt_robot_utils import get_robot_name
from duckietown_messages.actuators.differential_pwm import DifferentialPWM
from duckietown_messages.calibrations.camera_intrinsic import CameraIntrinsicCalibration
from duckietown_messages.calibrations.camera_extrinsic import CameraExtrinsicCalibration
from duckietown_messages.utils.exceptions import DataDecodingError
from duckietown_messages.sensors.camera import Camera
from duckietown_messages.actuators.car_lights import CarLights
from duckietown_messages.colors.rgba import RGBA
from duckietown_messages.sensors.compressed_image import CompressedImage
from duckietown_messages.standard.dictionary import Dictionary
from turbojpeg import TurboJPEG
from dt_computer_vision.ground_projection import GroundProjector

from solution.model import MLModel
from solution.config import DATA_COLLECTION_ROOT, SAVE_EVERY_N_FRAMES, MAX_LOG_IMAGES, BACKEND, MODEL_PATH
from utils import BACKENDS


# Detections are published on `switchboard/<robot>/object_detector/detections` so that the
# ROS 2 detection relayer (dt-ros2-interface) can turn them into `duckietown_msgs` obstacle
# lists. Keep this payload in step with the relayer's documented contract.
DETECTIONS_TOPIC = "object_detector"

# The model reports a class id; the relayer only distinguishes cones from duckies by name.
DEFAULT_CLASS_NAMES = {0: "duckie", 1: "cone"}


def detections_payload(detections, width, height, stamp, ground_projector, class_names,
                       inference_ms=None):
    """Build the DTPS detections payload the ROS 2 relayer consumes.

    Each detection is `(x1, y1, x2, y2, score, class_id)` in the rectified image. The ground
    point is the bottom-centre of the box projected onto the ground plane, which is where the
    object touches the floor; it is omitted when there is no projection available (no
    extrinsics yet) or when the ray does not meet the ground.
    """
    items = []
    for det in detections:
        x1, y1, x2, y2, score = (float(v) for v in det[:5])
        class_id = int(det[5]) if len(det) > 5 else 0
        ground, distance = None, None
        if ground_projector is not None:
            try:
                pixel = Pixel(x=0.5 * (x1 + x2), y=y2)
                point = ground_projector.vector2ground(
                    ground_projector.camera.pixel2vector(pixel)
                )
                ground = {"x": float(point.x), "y": float(point.y)}
                distance = float(np.linalg.norm([point.x, point.y]))
            except Exception:
                # a detection without a ground point is still worth reporting
                ground, distance = None, None
        items.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "score": score,
            "class_id": class_id,
            "class_name": class_names.get(class_id, str(class_id)),
            "ground": ground,
            "distance": distance,
        })
    return {
        "frame": {"width": int(width), "height": int(height), "stamp": stamp},
        "inference_ms": inference_ms,
        "provider": "lx-object-detection",
        "detections": items,
    }


def draw_detections(img, detections):
    # Make a copy so the original isn't modified
    out = img.copy()

    for det in detections:
        x1, y1, x2, y2, score = det[:5]

        # Draw bounding box
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      color=(0, 255, 0), thickness=2)

        # Label with score (rounded to 2 decimals)
        label = f"{score:.2f}"

        # Choose a location slightly above the top-left corner
        cv2.putText(out, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

    return out


def make_backend():
    """Instantiate the inference backend selected in `solution/config.py`."""
    try:
        backend_cls = BACKENDS[BACKEND]
    except KeyError:
        raise ValueError(f"Unknown inference backend '{BACKEND}', available backends: {sorted(BACKENDS)}")
    return backend_cls(MODEL_PATH)


class MLAgent(Node):
    def __init__(self, mode: str = "agent"):
        node_name: str = f"object_detection_agent"
        super().__init__(
            name=node_name,
            kind=NodeType.GENERIC,
            description="A node detecting duckies"
        )
        self._shutdown = False
        self._robot_name = get_robot_name()
        self.pwm_publisher: Optional[DTPSContext] = None
        self.led_publisher: Optional[DTPSContext] = None
        self.obj_img_publisher: Optional[DTPSContext] = None
        self.detections_publisher: Optional[DTPSContext] = None
        self.class_names: Dict[int, str] = dict(DEFAULT_CLASS_NAMES)
        # event loop
        self._loop: Optional[AbstractEventLoop] = None

        self.camera_intrinsics: Optional[CameraIntrinsicCalibration] = None
        self.camera_extrinsics: Optional[CameraExtrinsicCalibration] = None
        self.H: Optional[np.ndarray] = None
        self.ground_projector: Optional[GroundProjector] = None
        self.camera_info: Optional[Camera] = None
        self.camera: Optional[CameraModel] = None

        self.mode = mode
        self.data_collection = self.mode == "data_collection"

        # the model is only needed in `agent` mode (there is no trained model yet when collecting data)
        self.model: Optional[MLModel] = None if self.data_collection else MLModel(make_backend())
        # whether the robot is currently stopped (None: unknown, nothing published yet), used to publish the LEDs
        # pattern only when it changes
        self.stopped: Optional[bool] = None

        # register sigint handler
        signal.signal(signal.SIGINT, self._sigint_handler)
        self._jpeg = TurboJPEG()

        self.save_every_n_frames = SAVE_EVERY_N_FRAMES
        self.max_log_images = MAX_LOG_IMAGES
        self._frame_idx = 0
        self._logged_images = 0

        self.output_dir = DATA_COLLECTION_ROOT
        self.output_dir.mkdir(parents=True, exist_ok=True)


    async def save_camera_intrinsics(self, rdata: RawData):
        try:
            camera: CameraIntrinsicCalibration = CameraIntrinsicCalibration.from_rawdata(rdata)
        except DataDecodingError as e:
            print(f"Failed to decode an incoming message: {e.message}")
            print("Camera parameters not available yet.")
            return

        if self.camera_intrinsics is None:
            print("Received camera parameters.")

        self.camera_intrinsics = camera

    
    async def save_camera_extrinsics(self, rdata: RawData):
        try:
            extrinsics: CameraExtrinsicCalibration = CameraExtrinsicCalibration.from_rawdata(rdata)
        except DataDecodingError as e:
            print(f"Failed to decode extrinsics: {e.message}")
            print("Camera extrinsics not available yet.")
            return

        if self.camera_extrinsics is None:
            print("Received camera extrinsics.")

        self.camera_extrinsics = extrinsics
        self.H = np.array(extrinsics.homography, dtype=float).reshape(3,3)


    async def save_camera_info(self, rdata: RawData):
            """
            Get the camera specification and save it to a variable.
            """
            try:
                camera: Camera = Camera.from_rawdata(rdata)
            except DataDecodingError as e:
                print(f"Failed to decode an incoming message: {e.message}")
                print("Camera information not available yet.")
                return

            if self.camera_info is None:
                print("Received camera information.")

            self.camera_info = camera

    async def img_cb(self, data: RawData):

        if self._loop is None:
            return

        if self.camera is None:
            # the camera model (and the ground projector) need the camera info, the intrinsics and the homography
            if self.camera_info is not None and self.camera_intrinsics is not None and self.H is not None:
                print("Camera info, intrinsics and extrinsics received, initializing camera model.")
                
                self.camera = CameraModel(
                    width=self.camera_info.width,
                    height=self.camera_info.height,
                    K=np.reshape(self.camera_intrinsics.K, (3,3)),
                    D=np.reshape(self.camera_intrinsics.D, (5,)),
                    R=np.reshape(self.camera_intrinsics.R, (3,3)),
                    P=np.reshape(self.camera_intrinsics.P, (3,4)),
                    H=self.H
                )

                self.ground_projector = GroundProjector(self.camera)
                if self.model is not None:
                    self.model.set_ground_projector(self.ground_projector)

            else:
                print("Still waiting for camera info, intrinsics or extrinsics.")
                return

        try:
            jpeg_data: CompressedImage = CompressedImage.from_rawdata(data).data
        except DataDecodingError as e:
            print(f"Failed to decode an incoming message: {e.message}")
            return

        image_array = (np.frombuffer(jpeg_data,np.uint8))
        decoded_image = self._jpeg.decode(image_array)
        rectified_img = self.camera.rectifier.rectify(decoded_image)

        if self.data_collection:
            if self._frame_idx == 0:
                print(f"Data collection started - capturing the robot's view as it moves!")
            self._frame_idx += 1
            if self._logged_images >= self.max_log_images:
                print(f"Logging limit reached. Increase the limit if you need to collect more data or press CTRL-C to exit.")
                return
            
            if self._frame_idx % self.save_every_n_frames == 0:
                filename = self.output_dir / f"{self._robot_name}_{self._frame_idx}.png"
                try:
                    cv2.imwrite(str(filename), rectified_img)
                    if self._logged_images % 10 == 0:
                        print(f"Saved {self._logged_images} images (press CTRL-C to stop)")
                    self._logged_images += 1
                except Exception as e:
                    print(f"Failed to save image {filename}: {e}")
            return

        inference_started = time.time()
        [pwm, detections] = self.model.get_wheel_velocities_from_image(rectified_img)
        inference_ms = 1000.0 * (time.time() - inference_started)
        if detections is None:
            # inference failed, nothing to draw
            detections = []

        # wheels first: this is the time-critical command
        try:
            await self.pwm_publisher.publish(pwm.to_rawdata())
        except Exception:
            print("Error publishing wheels data.")

        # LEDs: brake lights on when we stop, off when we move; only published when the state changes
        white = RGBA(
            r = 1.0,
            g = 1.0,
            b = 1.0,
            a = 1.0,
        )
        red = RGBA(
            r = 1.0,
            b = 0.0,
            g = 0.0,
            a = 1.0,
        )
        black = RGBA(
            r = 0.0,
            b = 0.0,
            g = 0.0,
            a = 1.0,
        )
        stopped = pwm.left == 0.0 and pwm.right == 0.0
        if stopped != self.stopped:
            if stopped:
                # we are in STOP mode publish the back lights red
                led_raw: RawData = CarLights(
                    front_left = white,
                    front_right = white,
                    back_left = red,
                    back_right = red,
                ).to_rawdata()
            else:
                # we are moving turn the brake lights off
                led_raw: RawData = CarLights(
                    front_left = white,
                    front_right = white,
                    back_left = black,
                    back_right = black,
                ).to_rawdata()
            self.stopped = stopped
            asyncio.run_coroutine_threadsafe(self.led_publisher.publish(led_raw), self._loop)

        # debug image with the detections drawn on top
        annotated_img = draw_detections(rectified_img, detections)
        raw_img_data = CompressedImage(format="jpeg", data=self._jpeg.encode(annotated_img)).to_rawdata()
        asyncio.run_coroutine_threadsafe(self.obj_img_publisher.publish(raw_img_data),
                                         self._loop)

        # detections for the ROS 2 relayer; never let this disturb the agent
        if self.detections_publisher is not None:
            try:
                height, width = rectified_img.shape[:2]
                payload = detections_payload(
                    detections, width, height, inference_started,
                    self.ground_projector, self.class_names, inference_ms,
                )
                asyncio.run_coroutine_threadsafe(
                    self.detections_publisher.publish(
                        Dictionary(data=payload).to_rawdata()
                    ),
                    self._loop,
                )
            except Exception as e:
                print(f"Could not publish detections: {e}")

    async def worker(self):
        await self.dtps_init()
        switchboard = (await context("switchboard")).navigate(self._robot_name)

        jpeg = await (switchboard / "sensor" / "camera" / "front_center" / "jpeg").until_ready()
        params = await (switchboard / "sensor" / "camera" / "front_center" / "parameters").until_ready()
        info = await (switchboard / "sensor" / "camera" / "front_center" / "info").until_ready()
        extr   = await (switchboard / "sensor" / "camera" / "front_center" / "homography").until_ready()

        self.pwm_publisher = await (switchboard / "actuator" / "wheels" / "base" / "pwm").until_ready()
        self.led_publisher = await (switchboard / "actuator" / "lights" / "base" / "pattern").until_ready()

        out: DTPSContext = self.context / "out"
        obj_img_queue = await ( out / "jpeg").queue_create()
        detections_queue = await (out / "detections").queue_create()

        self.obj_img_publisher = await obj_img_queue.publisher()
        self.detections_publisher = await detections_queue.publisher()
        await self.dtps_expose()
        await (switchboard / "object_detector_image" / "jpeg").expose(obj_img_queue)
        await (switchboard / DETECTIONS_TOPIC / "detections").expose(detections_queue)
        jpeg = jpeg.configure(ContextConfig(patient=True))
        params = params.configure(ContextConfig(patient=True))
        info = info.configure(ContextConfig(patient=True))
        extr   = extr.configure(ContextConfig(patient=True))

        await params.subscribe(self.save_camera_intrinsics)
        await info.subscribe(self.save_camera_info)
        await extr.subscribe(self.save_camera_extrinsics)
        await jpeg.subscribe(self.img_cb)

        self._loop = asyncio.get_event_loop()
        await self.join()

    async def join(self):
        while not self._shutdown:
            await asyncio.sleep(1)

    def _sigint_handler(self, _, __):
        self._shutdown = True

    @property
    def is_shutdown(self):
        return self._shutdown


    def spin(self):
        try:
            asyncio.run(self.worker())
        except RuntimeError as e:
            if not self.is_shutdown:
                print(f"An error occurred while running the event loop: {e}")
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["agent","data_collection"], default="agent", help="Run mode: 'agent' to control wheels, 'data_collection' to log images")
    args = parser.parse_args()

    node = MLAgent(mode=args.mode)
    node.spin()
