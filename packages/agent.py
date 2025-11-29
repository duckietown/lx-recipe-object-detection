#!/usr/bin/env python3

import cv2
import signal
import asyncio
import argparse
import numpy as np
from pathlib import Path
from typing import Union, Dict, Optional

from dt_computer_vision.camera import CameraModel
from dtps import context, ContextConfig, DTPSContext
from dtps_http import RawData
from dt_robot_utils import get_robot_name
from duckietown_messages.actuators.differential_pwm import DifferentialPWM
from duckietown_messages.calibrations.camera_intrinsic import CameraIntrinsicCalibration
from duckietown_messages.calibrations.camera_extrinsic import CameraExtrinsicCalibration
from duckietown_messages.utils.exceptions import DataDecodingError
from duckietown_messages.sensors.camera import Camera
from duckietown_messages.sensors.compressed_image import CompressedImage
from turbojpeg import TurboJPEG
from dt_computer_vision.ground_projection import GroundProjector

from model import MLModel
from solution.config import DATA_COLLECTION_ROOT, SAVE_EVERY_N_FRAMES, MAX_LOG_IMAGES




class MLAgent:
    def __init__(self, mode: str = "agent", is_sim=None):
        self._shutdown = False
        self._robot_name = get_robot_name()
        self.pwm_publisher: Optional[DTPSContext] = None
        self.camera_intrinsics: Optional[CameraIntrinsicCalibration] = None
        self.camera_extrinsics: Optional[CameraExtrinsicCalibration] = None
        self.H: Optional[np.ndarray] = None
        self.ground_projector: Optional[GroundProjector] = None
        self.camera_info: Optional[Camera] = None
        self.camera: Optional[CameraModel] = None
        self.model = MLModel()
        # register sigint handler
        signal.signal(signal.SIGINT, self._sigint_handler)
        self._jpeg = TurboJPEG()

        self.mode = mode
        self.data_collection = self.mode == "data_collection"

        self.save_every_n_frames = SAVE_EVERY_N_FRAMES
        self.max_log_images = MAX_LOG_IMAGES
        self._frame_idx = 0
        self._logged_images = 0

        self.is_sim: Optional[bool] = is_sim
        self.output_dir: Optional[Path] = None

        if self.is_sim is not None:
            platform = "sim" if self.is_sim else "real"
            self.output_dir = DATA_COLLECTION_ROOT / f"{platform}"

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

        if self.camera is None:
            if self.camera_info is not None and self.camera_intrinsics is not None:
                print("Camera info and intrinsics received, initializing camera model")
                
                self.camera = CameraModel(
                    width=self.camera_info.width,
                    height=self.camera_info.height,
                    K=np.reshape(self.camera_intrinsics.K, (3,3)),
                    D=np.reshape(self.camera_intrinsics.D, (5,)),
                    R=np.reshape(self.camera_intrinsics.R, (3,3)),
                    P=np.reshape(self.camera_intrinsics.P, (3,4)),
                    H=self.H
                )

                if self.H is not None:
                    self.ground_projector = GroundProjector(self.camera)
                    self.model.set_ground_projector(self.ground_projector)

            else:
                print("Still waiting for camera info or intrinsics")
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
            self._frame_idx += 1
            if self._logged_images >= self.max_log_images:
                print(f"Logging limit reached. Increase the limit if you need to collect more data.")
                return
            
            if self._frame_idx % self.save_every_n_frames == 0:
                filename = self.output_dir / f"{self._frame_idx}.png"
                try:
                    cv2.imwrite(str(filename), rectified_img)
                    self._logged_images += 1
                except Exception as e:
                    print(f"Failed to save image {filename}: {e}")
            return

        pwm = self.model.get_wheel_velocities_from_image(rectified_img)

        try:
            await self.pwm_publisher.publish(pwm.to_rawdata())
        except Exception:
            print("Error publishing wheels data")


    async def worker(self):
        switchboard = (await context("switchboard")).navigate(self._robot_name)

        jpeg = await (switchboard / "sensor" / "camera" / "front_center" / "jpeg").until_ready()
        params = await (switchboard / "sensor" / "camera" / "front_center" / "parameters").until_ready()
        info = await (switchboard / "sensor" / "camera" / "front_center" / "info").until_ready()
        extr   = await (switchboard / "sensor" / "camera" / "front_center" / "homography").until_ready()

        self.pwm_publisher = await (switchboard / "actuator" / "wheels" / "base" / "pwm").until_ready()

        jpeg = jpeg.configure(ContextConfig(patient=True))
        params = params.configure(ContextConfig(patient=True))
        info = info.configure(ContextConfig(patient=True))
        extr   = extr.configure(ContextConfig(patient=True))

        await params.subscribe(self.save_camera_intrinsics)
        await info.subscribe(self.save_camera_info)
        await extr.subscribe(self.save_camera_extrinsics)
        await jpeg.subscribe(self.img_cb)


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
        except RuntimeError:
            if not self.is_shutdown:
                print(f"An error occurred while running the event loop: {RuntimeError}")
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["agent","data_collection"], default="agent", help="Run mode: 'agent' to control wheels, 'data_collection' to log images")
    parser.add_argument("--sim", action="store_true", help="Running in simulation")
    parser.add_argument("--real", action="store_true", help="Running on real hardware")
    args = parser.parse_args()

    if args.mode == "data_collection":
        if not (args.sim ^ args.real):
            parser.error("When mode=data_collection, you must specify exactly one of --sim or --real.")

    node = MLAgent(
        mode=args.mode, 
        is_sim=args.sim if args.mode == "data_collection" else None
        )
    node.spin()