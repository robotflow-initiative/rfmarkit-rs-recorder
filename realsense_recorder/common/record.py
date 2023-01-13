import logging
import os
import os.path as osp
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple, Callable

import numpy as np
import pyrealsense2 as rs
import rich
import rich.progress_bar
import yaml

from .config import (
    CALLBACKS, 
    get_device_by_cfg, 
    RealsenseSystemCfg, 
    new_camera_config_by_product_id, 
    RealsenseCameraCfg,
    new_system_config,
)

from .utils import (
    get_datetime_tag,
    enumerate_devices_that_supports_advanced_mode,
)



class RealsenseCameraModel:
    friendly_name: str = ""
    save_path: str = ""
    color_save_path: str = ""
    depth_save_path: str = ""
    bag_save_path: str = ""
    interactive: bool = False
    frame_queue_size: int = 0
    window_name: str = ""

    rs_config: Optional[rs.config] = None
    pipline: Optional[rs.pipeline] = None
    profile: Optional[rs.pipeline_profile] = None
    frame_queue: Optional[rs.frame_queue] = None
    device: Optional[rs.device] = None
    n_frames: Optional[int] = None
    intrinsics_matrix: Optional[List[List[float]]] = None
    progress_bar: Optional[rich.progress_bar.ProgressBar] = None
    align: Optional[rs.align] = None
    colorizer: Optional[rs.colorizer] = None

    # window_name: Optional[str] = None

    def __init__(self,
                 camera_cfg: RealsenseCameraCfg,
                 system_cfg: RealsenseSystemCfg,
                 callbacks: Dict[str, Callable] = None,
                 ):
        self.option = camera_cfg

        # Handle Callback
        # Handle save_path_cb
        if callbacks is not None and CALLBACKS.save_path_cb in callbacks:
            self.save_path = callbacks[CALLBACKS.save_path_cb](camera_cfg, system_cfg)
        else:
            self.save_path = osp.join(system_cfg.base_dir, camera_cfg.sn)

        # Handle friendly name cb
        if callbacks is not None and CALLBACKS.camera_friendly_name_cb in callbacks:
            self.friendly_name = callbacks[CALLBACKS.camera_friendly_name_cb](camera_cfg, system_cfg)
        else:
            self.friendly_name = self.option.sn

        self.color_save_path = osp.join(self.save_path, "color")
        self.depth_save_path = osp.join(self.save_path, "depth")
        self.bag_save_path = osp.join(self.save_path, f'recording_{time.time()}.bag')

        self.use_bag = system_cfg.use_bag
        self.interactive = system_cfg.interactive
        self.frame_queue_size = system_cfg.frame_queue_size
        self.window_name = f"Realsense {camera_cfg.sn}"

        self._reset_variables()

    def open(self):
        self.rs_config = rs.config()

        if self.frame_queue_size > 0:
            self.frame_queue = rs.frame_queue(self.frame_queue_size, keep_frames=True)
        else:
            self.frame_queue = None

        if self.option.use_depth:
            self.rs_config.enable_stream(rs.stream.depth,
                                         self.option.depth[0].width,
                                         self.option.depth[0].height,
                                         eval(self.option.depth[0].format),
                                         self.option.depth[0].fps)
        if self.option.use_imu:
            self.rs_config.enable_stream(rs.stream.accel, rs.format.xyz32f, self.option.imu[0].fps)
            self.rs_config.enable_stream(rs.stream.gyro, rs.format.xyz32f, self.option.imu[0].fps)

        if self.option.use_color:
            self.rs_config.enable_stream(rs.stream.color,
                                         self.option.color[0].width,
                                         self.option.color[0].height,
                                         eval(self.option.color[0].format),
                                         self.option.color[0].fps)
        self.rs_config.enable_device(self.option.sn)
        if self.use_bag:
            self.rs_config.enable_record_to_file(self.bag_save_path)

        self.device = get_device_by_cfg(self.option)
        assert self.device is not None, f"Device {self.option.sn} not found"

        # Depth Preset
        if len(self.option.depth) > 0:
            depth_sensor = self.device.first_depth_sensor()
            preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
            for i in range(int(preset_range.max) + 1):
                visual_preset = depth_sensor.get_option_value_description(rs.option.visual_preset, i)
                if visual_preset == self.option.depth[0].preset:
                    depth_sensor.set_option(rs.option.visual_preset, i)

        # RGB exposure
        sensor = self.device.query_sensors()[1]
        if self.option.color[0].use_auto_exposure:
            sensor.set_option(rs.option.enable_auto_exposure, 1)
        else:
            sensor.set_option(rs.option.exposure, self.option.color[0].exposure)

        # Alignment
        if not self.use_bag:
            self.align = rs.align(rs.stream.color)

        # Interactive
        if self.interactive:
            # self.window_name = f"Realsense {self.option.sn}"
            # cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self.colorizer = rs.colorizer()

    def start(self, delay_ms: int = 0):
        # Pipeline
        time.sleep(delay_ms / 1e3)
        self.pipeline = rs.pipeline()
        self.n_frames = 0
        if self.frame_queue is not None:
            self.profile = self.pipeline.start(self.rs_config, self.frame_queue)
        else:
            self.profile = self.pipeline.start(self.rs_config)
        # Intrinsics
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics_matrix = [
            [self.intrinsics.fx, 0., self.intrinsics.ppx],
            [0., self.intrinsics.fy, self.intrinsics.ppy],
            [0., 0., 1.]
        ]
        logging.debug(f"camera {self.option.sn} intrinsics: {self.intrinsics_matrix}")

    def stop(self, delay_ms: int = 0, clean: bool = False):
        time.sleep(delay_ms / 1e3)
        self.pipeline.stop()
        if clean:
            self.frame_queue = None

    def _get_global_timestamp(self, frames):
        # https://github.com/IntelRealSense/librealsense/issues/5612
        backend_t = frames.get_frame_metadata(rs.frame_metadata_value.backend_timestamp)
        try:
            senseor_t = frames.get_frame_metadata(rs.frame_metadata_value.sensor_timestamp)
            frame_t = frames.get_frame_metadata(rs.frame_metadata_value.frame_timestamp)
            global_t = backend_t - (frame_t - senseor_t)
        except RuntimeError as e:
            # Fallback to backend_t
            global_t = backend_t
        return global_t

    def get_frames(self, timeout_ms=5000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float, int]:
        if self.use_bag:
            self.frame_queue.wait_for_frame()
            ts = time.time()
            return None, None, -1, ts, -1
        else:
            if self.frame_queue is not None:
                ret, frames = self.frame_queue.try_wait_for_frame(timeout_ms=timeout_ms)
                if not ret:
                    return None, None, -1, time.time(), -1
                else:
                    frames = rs.composite_frame(frames)
            else:
                ret, frames = self.pipeline.try_wait_for_frames(timeout_ms=timeout_ms)
                if not ret:
                    return None, None, -1, time.time(), -1
                else:
                    pass

            frame_counter = frames.get_frame_metadata(rs.frame_metadata_value.frame_counter)
            ts = self._get_global_timestamp(frames)
            sys_ts = time.time()

            try:
                aligned_frames = self.align.process(frames)
            except RuntimeError as e:
                return None, None, ts, sys_ts, -1
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if depth_frame:
                if self.interactive:
                    depth_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
                else:
                    depth_image = np.asanyarray(depth_frame.get_data())
            else:
                depth_image = None
            color_image = np.asanyarray(color_frame.get_data()) if color_frame else None
            self.n_frames += 1
            return color_image, depth_image, ts, sys_ts, frame_counter

    def _reset_variables(self):
        self.rs_config: Optional[rs.config] = None
        self.pipline: Optional[rs.pipeline] = None
        self.profile: Optional[rs.pipeline_profile] = None
        self.frame_queue: Optional[rs.frame_queue] = None
        self.device: Optional[rs.device] = None
        self.n_frames: Optional[int] = None
        self.intrinsics_matrix: Optional[List[List[float]]] = None
        self.progress_bar: Optional[rich.progress_bar.ProgressBar] = None
        self.align: Optional[rs.align] = None
        self.colorizer: Optional[rs.colorizer] = None

    def close(self):
        logging.debug(f"closing camera {self.option.sn}")
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError as e:
                pass

        if self.device is not None:
            self.device.hardware_reset()
        self._reset_variables()


class RealsenseSystemModel:
    def __init__(self,
                 system_cfg: RealsenseSystemCfg,
                 camera_cfgs: List[RealsenseCameraCfg],
                 callbacks: Dict[str, Callable] = None):
        self.console = rich.console.Console()
        if callbacks is not None and CALLBACKS.tag_cb in callbacks:
            self.tag = callbacks[CALLBACKS.tag_cb]()
        else:
            self.tag = get_datetime_tag()

        self.options = system_cfg  # Config -> Option
        self.options.base_dir = osp.join(self.options.base_dir, self.tag)
        self.camera_options = sorted(camera_cfgs, key=lambda x: x.ref)
        self.cameras = [RealsenseCameraModel(cfg, system_cfg, callbacks) for cfg in self.camera_options]
        self._prepare_directories()

    def _prepare_directories(self):
        for cam in self.cameras:
            if not osp.lexists(cam.color_save_path):
                os.makedirs(cam.color_save_path)
            if not osp.lexists(cam.depth_save_path):
                os.makedirs(cam.depth_save_path)

    def _set_advanced_mode(self):
        self.console.log("setting advanced mode")
        while True:
            devs = enumerate_devices_that_supports_advanced_mode(rs.context())
            dev = list(filter(lambda x: not rs.rs400_advanced_mode(x).is_enabled(), devs))
            if len(dev) == 0:
                break
            for dev in devs:
                advnc_mode = rs.rs400_advanced_mode(dev)
                advnc_mode.toggle_advanced_mode(True)
                time.sleep(1)

    def open(self):
        for cam in self.cameras:
            cam.open()

    def close(self):
        for cam in self.cameras:
            cam.close()

    def start(self, interval_ms: int = 0):
        num_of_cameras = len(self.cameras)
        ret = []
        with ThreadPoolExecutor(max_workers=num_of_cameras) as executor:
            for idx, cam in enumerate(self.cameras):
                ret.append(executor.submit(lambda: cam.start(delay_ms=interval_ms * (num_of_cameras - idx))))

        list(map(lambda x: x.result(), ret))
        time.sleep(0.1)

    def stop(self, interval_ms: int = 0):
        num_of_cameras = len(self.cameras)
        ret = []
        with ThreadPoolExecutor(max_workers=num_of_cameras) as executor:
            for idx, cam in enumerate(self.cameras):
                ret.append(executor.submit(lambda: cam.stop(interval_ms * (num_of_cameras - idx))))

        list(map(lambda x: x.result(), ret))

    def app(self, *args, **kwargs):
        raise NotImplementedError

    def __del__(self):
        try:
            for cam in self.cameras:
                cam.close()
        finally:
            logging.debug("closing devices")


def load_realsense_cameras_cfg_from_dict(cfg_list: List[Dict[str, Any]]):
    camera_cfgs = []
    for cfg in cfg_list:
        camera_cfg = new_camera_config_by_product_id(cfg["product_id"])
        camera_cfg.load_dict(cfg)
        camera_cfgs.append(camera_cfg)

    return camera_cfgs


def load_realsense_system_cfg_from_dict(cfg_dict: Dict[str, Any]):
    system_cfg = new_system_config()
    system_cfg.load_dict(cfg_dict)
    return system_cfg


def new_realsense_camera_system_from_config(system: type, cfg_dict: Dict[str, Any], callbacks: Dict[str, Callable] = None) -> RealsenseSystemModel:
    camera_cfgs = load_realsense_cameras_cfg_from_dict(cfg_dict["cameras"])
    system_cfgs = load_realsense_system_cfg_from_dict(cfg_dict["system"])
    system = system(system_cfgs, camera_cfgs, callbacks)
    return system


def new_realsense_camera_system_from_yaml_file(system: type, path_to_yaml_file: str, callbacks: Dict[str, Callable] = None) -> RealsenseSystemModel:
    with open(path_to_yaml_file, 'r') as f:
        buf = yaml.load(f, Loader=yaml.SafeLoader)
    return new_realsense_camera_system_from_config(system, buf["realsense"], callbacks)
