from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Union, Optional

import pyrealsense2 as rs
from rich.console import Console

from .interaction import must_parse_cli_bool, must_parse_cli_sel, must_parse_cli_string, must_parse_cli_int

_ = {
        "product": "L500",
        "sn": "f0220485",
        "depth": {
            "width": 1024,
            "height": 768,
            "fps": 30,
            "format": "rs.format.z16",
            "preset": "Short Range",
        },
        "color": {
            "width": 1280,
            "height": 720,
            "fps": 30,
            "color_format": "rs.format.bgr8"
        },
        "imu": {
            "fps": 100
        },
        "timeout_ms": 5000,
        "endpoint": {
            "url": "http://10.233.234.1:5050"
        }
    },

class CALLBACKS:
    save_path_cb = "save_path_cb"
    tag_cb = "tag_cb"
    camera_friendly_name_cb = "camera_friendly_name_cb"


class BaseCfg:
    def get_dict(self) -> Dict[Any, Any]:
        raise NotImplemented

    def load_dict(self, src: Dict[str, Any]) -> None:
        raise NotImplemented

    def configure_from_keyboard(self) -> bool:
        raise NotImplemented


@dataclass
class RealsenseCameraColorCfg:
    resolution: Tuple[int, int] = (0, 0)
    width: int = 0
    height: int = 0
    fps: int = 0
    format: str = ""
    exposure: int = -1

    def __post_init__(self):
        self.width = self.resolution[0]
        self.height = self.resolution[1]
        pass

    @property
    def use_auto_exposure(self):
        return self.exposure == -1


@dataclass
class RealsenseCameraDepthCfg:
    resolution: Tuple[int, int] = (0, 0)
    width: int = 0
    height: int = 0
    fps: int = 0
    format: str = ""
    preset: str = ""

    def __post_init__(self):
        self.width = self.resolution[0]
        self.height = self.resolution[1]
        pass


@dataclass
class RealsenseCameraIMUCfg:
    fps: int = 0


@dataclass
class RealsenseCameraEndpointCfg:
    url: str = ""


class RealsenseCameraCfg(BaseCfg):
    __PRODUCT_ID__: str
    __PRODUCT_LINE__: str
    __COLOR_RESOLUTION_CANDIDATES__: Tuple[Tuple[int]]
    __COLOR_RESOLUTION_DEFAULT_SEL__: int
    __COLOR_FPS_CANDIDATES__: Tuple[int]
    __COLOR_FPS_DEFAULT_SEL__: int
    __COLOR_FORMAT_CANDIDATES__: Tuple[str]
    __COLOR_FORMAT_DEFAULT_SEL__: int
    __DEPTH_RESOLUTION_CANDIDATES__: Tuple[Tuple[int]]
    __DEPTH_RESOLUTION_DEFAULT_SEL__: int
    __DEPTH_FPS_CANDIDATES__: Tuple[int]
    __DEPTH_FPS_DEFAULT_SEL__: int
    __DEPTH_PRESET_CANDIDATES__: Tuple[str]
    __DEPTH_PRESET_DEFAULT_SEL__: int
    __DEPTH_FORMAT_CANDIDATES__: Tuple[str]
    __DEPTH_FORMAT_DEFAULT_SEL__: int
    __IMU_FPS_CANDIDATES__: Tuple[int]
    __IMU_FPS_DEFAULT_SEL__: int

    ref: int = -1
    sn: str = ""
    depth: List[RealsenseCameraDepthCfg] = None
    color: List[RealsenseCameraColorCfg] = None
    imu: Union[None, List[RealsenseCameraIMUCfg]] = None
    endpoint: RealsenseCameraEndpointCfg = None

    def __init__(self):
        self.depth = []
        self.color = []
        self.imu = []

    @property
    def use_depth(self):
        return len(self.depth) > 0

    @property
    def use_color(self):
        return len(self.color) > 0

    @property
    def use_imu(self):
        return self.imu is not None and len(self.imu) > 0

    def _get_depth_config(self) -> List[Dict[str, Any]]:
        res = []
        for o in self.depth:
            res.append({
                "width": o.width,
                "height": o.height,
                "fps": o.fps,
                "format": o.format,
                "preset": o.preset
            })

        return res

    def _load_depth_config(self, cfg_list: List[Dict[str, Any]]):
        self.depth = []
        for o in cfg_list:
            cfg = RealsenseCameraDepthCfg()
            cfg.width = o["width"]
            cfg.height = o["height"]
            cfg.resolution = (o["width"], o["height"])
            cfg.fps = o["fps"]
            cfg.format = o["format"]
            cfg.preset = o["preset"]
            self.depth.append(cfg)

    def _get_color_config(self) -> List[Dict[str, Any]]:
        res = []
        for o in self.color:
            res.append({
                "width": o.width,
                "height": o.height,
                "fps": o.fps,
                "format": o.format,
                "exposure": o.exposure
            })

        return res

    def _load_color_config(self, cfg_list: List[Dict[str, Any]]):
        self.color = []
        for o in cfg_list:
            cfg = RealsenseCameraColorCfg()
            cfg.width = o["width"]
            cfg.height = o["height"]
            cfg.resolution = (o["width"], o["height"])
            cfg.fps = o["fps"]
            cfg.format = o["format"]
            cfg.exposure = o["exposure"]
            self.color.append(cfg)

    def _get_imu_config(self) -> List[Dict[str, Any]]:
        res = []
        for o in self.imu:
            res.append({
                "fps": o.fps
            })

        return res

    def _load_imu_config(self, cfg_list: List[Dict[str, Any]]):
        if self.imu is not None:
            self.imu = []
            for o in cfg_list:
                cfg = RealsenseCameraIMUCfg()
                cfg.fps = o["fps"]
                self.imu.append(cfg)

    def _get_endpoint_config(self) -> Dict[str, Any]:
        if self.endpoint is not None:
            return {
                "url": self.endpoint.url
            }
        else:
            return {}

    def get_dict(self):
        raise NotImplemented

    def load_dict(self, src: Dict[str, Any]):
        raise NotImplemented

    def configure_from_keyboard(self) -> bool:

        console = Console()

        console.rule(f"[bold blue] {self.__repr__()}")
        enable_this_camera = must_parse_cli_bool("Enable this camera?", default_value=True)
        if not enable_this_camera:
            return False

        enable_color_stream = must_parse_cli_bool("Enable Color Stream?", default_value=True)
        if enable_color_stream:
            resolution_sel = must_parse_cli_sel("Select color resolution", self.__COLOR_RESOLUTION_CANDIDATES__,
                                                default_value=self.__COLOR_RESOLUTION_DEFAULT_SEL__)
            fps_sel = must_parse_cli_sel("Select color fps", self.__COLOR_FPS_CANDIDATES__, default_value=self.__COLOR_FPS_DEFAULT_SEL__)
            format_sel = must_parse_cli_sel("Select color format", self.__COLOR_FORMAT_CANDIDATES__,
                                            default_value=self.__COLOR_FORMAT_DEFAULT_SEL__)
            exposure_sel = must_parse_cli_int("Set exposure [ 0 - 10000 ], -1 for auto exposure", min=-1, max=10001, default_value=-1)
            self.color.append(RealsenseCameraColorCfg(resolution=self.__COLOR_RESOLUTION_CANDIDATES__[resolution_sel],
                                                      fps=self.__COLOR_FPS_CANDIDATES__[fps_sel],
                                                      format=self.__COLOR_FORMAT_CANDIDATES__[format_sel],
                                                      exposure=exposure_sel))

        enable_depth_stream = must_parse_cli_bool("Enable Depth Stream?", default_value=True)
        if enable_depth_stream:
            resolution_sel = must_parse_cli_sel("Select depth resolution", self.__DEPTH_RESOLUTION_CANDIDATES__,
                                                default_value=self.__DEPTH_RESOLUTION_DEFAULT_SEL__)
            fps_sel = must_parse_cli_sel("Select depth fps", self.__DEPTH_FPS_CANDIDATES__, default_value=self.__DEPTH_FPS_DEFAULT_SEL__)
            preset_sel = must_parse_cli_sel("Select depth preset", self.__DEPTH_PRESET_CANDIDATES__,
                                            default_value=self.__DEPTH_PRESET_DEFAULT_SEL__)
            format_sel = must_parse_cli_sel("Select depth format", self.__DEPTH_FORMAT_CANDIDATES__,
                                            default_value=self.__DEPTH_FORMAT_DEFAULT_SEL__)
            self.depth.append(RealsenseCameraDepthCfg(resolution=self.__DEPTH_RESOLUTION_CANDIDATES__[resolution_sel],
                                                      fps=self.__DEPTH_FPS_CANDIDATES__[fps_sel],
                                                      preset=self.__DEPTH_PRESET_CANDIDATES__[preset_sel],
                                                      format=self.__DEPTH_FORMAT_CANDIDATES__[format_sel]))

        if self.imu is not None:
            enable_imu_stream = must_parse_cli_bool("Enable IMU Stream?", default_value=False)
            if enable_imu_stream:
                fps_sel = must_parse_cli_sel("Select IMU fps", self.__IMU_FPS_CANDIDATES__, default_value=self.__IMU_FPS_DEFAULT_SEL__)
                self.imu.append(RealsenseCameraIMUCfg(fps=self.__IMU_FPS_CANDIDATES__[fps_sel]))

        console.print("Summary: ")
        console.print(self.color)
        console.print(self.depth)
        console.print(self.imu)
        console.rule(f"[bold blue] End ")

        return True


@dataclass
class RealsenseD435CameraCfg(RealsenseCameraCfg):
    __PRODUCT_ID__: str = "0B07"
    __PRODUCT_LINE__: str = "D400"

    __COLOR_RESOLUTION_CANDIDATES__: Tuple[Tuple[int]] = (
        (320, 180),
        (320, 240),
        (424, 240),
        (640, 360),
        (640, 480),
        (848, 480),
        (960, 540),
        (1280, 720),
    )
    __COLOR_RESOLUTION_DEFAULT_SEL__: int = 7
    __COLOR_FPS_CANDIDATES__: Tuple[int] = (
        6,
        15,
        30,
        60,
    )
    __COLOR_FPS_DEFAULT_SEL__: int = 2
    __COLOR_FORMAT_CANDIDATES__: Tuple[str] = (
        "rs.format.yuyv",
        "rs.format.bgr8",
        "rs.format.rgba8",
        "rs.format.bgra8",
        "rs.format.y16",
        "rs.format.rgb8",
        "rs.format.raw16",
    )
    __COLOR_FORMAT_DEFAULT_SEL__: int = 3
    __COLOR_EXPOSURE_RANGE__: Tuple[int, int] = (0, 10000)
    __DEPTH_RESOLUTION_CANDIDATES__: Tuple[Tuple[int]] = (
        (256, 144),
        (424, 240),
        (480, 270),
        (640, 360),
        (640, 400),
        (640, 480),
        (848, 100),
        (848, 480),
        (1280, 720),
    )
    __DEPTH_RESOLUTION_DEFAULT_SEL__: int = 8
    __DEPTH_FPS_CANDIDATES__: Tuple[int] = (
        6,
        15,
        30,
        60,
        90,
        100,
        300
    )
    __DEPTH_FPS_DEFAULT_SEL__: int = 2
    __DEPTH_PRESET_CANDIDATES__: Tuple[str] = (
        "Default",
        "Hand",
        "High Accuracy",
        "High Density",
        "Medium Density"
    )
    __DEPTH_PRESET_DEFAULT_SEL__: int = 0
    __DEPTH_FORMAT_CANDIDATES__: Tuple[str] = (
        "rs.format.z16",
    )
    __DEPTH_FORMAT_DEFAULT_SEL__: int = 0

    def __repr__(self):
        return f"<D435 Camera Config, SN={self.sn}>"

    def __post_init__(self):
        super().__init__()
        self.imu = None
        pass

    def get_dict(self) -> Dict[str, Any]:
        return {
            "ref": self.ref,
            "product_line": self.__PRODUCT_LINE__,
            "product_id": self.__PRODUCT_ID__,
            "sn": self.sn,
            "depth": self._get_depth_config(),
            "color": self._get_color_config(),
            "endpoint": self._get_endpoint_config()
        }

    def load_dict(self, cfg_dict) -> None:
        self.ref = cfg_dict['ref']
        self.sn = cfg_dict['sn']
        self._load_depth_config(cfg_dict['depth'])
        self._load_color_config(cfg_dict['color'])


@dataclass
class RealsenseL515CameraCfg(RealsenseCameraCfg):
    __PRODUCT_ID__: str = "0B64"
    __PRODUCT_LINE__: str = "L500"

    __COLOR_RESOLUTION_CANDIDATES__: Tuple[Tuple[int]] = (
        (640, 360),
        (640, 480),
        (960, 540),
        (1280, 720),
        (1920, 1080),
    )
    __COLOR_RESOLUTION_DEFAULT_SEL__: int = 4
    __COLOR_FPS_CANDIDATES__: Tuple[int] = (
        6,
        15,
        30,
        60,
    )
    __COLOR_FPS_DEFAULT_SEL__: int = 2
    __COLOR_FORMAT_CANDIDATES__: Tuple[str] = (
        "rs.format.yuyv",
        "rs.format.bgr8",
        "rs.format.rgba8",
        "rs.format.bgra8",
        "rs.format.y16",
        "rs.format.rgb8"
    )
    __COLOR_FORMAT_DEFAULT_SEL__: int = 3
    __COLOR_EXPOSURE_RANGE__: Tuple[int, int] = (0, 10000)
    __DEPTH_RESOLUTION_CANDIDATES__: Tuple[Tuple[int]] = (
        (320, 240),
        (640, 480),
        (1024, 768)
    )
    __DEPTH_RESOLUTION_DEFAULT_SEL__: int = 2
    __DEPTH_FPS_CANDIDATES__: Tuple[int] = (
        30,
    )
    __DEPTH_FPS_DEFAULT_SEL__: int = 0
    __DEPTH_PRESET_CANDIDATES__: Tuple[str] = (
        "No Ambient Light",
        "Low Ambient Light",
        "Short Range",
        "Max Range",
    )
    __DEPTH_PRESET_DEFAULT_SEL__: int = 2
    __DEPTH_FORMAT_CANDIDATES__: Tuple[str] = (
        "rs.format.z16",
    )
    __DEPTH_FORMAT_DEFAULT_SEL__: int = 0
    __IMU_FPS_CANDIDATES__: Tuple[int] = (
        100,
        200,
        400
    )
    __IMU_FPS_DEFAULT_SEL__: int = 0

    def __repr__(self):
        return f"<L515 Camera Config, SN={self.sn}>"

    def __post_init__(self):
        super().__init__()
        pass

    def get_dict(self) -> Dict[str, Any]:
        return {
            "ref": self.ref,
            "product_line": self.__PRODUCT_LINE__,
            "product_id": self.__PRODUCT_ID__,
            "sn": self.sn,
            "depth": self._get_depth_config(),
            "color": self._get_color_config(),
            "imu": self._get_imu_config(),
            "endpoint": self._get_endpoint_config()
        }

    def load_dict(self, cfg_dict) -> None:
        self.ref = cfg_dict['ref']
        self.sn = cfg_dict['sn']
        self._load_depth_config(cfg_dict['depth'])
        self._load_color_config(cfg_dict['color'])
        self._load_imu_config(cfg_dict['imu'])


_CAMERA_MAPPING_BY_PRODUCT_ID = {
    "0B64": RealsenseL515CameraCfg,
    "0B07": RealsenseD435CameraCfg,
}

_CAMERA_MAPPING_BY_NAME = {
    "L515": RealsenseL515CameraCfg,
    "D435": RealsenseD435CameraCfg,
}


def new_camera_config_by_name(name: str) -> RealsenseCameraCfg:
    if name in _CAMERA_MAPPING_BY_NAME.keys():
        return _CAMERA_MAPPING_BY_NAME[name]()
    else:
        raise ValueError(f"{name} is not a valid camera name")


def new_camera_config_by_product_id(product_id: str) -> RealsenseCameraCfg:
    if product_id in _CAMERA_MAPPING_BY_PRODUCT_ID.keys():
        return _CAMERA_MAPPING_BY_PRODUCT_ID[product_id]()
    else:
        raise ValueError(f"{product_id} is not a valid product id")


def new_camera_config_by_device(dev: rs.device):
    product_id = dev.get_info(rs.camera_info.product_id)
    cfg = new_camera_config_by_product_id(product_id)
    cfg.sn = dev.get_info(rs.camera_info.serial_number)
    return cfg


@dataclass
class RealsenseSystemCfg(BaseCfg):
    base_dir: str = ""
    interactive: bool = False
    use_bag: bool = False
    frame_queue_size: int = 100
    interval_ms: int = 100

    def get_dict(self) -> Dict[str, Any]:
        return {
            "base_dir": self.base_dir,
            "interactive": self.interactive,
            "use_bag": self.use_bag,
            "frame_queue_size": self.frame_queue_size,
            "interval_ms": self.interval_ms
        }

    def load_dict(self, cfg_dict) -> None:
        self.base_dir = cfg_dict['base_dir']
        self.interactive = cfg_dict['interactive']
        self.use_bag = cfg_dict['use_bag']
        self.frame_queue_size = cfg_dict['frame_queue_size']
        self.interval_ms = cfg_dict['interval_ms']

    def configure_from_keyboard(self):
        console = Console()

        console.rule(f"[bold blue] System Configuration")
        configura_system = must_parse_cli_bool("Configure system?", default_value=True)
        if not configura_system:
            return False

        base_dir_sel = must_parse_cli_string("Base Directory", default_value="./realsense_data")
        interactive_sel = must_parse_cli_bool("Interactive?", default_value=False)
        use_bag_sel = must_parse_cli_bool("Use Bag?", default_value=False)
        frame_queue_size_sel = must_parse_cli_int("Frame Queue Size",min=0, max=1001, default_value=100)
        interval_ms_sel = must_parse_cli_int("Interval", min=0, max=1001, default_value=0)


        self.base_dir = base_dir_sel
        self.interactive = interactive_sel
        self.use_bag = use_bag_sel
        self.frame_queue_size = frame_queue_size_sel
        self.interval_ms = interval_ms_sel

        console.print("Summary: ")
        console.print(self.get_dict())
        console.rule(f"[bold blue] End ")


def new_system_config() -> RealsenseSystemCfg:
    return RealsenseSystemCfg()


def get_device_by_cfg(cfg: RealsenseCameraCfg) -> Optional[rs.device]:
    context = rs.context()

    for d in context.devices:
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            sn = d.get_info(rs.camera_info.serial_number)
            if sn == cfg.sn:
                return d

    return None
