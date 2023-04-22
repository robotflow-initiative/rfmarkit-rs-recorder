import datetime
from typing import Dict, Any, List

import numpy as np
import pyrealsense2 as rs
import cv2

from markit_rs_recorder.common import new_camera_config_by_device, new_system_config


def enumerate_connected_devices(ctx: rs.context):
    """
    Enumerate the connected Intel RealSense devices

    Parameters:
    -----------
    ctx      	   : rs.context()
                     The context created for using the realsense library

    Return:
    -----------
    connect_device : array
                     Array of (serial, product-line) tuples of devices which are connected to the PC

    """
    connect_device = []

    for d in ctx.devices:
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            serial = d.get_info(rs.camera_info.serial_number)
            product_line = d.get_info(rs.camera_info.product_line)
            device_info = (serial, product_line)  # (serial_number, product_line)
            connect_device.append(device_info)
    return connect_device


def enumerate_devices_that_supports_advanced_mode(ctx: rs.context) -> List[rs.device]:
    DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A", "0B5C"]
    devices = ctx.query_devices()
    res = []
    for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
            res.append(dev)
    return res


def configure_realsense_system_from_keyboard() -> Dict[str, Dict[str, Any]]:
    ctx = rs.context()

    res = {
        'realsense': {
            'cameras': [],
            'system': {}
        }
    }
    ref = 0
    for d in ctx.devices:
        camera_cfg = new_camera_config_by_device(d)
        ret = camera_cfg.configure_from_keyboard()
        if ret:
            camera_cfg.ref = ref
            res['realsense']['cameras'].append(camera_cfg.get_dict())
            ref += 1

    system_cfg = new_system_config()
    system_cfg.configure_from_keyboard()
    res['realsense']['system'] = system_cfg.get_dict()
    return res


def get_datetime_tag() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

def rvec_tvec_to_matrix(rvec: np.ndarray=None, tvec: np.ndarray=None) -> np.ndarray:
    """
    Convert rotation vector and translation vector to transformation matrix
    Args:
        rvec:  rotation vector
        tvec:  translation vector

    Returns:
        transformation matrix
    """
    R, _ = cv2.Rodrigues(rvec) if rvec is not None else (np.eye(3), None)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec if tvec is not None else np.zeros(3)
    return T

def R_T_to_matrix(R: np.ndarray, T:np.ndarray) -> np.ndarray:
    """
    Convert R matrix and T matrix to transformation matrix
    Args:
        R:  rotation
        T:  translation

    Returns:
        transformation matrix
    """
    res = np.eye(4)
    res[:3, :3] = R if R is not None else np.eye(3)
    res[:3, 3] = T if T is not None else np.zeros(3)
    return res