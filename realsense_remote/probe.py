import enum
import json
import logging
import multiprocessing as mp
from multiprocessing import context
import signal
import sys
import time
from typing import List, Union, Dict
import argparse
import os
import os.path as osp
import json

import pyrealsense2 as rs
import cv2
from tqdm import tqdm
import numpy as np
import tqdm
from flask import Flask, request, Response
from gevent import pywsgi


def enumerate_connected_devices(context):
    """
    Enumerate the connected Intel RealSense devices

    Parameters:
    -----------
    context 	   : rs.context()
                     The context created for using the realsense library

    Return:
    -----------
    connect_device : array
                     Array of (serial, product-line) tuples of devices which are connected to the PC

    """
    connect_device = []

    for d in context.devices:
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            serial = d.get_info(rs.camera_info.serial_number)
            product_line = d.get_info(rs.camera_info.product_line)
            device_info = (serial, product_line)  # (serial_number, product_line)
            connect_device.append(device_info)
    return connect_device

def main():
    EXEC_STRING = "python -m realsense_remote.main"
    context = rs.context()
    available_devices = enumerate_connected_devices(context)
    enumerated_device_types = set([device[1] for device in available_devices])

    device_by_type = {tag:list(filter(lambda x: x[1] == tag, available_devices)) for tag in enumerated_device_types}

    # for tag in device_by_type.keys():
    #     print(f"There are {len(device_by_type[tag])} {tag} devices connected to your system")
    
    port = 5050
    for tag in device_by_type.keys():
        for idx in range(len(device_by_type[tag])):
            print(f"{EXEC_STRING} --device={tag} --idx={idx} --port={port}")
            port += 1

if __name__ == '__main__':
    main()