import threading
import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import os.path as osp
import time
import queue
import json

from datetime import datetime

import uuid
from tqdm import tqdm

# if using L515, depth scale 0.00025, that means the depth image needs to be divided with 4

# if using D435
# depth_scale = 0.001
use_depth = True
show_depth = True

# L500
# L515_depth_resolution_width = 640  # pixels
# L515_depth_resolution_height = 480  # pixels
# L515_color_resolution_width = 640
# L515_color_resolution_height = 480
L515_depth_resolution_width = 1024  # pixels
L515_depth_resolution_height = 768  # pixels
L515_color_resolution_width = 1280
L515_color_resolution_height = 720
L515_depth_frame_rate = 30
L515_color_frame_rate = 30

# for D400
# D435_depth_resolution_width = 848  # pixels
# D435_depth_resolution_height = 480  # pixels
# color_resolution_width = 848  # pixels
# color_resolution_height = 480  # pixels
D435_depth_resolution_width = 1280  # pixels
D435_depth_resolution_height = 720  # pixels
color_resolution_width = 1280  # pixels
color_resolution_height = 720  # pixels
D435_frame_rate = 30  # fps

# Enable the streams from all the intel realsense devices
L515_rs_config = rs.config()
if use_depth:
    L515_rs_config.enable_stream(rs.stream.depth, L515_depth_resolution_width, L515_depth_resolution_height,
                                 rs.format.z16,
                                 L515_depth_frame_rate)
L515_rs_config.enable_stream(rs.stream.color, L515_color_resolution_width, L515_color_resolution_height, rs.format.bgr8,
                             L515_color_frame_rate)

L400_rs_config = rs.config()
if use_depth:
    L400_rs_config.enable_stream(rs.stream.depth, D435_depth_resolution_width, D435_depth_resolution_height,
                                 rs.format.z16, D435_frame_rate)
L400_rs_config.enable_stream(rs.stream.color, color_resolution_width, color_resolution_height, rs.format.bgr8,
                             D435_frame_rate)


class Recorder:
    def __init__(self, device, save_path, command, pipeline, parent_queue, parent2child, tag, using_L515=False):
        self.device = device
        self.save_path = save_path
        self.device_save_path = osp.join(save_path, tag, device[0])
        self.command = command
        self.queue = parent_queue
        self.parent2child = parent2child
        self.pipeline = pipeline
        self.using_L515 = using_L515
        self.frame_queue = queue.Queue()
        # self.display_queue = queue.Queue()
        self.count = 0
        self.capture_thread = threading.Thread(target=self.save_frames)
        self.save_thread = threading.Thread(target=self.capture_frames)
        # self.display_thread = threading.Thread(target=self.display_frames)
        self.init_frame_number = -1

    def run(self):
        self.capture_thread.start()
        # self.display_thread.start()
        self.save_thread.start()

        self.frame_queue.join()
        self.capture_thread.join()
        # self.display_thread.join()
        self.save_thread.join()
        print('Finish recording!')

    def display_frames(self):
        colorizer = rs.colorizer()
        while True:
            if self.command.value == 2:
                break

            try:
                color_image, depth_frame = self.display_queue.get(block=True, timeout=1.0)
            except queue.Empty:
                continue

            # Show images
            cv2.namedWindow('RealSense_{}'.format(self.device[0]), cv2.WINDOW_AUTOSIZE)

            if use_depth:
                colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                # images = np.hstack((color_image, colorized_depth))
                mix = cv2.addWeighted(color_image, 0.5, colorized_depth, 0.5, 0)
            else:
                mix = color_image
            cv2.imshow('RealSense_{}'.format(self.device[0]), mix)
            cv2.waitKey(1)

            if self.using_L515:
                time.sleep(0.02)
            else:
                time.sleep(0.02)

    def save_frames(self):
        while True:
            try:
                color_image, depth_image, frame_number, timestamp, sys_timestamp = self.frame_queue.get(block=True,
                                                                                                        timeout=1.0)
            except queue.Empty:
                if self.command.value == 2:
                    print('Device {} quit saving now!'.format(self.device[0]))
                    break
                continue

            cv2.imwrite(osp.join(self.device_save_path, 'color',
                                 '{:06d}_{}_{}.jpg'.format(frame_number, timestamp, sys_timestamp)), color_image)
            if use_depth:
                if self.using_L515:
                    cv2.imwrite(osp.join(self.device_save_path, 'depth',
                                         '{:06d}_{}_{}.png'.format(frame_number, timestamp, sys_timestamp)),
                                (depth_image / 4).astype(np.uint16))
                else:
                    cv2.imwrite(osp.join(self.device_save_path, 'depth',
                                         '{:06d}_{}_{}.png'.format(frame_number, timestamp, sys_timestamp)),
                                depth_image)

            time.sleep(0.01)

        print('Device {} finished saving!'.format(self.device[1]))


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
    parser = argparse.ArgumentParser(description='Recorder')
    parser.add_argument('--device', type=str, choices=['L500', 'D400', 'all'], help='The cameras to use, seperated by comma e.g. "L500,D400"', default='all')
    parser.add_argument('--tag', type=str, help='The save tag', default='default')
    parser.add_argument('--idx', type=int, help='The index of the cam', default='0')
    parser.add_argument('--base_dir', type=str, help='The path to save frames', default='./realsense_data')
    args = parser.parse_args()

    # >>>>>> 3RDPARTY SYNC
    BASE_DIR = args.base_dir
    # subpath = args.tag + '-' + str(uuid.uuid1()).split('-')[0] + '-' + datetime.now().strftime("%Y-%m-%d_%H%M%S")
    subpath = args.tag
    save_path = osp.join(BASE_DIR, subpath)
    # <<<<<<

    context = rs.context()
    available_devices = enumerate_connected_devices(context)
    if args.device == 'all':
        valid_devices = available_devices
    else:
        valid_devices = list(filter(lambda x: x[1] in args.device.split(','), available_devices))
    device = valid_devices[int(args.idx)]
    print('Find {} valid {} devices, using the {}-th device'.format(len(valid_devices), args.device, args.idx))

    device_save_path = osp.join(save_path, device[0])
    if not os.path.lexists(osp.join(device_save_path, 'color')):
        os.makedirs(osp.join(device_save_path, 'color'))
    if not os.path.lexists(osp.join(device_save_path, 'depth')):
        os.makedirs(osp.join(device_save_path, 'depth'))

    # Configure depth and color streams
    pipeline = rs.pipeline()

    device_serial = device[0]
    product_line = device[1]

    if product_line == "L500":
        using_L515 = True
        # Enable L515 device
        L515_rs_config.enable_device(device_serial)
        # dev.first_depth_sensor().set_option(rs.option.inter_cam_sync_mode, 1 if is_master else 0)
        pipeline_profile = pipeline.start(L515_rs_config)
    else:
        # Enable D400 device
        using_L515 = False
        L400_rs_config.enable_device(device_serial)
        pipeline_profile = pipeline.start(L400_rs_config)

    # set preset
    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
    for i in range(int(preset_range.max) + 1):
        visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset, i)
        if visulpreset == "Short Range":
            depth_sensor.set_option(rs.option.visual_preset, i)

    # color sensor
    sensor = pipeline_profile.get_device().query_sensors()[1]
    if using_L515:
        # sensor.set_option(rs.option.exposure, 300.0)
        sensor.set_option(rs.option.enable_auto_exposure, 1)
    else:
        sensor.set_option(rs.option.enable_auto_exposure, 1)
        # sensor.set_option(rs.option.exposure, 15.0)

    # Set Sync mode
    # sensor = pipeline_profile.get_device().first_depth_sensor()
    # sensor.set_option(rs.option.inter_cam_sync_mode, 1 if is_master else 0)

    # store intrinsics
    init_profile = pipeline_profile.get_stream(rs.stream.color)
    intrinsics = init_profile.as_video_stream_profile().get_intrinsics()
    '''Convert intrinsic data to dict (readable in Open3D)'''
    mat = [intrinsics.fx, 0, 0, 0, intrinsics.fy, 0, intrinsics.ppx, intrinsics.ppy, 1]
    intrinsics_dict = {'width': intrinsics.width, 'height': intrinsics.height, 'intrinsic_matrix': mat}
    if not os.path.lexists(osp.join(device_save_path, 'realsense_intrinsic.json')):
        with open(os.path.join(device_save_path, 'camera_intrinsic.json'), 'w') as f:
            json.dump(intrinsics_dict, f, sort_keys=False,
                      indent=4,
                      ensure_ascii=False)

    # aligner
    # must align depth frame to color frame, because the depth camera has offset with rgb camera in realsense
    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    try:
        n_frames = 0
        with tqdm(ncols=0) as pbar:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                # depth_frame_org = frames.get_depth_frame()
                if use_depth:
                    depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                # color_frame = aligned_frames.get_color_frame()

                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())

                if use_depth and show_depth:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

                # Show images
                show = False
                if show:
                    cv2.namedWindow('RealSense_{}'.format(device[0]), cv2.WINDOW_AUTOSIZE)
                    # images = np.hstack((color_image, colorized_depth))
                    if use_depth and show_depth:
                        mix = cv2.addWeighted(color_image, 0.5, colorized_depth, 0.5, 0)
                    else:
                        mix = color_image
                    cv2.imshow('RealSense_{}'.format(device[0]), mix)
                    cv2.waitKey(1)

                ts = time.time()
                cv2.imwrite(osp.join(device_save_path, 'color', f'{n_frames}_{ts}.bmp'), color_image)
                # cv2.imwrite(osp.join(device_save_path, 'depth', f'{n_frames}_{ts}.png'), depth_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                np.save(osp.join(device_save_path, 'depth', f'{n_frames}_{ts}.npy'), depth_image)

                n_frames += 1

                pbar.update(1)

    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    main()
