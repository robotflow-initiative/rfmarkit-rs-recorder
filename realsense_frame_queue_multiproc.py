import multiprocessing
import threading
import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
import os.path as osp
import time
from multiprocessing import Process
import sys
import queue

from datetime import datetime
from tcpbroker import broadcast_command

import uuid

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
# color_resolution_width = 1280  # pixels
# color_resolution_height = 720  # pixels
D435_depth_resolution_width = 848  # pixels
D435_depth_resolution_height = 480  # pixels
color_resolution_width = 848  # pixels
color_resolution_height = 480  # pixels
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

    def capture_frames(self):
        # aligner
        # must align depth frame to color frame, because the depth camera has offset with rgb camera in realsense
        align = rs.align(rs.stream.color)
        colorizer = rs.colorizer()

        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
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
                cv2.namedWindow('RealSense_{}'.format(self.device[0]), cv2.WINDOW_AUTOSIZE)
                # images = np.hstack((color_image, colorized_depth))
                if use_depth and show_depth:
                    mix = cv2.addWeighted(color_image, 0.5, colorized_depth, 0.5, 0)
                else:
                    mix = color_image
                cv2.imshow('RealSense_{}'.format(self.device[0]), mix)
                cv2.waitKey(1)

                # pres Space for manual image capture (ascii for Space is 32)
                if self.command.value == 0:

                    # find the number of stored images to avoid overlap
                    count = 0
                    for filename in os.listdir(osp.join(self.device_save_path, 'color')):
                        if filename.endswith('.png'):
                            count += 1
                    # print(count),

                    cv2.imwrite(osp.join(self.device_save_path, 'color', '%06d.jpg' % count), color_image)
                    if use_depth:
                        cv2.imwrite(osp.join(self.device_save_path, 'depth', '%06d.png' % count), depth_image)

                # pres Enter for automatic image capture (ascii for Enter is 13)
                if self.command.value == 1:
                    cv2.destroyAllWindows()
                    self.queue.get()

                    while True:
                        # Wait for a coherent pair of frames: depth and color
                        frames = self.pipeline.wait_for_frames()
                        sys_timestamp = time.time()
                        timestamp = frames.timestamp
                        try:
                            aligned_frames = align.process(frames)
                        except Exception as e:
                            print('Device: {}, error in frame {}, {}'.format(self.device[0], frames.frame_number, e))
                            continue

                        if self.init_frame_number == -1:
                            self.init_frame_number = frames.frame_number

                        if use_depth:
                            depth_frame = aligned_frames.get_depth_frame()
                        color_frame = aligned_frames.get_color_frame()  # should get from aligned_frames

                        if not color_frame:
                            continue

                        color_image = np.asanyarray(color_frame.get_data())
                        if use_depth:
                            depth_image = np.asanyarray(depth_frame.get_data())
                        else:
                            depth_image = None

                        # TODO: fix bug in display
                        # if not self.display_queue.not_empty:
                        #     self.display_queue.put((color_image, depth_frame))
                        self.frame_queue.put((color_image, depth_image,
                                              frames.frame_number - self.init_frame_number, timestamp, sys_timestamp))

                        if self.using_L515:
                            time.sleep(0.01)
                        else:
                            # time.sleep(0.0033)
                            time.sleep(0.002)

                        if self.command.value == 2:
                            print('Device {} quit recording now!'.format(self.device[0]))
                            break

                if self.command.value == 2:
                    cv2.destroyAllWindows()
                    break

        finally:
            # Stop streaming
            self.pipeline.stop()


def camera_run(device, save_path, command, queue, parent2child, child2parent, tag, is_master=False):
    child2parent.value = child2parent.value + 1
    while parent2child.value == 0:
        pass

    device_save_path = osp.join(save_path, tag, device[0])
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
        # sensor.set_option(rs.option.enable_auto_exposure, 1)
        sensor.set_option(rs.option.exposure, 4.0)
        # sensor.set_option(rs.option.exposure, 10.0)
    # sensor.set_option(rs.option.exposure, 30.0)
    # sensor.set_option(rs.option.exposure, 100.0)

    # Set Sync mode
    # sensor = pipeline_profile.get_device().first_depth_sensor()
    # sensor.set_option(rs.option.inter_cam_sync_mode, 1 if is_master else 0)

    recorder = Recorder(device, save_path, command, pipeline, queue, parent2child, tag, using_L515)
    recorder.run()

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
    parser.add_argument('--device', type=str, choices=['L500', 'L400', 'all'], help='whether to use L515 camera')
    parser.add_argument('--tag', type=str, help='the save tag')
    args = parser.parse_args()

    # >>>>>> 3RDPARTY SYNC
    BASE_PATH = r"C:\Users\imtan\data"
    subpath = args.tag + '-' + str(uuid.uuid1()).split('-')[0] + '-' + datetime.now().strftime("%Y-%m-%d_%H%M%S")
    SUBNET = [10, 53, 24, 0]
    # <<<<<<

    context = rs.context()
    available_devices = enumerate_connected_devices(context)
    valid_devices = []
    for device in available_devices:
        if device[1] == 'L500' and args.device == 'L500':
            valid_devices.append(device)
        elif device[1] != 'L500' and args.device == 'L400':
            valid_devices.append(device)
        elif args.device == 'all':
            valid_devices.append(device)

    print('Find {} valid {} devices!'.format(len(valid_devices), args.device))
    # save_path = osp.join('C:\\Users\\liyutong\\Data\\realsense', subpath)
    save_path = osp.join(BASE_PATH, subpath)

    processes = []
    device_num = len(valid_devices)
    command_list = [multiprocessing.Value('d', -1) for _ in range(device_num)]
    queue_list = [multiprocessing.Queue() for _ in range(device_num)]
    device_num = len(valid_devices)
    parent2child_list = []
    child2parent_list = []
    for idx in range(device_num):
        parent2child_list.append(multiprocessing.Value('d', 0))
        child2parent_list.append(multiprocessing.Value('d', 0))

    for idx, device in enumerate(valid_devices):
        p = Process(target=camera_run, args=(device, save_path, command_list[idx], queue_list[idx],
                                             parent2child_list[idx], child2parent_list[idx],
                                             args.tag, idx == 0))
        p.start()
        processes.append(p)

    time.sleep(1)

    while True:
        active_process_num = 0
        for idx in range(device_num):
            if child2parent_list[idx].value == 1:
                active_process_num += 1
        if active_process_num == device_num:
            for idx in range(device_num):
                parent2child_list[idx].value += 1
            break

    while True:
        c = sys.stdin.readline()
        if 'c' in c:
            print('Capture one image!')
            for idx in range(device_num):
                command_list[idx].value = 0
        if 'r' in c:
            print('Start recording!')
            # Start all imu
            broadcast_command(SUBNET, 18888, "start\n", None)

            for idx in range(device_num):
                command_list[idx].value = 1
            time.sleep(1)
            for idx in range(device_num):
                queue_list[idx].put(None)
        elif 'q' in c:
            print('Quit now!')
            # stop all imu
            broadcast_command(SUBNET, 18888, "start\n", None)
            for idx in range(device_num):
                command_list[idx].value = 2
            break

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
