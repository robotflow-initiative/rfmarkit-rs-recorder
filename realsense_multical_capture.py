import sys
import os
import cv2
import pyrealsense2 as rs
import logging
from realsense_frame_queue_multiproc import enumerate_connected_devices
import numpy as np
import time

CAM_CONFIG = {
    'D400': {
        'color_width': 1280,
        'color_height': 720,
        'depth_width': 1280,
        'depth_height': 720,
    },
    'L500': {
        'color_width': 1280,
        'color_height': 720,
        'depth_width': 1024,
        'depth_height': 768,
    }
}


if __name__ == '__main__':
    SAVE_PATH_BASE = 'C:/Users/imtan/data/multical/220527_01/{}{}/'
    CAPTURE_PATTERN = '{}.jpg'

    logging.basicConfig(level=logging.INFO)

    # Realsense
    context = rs.context()
    valid_devices = enumerate_connected_devices(context)
    print('Find {} valid devices!'.format(len(valid_devices)))

    realsense_cameras = []
    for idx, device in enumerate(valid_devices):
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        device_serial = device[0]
        device_product_line = device[1]
        print('[{}] {}:{}'.format(idx, device_serial, device_product_line))
        config.enable_stream(rs.stream.depth, CAM_CONFIG[device_product_line]['depth_width'], CAM_CONFIG[device_product_line]['depth_height'], rs.format.z16, 30)
        config.enable_stream(rs.stream.color, CAM_CONFIG[device_product_line]['color_width'], CAM_CONFIG[device_product_line]['color_height'], rs.format.bgr8, 30)
        # Start streaming
        if device_product_line == 'L500':
            config.enable_device(device_serial)
        init_profile = pipeline.start(config)
        # Set preset for L515
        if device_product_line == 'L500':
            depth_sensor = init_profile.get_device().first_depth_sensor()
            preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
            for i in range(int(preset_range.max) + 1):
                visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset, i)
                if visulpreset == "Short Range":
                    depth_sensor.set_option(rs.option.visual_preset, i)
                    print('Changed to Short Range')
        # change exposure
        sensor = init_profile.get_device().query_sensors()[1]
        sensor.set_option(rs.option.enable_auto_exposure, 1)
        intrinsics = init_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intrinsics_matrix = np.array([
            [intrinsics.fx, 0., intrinsics.ppx],
            [0., intrinsics.fy, intrinsics.ppy],
            [0., 0., 1.]
        ])
        print(f'intrinsics matrix: {intrinsics_matrix}')

        align = rs.align(rs.stream.color)  # align to color is better than to depth

        device_save_prefix = device_product_line  # device[0][-2:]
        realsense_cameras.append((device_save_prefix, (pipeline, align)))

        device_save_path = SAVE_PATH_BASE.format('', device_save_prefix)
        if not os.path.lexists(device_save_path):
            os.makedirs(device_save_path)
        device_save_path = SAVE_PATH_BASE.format('', device_save_prefix + 'depth')
        if not os.path.lexists(device_save_path):
            os.makedirs(device_save_path)

    capture_count = 0
    try:
        while True:
            command = input('Press Enter to capture, q to exit.')
            if command == '' or command == 'c':

                # Realsense
                for cname, (pipe, align) in realsense_cameras:
                    # Wait for a coherent pair of frames: depth and color
                    frames = pipe.wait_for_frames()  # @fixme: will get multiple repeated depth images for r22 camera
                    # align
                    frames = align.process(frames)
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    fname = os.path.join(SAVE_PATH_BASE.format('', cname), CAPTURE_PATTERN.format(capture_count))
                    cv2.imwrite(fname, color_image)
                    fname = os.path.join(SAVE_PATH_BASE.format('', cname + 'depth'), CAPTURE_PATTERN.format(capture_count))
                    fname = str(fname).replace('.jpg', '.png')
                    cv2.imwrite(fname, (depth_image / 4).astype(np.uint16))
                    print('-------', cname + 'OK', 'time', time.time())

                capture_count += 1

            elif command == 'q':
                raise RuntimeError('Quit!')

            else:
                print(command, 'not found')
    except Exception as e:
        print(e)

    finally:
        for cname, (pipe, align) in realsense_cameras:
            pipe.stop()
            print(cname, 'closed')
        quit()
