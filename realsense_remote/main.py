import json
import logging
import multiprocessing as mp
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

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

STOP_EV: mp.Event = mp.Event()
FINISH_EV: mp.Event = mp.Event()
CAPTURE_PROCS: List[mp.Process] = []
DEVICE_IDX = None
SAVE_PATH_BASE: str = None


# if using D435
# depth_scale = 0.001
SAVE_BAG = True
USE_DEPTH = True
SHOW_DEPTH = True
SHOW_FRAMES = False

# L500
L515_depth_resolution_width = 1024  # pixels, or 640
L515_depth_resolution_height = 768  # pixels, or 480
L515_color_resolution_width = 1280  # pixels, or 640
L515_color_resolution_height = 720  # pixels, or 480
L515_depth_frame_rate = 30
L515_color_frame_rate = 30

# D400
D435_depth_resolution_width = 1280  # pixels, or 848
D435_depth_resolution_height = 720  # pixels, or 848
color_resolution_width = 1280  # pixels, or 848
color_resolution_height = 720  # pixels, or 480
D435_frame_rate = 30  # fps

# Enable the streams from all the intel realsense devices
L515_rs_config = rs.config()
if USE_DEPTH:
    L515_rs_config.enable_stream(rs.stream.depth, L515_depth_resolution_width, L515_depth_resolution_height,
                                 rs.format.z16,
                                 L515_depth_frame_rate)
L515_rs_config.enable_stream(rs.stream.color, L515_color_resolution_width, L515_color_resolution_height, rs.format.bgr8,
                             L515_color_frame_rate)

D400_rs_config = rs.config()
if USE_DEPTH:
    D400_rs_config.enable_stream(rs.stream.depth, D435_depth_resolution_width, D435_depth_resolution_height,
                                 rs.format.z16, D435_frame_rate)
D400_rs_config.enable_stream(rs.stream.color, color_resolution_width, color_resolution_height, rs.format.bgr8,
                             D435_frame_rate)

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


def make_response(status, msg="", **kwargs):
    data = {'status': status, 'msg': msg, 'timestamp': time.time()}
    data.update(**kwargs)
    resp = Response(mimetype='application/json', status=200)
    resp.data = json.dumps(data)
    return resp


def capture_frames(device_idx,
                   stop_ev: mp.Event,
                   finish_ev: mp.Event,
                   save_path_tagged: str,
                   save_bag: bool):

    # Initialize parameters
    context = rs.context()
    available_devices = enumerate_connected_devices(context)
    if args.device == 'all':
        valid_devices = available_devices
    else:
        valid_devices = list(filter(lambda x: x[1] in args.device.split(','), available_devices))
    device = valid_devices[device_idx]
    
    device_serial = device[0]
    product_line = device[1]

    device_save_path = osp.join(save_path_tagged, device_serial)
    if not os.path.lexists(osp.join(device_save_path, 'color')):
        os.makedirs(osp.join(device_save_path, 'color'))
    if not os.path.lexists(osp.join(device_save_path, 'depth')):
        os.makedirs(osp.join(device_save_path, 'depth'))
    # Configure depth and color streams
    pipeline = rs.pipeline()


    if product_line == "L500":
        using_L515 = True
        # Enable L515 device
        L515_rs_config.enable_device(device_serial)
        if save_bag:
            L515_rs_config.enable_record_to_file(osp.join(device_save_path, f'recording_{time.time()}.bag'))
        # dev.first_depth_sensor().set_option(rs.option.inter_cam_sync_mode, 1 if is_master else 0)
        pipeline_profile = pipeline.start(L515_rs_config)
    else:
        # Enable D400 device
        using_L515 = False
        D400_rs_config.enable_device(device_serial)
        if save_bag:
            D400_rs_config.enable_record_to_file(osp.join(device_save_path, f'recording_{time.time()}.bag'))

        pipeline_profile = pipeline.start(D400_rs_config)

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
        with tqdm.tqdm(ncols=0) as pbar:
            while True:
                # Wait for a coherent pair of frames: depth and color
                if save_bag:
                    pipeline.wait_for_frames()  # we don't have to do anything!
                else:
                    frames = pipeline.wait_for_frames()
                    
                    aligned_frames = align.process(frames)
                    # depth_frame_org = frames.get_depth_frame()
                    if USE_DEPTH:
                        depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()

                    if not color_frame:
                        continue

                    color_image = np.asanyarray(color_frame.get_data())

                    if USE_DEPTH and SHOW_DEPTH:
                        depth_image = np.asanyarray(depth_frame.get_data())
                        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

                    # Show images
                    if SHOW_FRAMES:
                        cv2.namedWindow('RealSense_{}'.format(device[0]), cv2.WINDOW_AUTOSIZE)
                        # images = np.hstack((color_image, colorized_depth))
                        if USE_DEPTH and SHOW_DEPTH:
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

                if stop_ev.is_set():
                    logging.info(f'[realsense]: Stop recording, SN={device[0]}, frames={n_frames}')
                    pipeline.stop()
                    finish_ev.set()
                    break
    except Exception as e:
        # raise(e)
        print(f"[Recorder]:  SN={device.SN} got Exception {e}")
        pipeline.stop()
        finish_ev.set()
        return


@app.route("/", methods=['POST', 'GET'])
def index():
    return make_response(status=200, msg=f"Cameras={DEVICE_IDX}, ActiveProcesses={len(CAPTURE_PROCS)}")


@app.route("/start", methods=['POST', 'GET'])
def start_record():
    global CAPTURE_PROCS, STOP_EV, FINISH_EV, SAVE_PATH_BASE, DEVICE_IDX, SAVE_BAG

    # Wait until last capture ends
    if len(CAPTURE_PROCS) > 0:
        if STOP_EV.is_set():
            FINISH_EV.wait(timeout=0.5)
            if FINISH_EV.is_set():
                [proc.join(timeout=3) for proc in CAPTURE_PROCS]
                if any([proc.is_alive() for proc in CAPTURE_PROCS]):
                    logging.warning("[realsense] Join timeout")
                    [os.kill(proc.pid, signal.SIGTERM) for proc in CAPTURE_PROCS if proc.is_alive()]
                CAPTURE_PROCS = []
            else:
                return make_response(status=500, msg="NOT FINISHED")
        else:
            return make_response(status=500, msg="RUNNING")

    if len(CAPTURE_PROCS) == 0:
        STOP_EV.clear()
        FINISH_EV.clear()

        try:
            tag = request.get_json()["tag"]  # extract real arguments
            logging.info(f"[realsense] tag={tag}")
        except Exception:
            tag = str(int(time.time()))
        save_path_tagged: str = os.path.join(SAVE_PATH_BASE, tag)
        if not os.path.exists(save_path_tagged):
            os.makedirs(save_path_tagged)

        if len(CAPTURE_PROCS) <= 0:
            CAPTURE_PROCS = [mp.Process(target=capture_frames,
                                        args=(DEVICE_IDX,
                                              STOP_EV,
                                              FINISH_EV,
                                              save_path_tagged,
                                              SAVE_BAG))]
            DELAY_S = 2  # Magic delay duration to avoid U3V communication error
            [(proc.start(), time.sleep(DELAY_S
                                       )) for proc in CAPTURE_PROCS]

        return make_response(status=200, msg=f"START OK, subpath={tag}")


@app.route("/stop", methods=['POST', 'GET'])
def stop_record():
    global CAPTURE_PROCS, STOP_EV
    logging.info("[realsense] Stop")

    if len(CAPTURE_PROCS) and any([proc.is_alive() for proc in CAPTURE_PROCS]):
        STOP_EV.set()
        return make_response(status=200, msg=f"STOP OK: {len(CAPTURE_PROCS)} procs are running")
    else:
        return make_response(status=500, msg="NOT RUNNING")


@app.route("/kill", methods=['POST', 'GET'])
def kill_record():
    global CAPTURE_PROCS, STOP_EV, FINISH_EV
    logging.info("[realsense] kill")

    if len(CAPTURE_PROCS) and any([proc.is_alive() for proc in CAPTURE_PROCS])> 0:
        STOP_EV.set()
        FINISH_EV.wait(timeout=1)
        [proc.join(timeout=1) for proc in CAPTURE_PROCS]
        if any([proc.is_alive() for proc in CAPTURE_PROCS]):
            logging.warning("[realsense] Join timeout, force kill all processes")
            [os.kill(proc.pid, signal.SIGTERM) for proc in CAPTURE_PROCS if proc.is_alive()]
        return make_response(status=200, msg="KILL OK")
    else:
        return make_response(status=500, msg="NOT RUNNING")


@app.route("/quit", methods=['POST', 'GET'])
def quit():
    global CAPTURE_PROCS, STOP_EV, FINISH_EV
    logging.info("[realsense] quit")

    if len(CAPTURE_PROCS) and any([proc.is_alive() for proc in CAPTURE_PROCS])> 0:
        sys.exit(1)
    else:
        sys.exit(0)


def main(args=None):
    global SAVE_PATH_BASE, DEVICE_IDX

    # Register global parameters
    SAVE_PATH_BASE = args.base_dir
    DEVICE_IDX = args.idx

    # Prepare system
    logging.info('Using the {}-th {} device'.format(args.idx, args.device))
    logging.info('The server listens at port {}'.format(args.port))


    try:
        # app.run(host='0.0.0.0', port=args.port)
        server = pywsgi.WSGIServer(('0.0.0.0', args.port), app)
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"[realsense]:  Main got KeyboardInterrupt")
        exit(1)


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Recorder')
    parser.add_argument('--device', type=str, choices=['L500', 'D400', 'all'], help='The cameras to use, seperated by comma e.g. "L500,D400"', default='all')
    parser.add_argument('--idx', type=int, help='The index of the cam', default='0')
    parser.add_argument('--base_dir', type=str, help='The path to save frames', default='./realsense_data')
    parser.add_argument('--port', type=int, help="Port to listen", default=5050)
    args = parser.parse_args()
    main(args)
