import argparse
import json
import logging
import multiprocessing as mp
import os
import os.path as osp
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import tqdm
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, RedirectResponse

from realsense_recorder.common import (
    CALLBACKS,
    new_realsense_camera_system_from_yaml_file,
    RealsenseSystemModel,
    RealsenseSystemCfg,
    RealsenseCameraCfg,
    get_datetime_tag
)
app = FastAPI()
logging.basicConfig(level=logging.INFO)

STOP_EV: mp.Event = mp.Event()
FINISH_EV: mp.Event = mp.Event()
READY_EV: mp.Event = mp.Event()
CAPTURE_PROCS: List[mp.Process] = []
ARGS: Optional[argparse.Namespace] = None


class RemoteRecordSeq(RealsenseSystemModel):
    def __init__(self,
                 system_cfg: RealsenseSystemCfg,
                 camera_cfgs: List[RealsenseCameraCfg],
                 callbacks: Dict[str, Callable] = None):
        super().__init__(system_cfg, camera_cfgs, callbacks)
        self.metadata: Dict[Dict] = {cam.friendly_name: [] for cam in self.cameras}

    def insert_meta_data(self, idx, ts, sys_ts, frame_counter):
        self.metadata[idx].append({
            "ts": ts,
            "sys_ts": sys_ts,
            "frame_counter": frame_counter
        })

    def save_meta_data(self):
        save_path = osp.join(self.options.base_dir, "metadata_all.json")
        config_save_path = osp.join(self.options.base_dir, "realsense_config.json")
        bundle = {
            "camera_sn": [cam.option.sn for cam in self.cameras],
            "metadata": self.metadata
        }
        with open(save_path, 'w') as f:
            json.dump(bundle, f, indent=4)

        with open(config_save_path, 'w') as f:
            json.dump({"realsense": {"system": self.options.get_dict(), "cameras": list(map(lambda x: x.get_dict(), self.camera_options))}}, f, indent=4)

    def app(self, stop_ev: mp.Event, finish_ev: mp.Event, ready_ev: mp.Event = None):

        def save_color_frame(path, frame):
            cv2.imwrite(path, frame)

        def save_depth_frame(path, frame):
            np.save(path, frame)

        self.console.log(f"tag is {self.tag}")
        self.console.log(f"save path is {self.options.base_dir}")
        self.console.log(f"number of cameras is {len(self.cameras)}")
        self._set_advanced_mode()
        self.console.log("opening devices")
        self.open()
        self.console.log("starting devices")
        self.start(interval_ms=self.options.interval_ms)

        self.console.log("saving intrinsics")
        for cam in self.cameras:
            '''Convert intrinsic data to dict (readable in Open3D)'''
            mat = [cam.intrinsics.fx, 0, 0, 0, cam.intrinsics.fy, 0, cam.intrinsics.ppx, cam.intrinsics.ppy, 1]
            intrinsics_dict = {'width': cam.intrinsics.width, 'height': cam.intrinsics.height, 'intrinsic_matrix': mat}
            save_path = osp.join(cam.save_path, 'realsense_intrinsic.json')
            if not osp.lexists(save_path):
                with open(save_path, 'w') as f:
                    json.dump(intrinsics_dict, f, sort_keys=False,
                              indent=4,
                              ensure_ascii=False)

        self.console.log("[]start recording")
        progress_bars = [tqdm.tqdm(ncols=0) for _ in range(len(self.cameras))]
        save_workers = ThreadPoolExecutor(max_workers=len(self.cameras) * 2)

        if ready_ev is not None:
            ready_ev.set()

        try:
            while True:
                for idx, cam in enumerate(self.cameras):
                    tic = time.time()
                    color_image, depth_image, ts, sys_ts, frame_counter = cam.get_frames()
                    self.insert_meta_data(cam.friendly_name, ts, sys_ts, frame_counter)
                    toc = time.time()

                    if color_image is not None:
                        save_workers.submit(save_color_frame, osp.join(cam.color_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.bmp'), color_image)
                        # i5 12400K save jpg at 17 fps (load 60%), write to PM9A1 at 200 - 500MB/s, memory consumption 1GB/min percamera
                        # i5 12400K save bmp at 20 fps, but write at 1.0GB/s, memory consumption 0GB/min per camera

                        # save_workers.submit(cv2.imwrite, (osp.join(cam.color_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.bmp'), color_image,))
                        # cv2.imwrite(osp.join(cam.color_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.bmp'), color_image)

                    if depth_image is not None:
                        save_workers.submit(save_depth_frame, osp.join(cam.depth_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.npy'), depth_image)
                        # save_workers.submit(lambda: cv2.imwrite(osp.join(cam.depth_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.png'), depth_image, [cv2.IMWRITE_PNG_COMPRESSION, 0]))

                        # np.save(osp.join(cam.depth_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.npy'), depth_image)
                        # cv2.imwrite(osp.join(cam.depth_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.png'), depth_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                    progress_bars[idx].set_description(f'SN={cam.option.sn}, ProcessTime={int((toc - tic) * 1000)}(ms), FrameCounter={frame_counter}')
                    progress_bars[idx].update(1)

                    if self.options.interactive and not self.options.use_bag:
                        if color_image is not None and depth_image is not None:
                            mix = cv2.addWeighted(color_image[..., 0:3], 0.5, cv2.resize(depth_image, (color_image.shape[1], color_image.shape[0])), 0.5, 0)
                        elif color_image is not None and depth_image is None:
                            mix = color_image
                        elif color_image is None and depth_image is not None:
                            mix = depth_image
                        else:
                            break
                        cv2.imshow(cam.window_name, mix)
                        cv2.waitKey(1)
                if stop_ev.is_set():
                    self.console.log('stop recording')
                    finish_ev.set()
                    raise KeyboardInterrupt
        except KeyboardInterrupt as e:
            # raise(e)
            self.console.log("\n" * len(self.cameras))
            self.console.log("stopped in response to KeyboardInterrupt")

            self.console.log("stopping cameras")
            self.stop(interval_ms=self.options.interval_ms)
            for idx, cam in enumerate(self.cameras):
                self.console.log(f"saving last frames of camera {cam.option.sn}")
                while True:
                    color_image, depth_image, ts, sys_ts, frame_counter = cam.get_frames(timeout_ms=50)
                    self.insert_meta_data(cam.friendly_name, ts, sys_ts, frame_counter)
                    if frame_counter > 0:
                        if color_image is not None:
                            save_workers.submit(save_color_frame, osp.join(cam.color_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.bmp'), color_image)

                        if depth_image is not None:
                            save_workers.submit(save_depth_frame, osp.join(cam.depth_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.npy'), depth_image)

                        progress_bars[idx].set_description(f'SN={cam.option.sn}, FrameCounter={frame_counter}')
                        progress_bars[idx].update(1)
                    else:
                        break
            self.close()
            save_workers.shutdown(wait=True)
            self.save_meta_data()
            cv2.destroyAllWindows()
            return

        except Exception as e:
            self.console.log(e)
            raise e
        finally:
            finish_ev.set()
            ready_ev.clear()
            self.console.log(f"finished recording, tag={self.tag}")


def make_response(status_code, **kwargs):
    data = {'code': status_code, 'timestamp': time.time()}
    data.update(**kwargs)
    json_compatible_data = jsonable_encoder(data)
    resp = JSONResponse(content=json_compatible_data, status_code=status_code)
    return resp


def capture_frames(stop_ev: mp.Event,
                   finish_ev: mp.Event,
                   ready_ev: mp.Event,
                   config: str,
                   tag: str):
    callbacks = {
        CALLBACKS.tag_cb: lambda: tag,
        CALLBACKS.save_path_cb: lambda cam_cfg, sys_cfg: osp.join(sys_cfg.base_dir, "r" + cam_cfg.sn[-2:]),
        CALLBACKS.camera_friendly_name_cb: lambda cam_cfg, _: "r" + cam_cfg.sn[-2:]
    }

    sys = new_realsense_camera_system_from_yaml_file(RemoteRecordSeq, config, callbacks)

    sys.app(stop_ev, finish_ev, ready_ev)


@app.get("/")
def root():
    return RedirectResponse(url='/docs')


@app.get("/v1/status")
def status():
    return make_response(status_code=200, active_processes=[proc.is_alive() for proc in CAPTURE_PROCS].count(True))


@app.get("/v1/ready")
def ready():
    global READY_EV
    if READY_EV is not None:
        if READY_EV.is_set():
            return make_response(status_code=200, ready=True)
        else:
            return make_response(status_code=200, ready=False)
    else:
        return make_response(status_code=500, msg="NOT SUPPORTED")


@app.post("/v1/start")
def start_process(tag: str = None):
    global CAPTURE_PROCS, STOP_EV, FINISH_EV, READY_EV, ARGS

    # Wait until last capture ends
    if len(CAPTURE_PROCS) > 0:
        if STOP_EV.is_set():
            FINISH_EV.wait(timeout=5)
            if FINISH_EV.is_set():
                [proc.join(timeout=3) for proc in CAPTURE_PROCS]
                if any([proc.is_alive() for proc in CAPTURE_PROCS]):
                    logging.warning("[realsense] join timeout")
                    [os.kill(proc.pid, signal.SIGTERM) for proc in CAPTURE_PROCS if proc.is_alive()]
                # clean up resources
                CAPTURE_PROCS = []
                READY_EV.clear()
            else:
                return make_response(status_code=500, msg="NOT FINISHED")
        else:
            return make_response(status_code=500, msg="RUNNING")

    if len(CAPTURE_PROCS) == 0:
        STOP_EV.clear()
        FINISH_EV.clear()

        if tag is None:
            tag = get_datetime_tag()
        logging.info(f"[realsense] tag={tag}")

        if len(CAPTURE_PROCS) <= 0:
            CAPTURE_PROCS = [mp.Process(target=capture_frames,
                                        args=(STOP_EV,
                                              FINISH_EV,
                                              READY_EV,
                                              ARGS.config,
                                              tag))]
            DELAY_S = 2  # Magic delay duration to avoid U3V communication error
            [(proc.start(), time.sleep(DELAY_S
                                       )) for proc in CAPTURE_PROCS]

        return make_response(status_code=200, msg="START OK", subpath=tag)


@app.post("/v1/stop")
def stop_process():
    global CAPTURE_PROCS, STOP_EV
    logging.info("[realsense] Stop")

    if len(CAPTURE_PROCS) > 0 and any([proc.is_alive() for proc in CAPTURE_PROCS]):
        STOP_EV.set()
        return make_response(status_code=200, msg=f"STOP OK: {len([None for proc in CAPTURE_PROCS if proc.is_alive()])} procs are running")
    else:
        return make_response(status_code=500, msg="NOT RUNNING")


@app.post("/v1/kill")
def kill_process():
    global CAPTURE_PROCS, STOP_EV, FINISH_EV
    logging.info("[realsense] kill")

    if len(CAPTURE_PROCS) and any([proc.is_alive() for proc in CAPTURE_PROCS]) > 0:
        STOP_EV.set()
        FINISH_EV.wait(timeout=1)
        [proc.join(timeout=1) for proc in CAPTURE_PROCS]
        if any([proc.is_alive() for proc in CAPTURE_PROCS]):
            logging.warning("[realsense] join timeout, force kill all processes")
            [os.kill(proc.pid, signal.SIGTERM) for proc in CAPTURE_PROCS if proc.is_alive()]
        return make_response(status_code=200, msg="KILL OK")
    else:
        return make_response(status_code=500, msg="NOT RUNNING")


def main(args: argparse.Namespace):
    global ARGS
    ARGS = args
    # Prepare system
    logging.info('the server listens at port {}'.format(args.port))

    try:
        uvicorn.run(app=app, port=args.port)
    except KeyboardInterrupt:
        logging.info(f"main() got KeyboardInterrupt")
        exit(1)

def entry_point(argv):
    parser = argparse.ArgumentParser(description='Recorder')
    parser.add_argument('--config', type=str, help='The realsense system configuration', default='./realsense_config.yaml')
    parser.add_argument('--port', type=int, help="Port to listen", default=5050)
    parser.add_argument('--debug', action='store_true', help='Toggle Debug mode')
    args = parser.parse_args(argv)
    main(args)

if __name__ == '__main__':
    import sys
    entry_point(sys.argv)