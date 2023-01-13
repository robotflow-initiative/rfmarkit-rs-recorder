import argparse
import json
import os.path as osp
import time
from typing import List, Dict, Callable

import cv2
from rich.progress import track
import numpy as np

from realsense_recorder.common import (
    RealsenseCameraCfg,
    RealsenseSystemCfg,
    RealsenseSystemModel,
    new_realsense_camera_system_from_yaml_file,
    CALLBACKS,
    get_datetime_tag
)

class LocalCaptureStaticInteractive(RealsenseSystemModel):
    def __init__(self,
                 system_cfg: RealsenseSystemCfg,
                 camera_cfgs: List[RealsenseCameraCfg],
                 callbacks: Dict[str, Callable] = None):
        system_cfg.frame_queue_size = 0  # disable frame queue
        # lower the frame rate to 5/6 fps
        for camera_idx in range(len(camera_cfgs)):
            for color_idx in range(len(camera_cfgs[camera_idx].color)):
                camera_cfgs[camera_idx].color[color_idx].fps = camera_cfgs[camera_idx].__COLOR_FPS_CANDIDATES__[0]
                camera_cfgs[camera_idx].color[color_idx].exposure = -1
        system_cfg.interval_ms = 500

        super().__init__(system_cfg, camera_cfgs, callbacks)

    def app(self):
        _resolution = (640, 480)

        def create_windows():
            for cam in self.cameras:
                cv2.namedWindow(cam.window_name, cv2.WINDOW_AUTOSIZE)

        self.console.log(f"tag is {self.tag}")
        self.console.log(f"save path is {self.options.base_dir}")
        self.console.log(f"number of cameras is {len(self.cameras)}")
        self._set_advanced_mode()
        self.console.log("opening devices")
        self.open()
        self.console.log("creating windows")
        create_windows()
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
        self.console.log("start recording")
        for _ in track(range(50), description="waiting for streams to stabilize"):
            time.sleep(0.1)

        n_frames = 0
        try:
            while True:
                string_input = self.console.input(f"Press Enter to capture the {n_frames} frame or type 'q' to quit: ")
                if string_input == 'q':
                    break

                for idx, cam in enumerate(self.cameras):

                    self.console.print(f"capturing from camera {idx}")
                    color_image, depth_image, ts, sys_ts, frame_counter = cam.get_frames()

                    if color_image is not None:
                        cv2.imwrite(osp.join(cam.color_save_path, f'{n_frames}.jpg'), color_image)

                    if depth_image is not None:
                        np.save(osp.join(cam.depth_save_path, f'{n_frames}.npy'), depth_image)

                    if color_image is not None:
                        mix = cv2.resize(color_image, _resolution)
                    cv2.imshow(cam.window_name, mix)
                    cv2.waitKey(1)

                n_frames += 1

        except KeyboardInterrupt as e:
            # raise(e)
            self.console.log("\n" * len(self.cameras))
            self.console.log("stopped in response to KeyboardInterrupt")
            return

        except Exception as e:
            self.console.log("\n" * len(self.cameras))
            self.console.log(e)
            raise e

        finally:
            self.console.log("stopping cameras")
            self.stop(interval_ms=self.options.interval_ms)
            self.close()
            cv2.destroyAllWindows()


def main(args):

    callbacks = {
        CALLBACKS.tag_cb: (lambda: get_datetime_tag()) if args.tag is None else (lambda: args.tag),
        CALLBACKS.save_path_cb: lambda cam_cfg, sys_cfg: osp.join(sys_cfg.base_dir, "r" + cam_cfg.sn[-2:]),
        CALLBACKS.camera_friendly_name_cb: lambda cam_cfg, _: "r" + cam_cfg.sn[-2:]
    }

    sys = new_realsense_camera_system_from_yaml_file(LocalCaptureStaticInteractive, args.config, callbacks)

    sys.app()

def entry_point(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./realsense_config.yaml')
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args(argv)
    main(args)

if __name__ == '__main__':
    import sys
    entry_point(sys.argv)
