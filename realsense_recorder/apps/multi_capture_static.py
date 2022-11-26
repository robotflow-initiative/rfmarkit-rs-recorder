import argparse
import json
import os.path as osp
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Callable

import cv2
import numpy as np
import tqdm
from realsense_recorder.common import (
    RealsenseCameraCfg,
    RealsenseSystemCfg,
    RealsenseSystemModel,
    new_realsense_camera_system_from_yaml_file,
    CALLBACKS,
    get_datetime_tag
)


class MultiCaptureStatic(RealsenseSystemModel):
    def __init__(self,
                 system_cfg: RealsenseSystemCfg,
                 camera_cfgs: List[RealsenseCameraCfg],
                 callbacks: Dict[str, Callable] = None):
        super().__init__(system_cfg, camera_cfgs, callbacks)

    def app(self):

        def create_windows():
            for cam in self.cameras:
                cv2.namedWindow(cam.window_name, cv2.WINDOW_AUTOSIZE)

        self.console.log(f"tag is {self.tag}")
        self.console.log(f"save path is {self.options.base_dir}")
        self.console.log(f"number of cameras is {len(self.cameras)}")
        self._set_advanced_mode()
        self.console.log("opening devices")
        self.open()
        if self.options.interactive:
            console.log("creating windows")
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
        progress_bars = [tqdm.tqdm(ncols=0) for _ in range(len(self.cameras))]
        save_workers = ThreadPoolExecutor(max_workers=len(self.cameras) * 2)

        try:
            while True:
                for idx, cam in enumerate(self.cameras):
                    tic = time.time()
                    color_image, depth_image, ts, sys_ts, frame_counter = cam.get_frames()
                    self.insert_meta_data(cam.friendly_name, ts, sys_ts, frame_counter)
                    toc = time.time()

                    if color_image is not None:
                        save_workers.submit(lambda: cv2.imwrite(osp.join(cam.color_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.bmp'), color_image))
                        # i5 12400K save jpg at 27 fps (load 80%), write to PM9A1 at 200 - 500MB/s, memory consumption 1GB/min percamera
                        # i5 12400K save bmp at 30 fps, but write at 1.0GB/s, memory consumption 0GB/min per camera

                        # save_workers.submit(cv2.imwrite, (osp.join(cam.color_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.bmp'), color_image,))
                        # cv2.imwrite(osp.join(cam.color_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.bmp'), color_image)

                    if depth_image is not None:
                        save_workers.submit(lambda: np.save(osp.join(cam.depth_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.npy'), depth_image))
                        # np.save(osp.join(cam.depth_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.npy'), depth_image)
                        # cv2.imwrite(osp.join(cam.depth_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.png'), depth_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                    progress_bars[idx].set_description(f'SN={cam.option.sn}, ProcessTime={int((toc - tic) * 1000)}(ms), FrameCounter={frame_counter}')
                    progress_bars[idx].update(1)

                    if self.options.interactive and not self.options.use_bag:
                        if color_image is not None and depth_image is not None:
                            mix = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)
                        elif color_image is not None and depth_image is None:
                            mix = color_image
                        elif color_image is None and depth_image is not None:
                            mix = depth_image
                        else:
                            break
                        cv2.imshow(cam.window_name, mix)
                        cv2.waitKey(1)

        except KeyboardInterrupt as e:
            # raise(e)
            self.console.log("\n" * len(self.cameras))
            self.console.log("stopped in response to KeyboardInterrupt")

            self.console.log("stopping cameras")
            self.stop(interval_ms=self.options.interval_ms)
            for idx, cam in enumerate(self.cameras):
                self.console.log(f"saving last frames of camera {cam.option.sn}")
                while True:
                    color_image, depth_image, ts, sys_ts, frame_counter = cam.get_frames()
                    self.insert_meta_data(cam.friendly_name, ts, sys_ts, frame_counter)
                    if frame_counter > 0:
                        if color_image is not None:
                            save_workers.submit(lambda: cv2.imwrite(osp.join(cam.color_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.bmp'), color_image))

                        if depth_image is not None:
                            save_workers.submit(lambda: np.save(osp.join(cam.depth_save_path, f'{cam.n_frames}_{ts}_{sys_ts}.npy'), depth_image))

                        progress_bars[idx].set_description(f'SN={cam.option.sn}, FrameCounter={frame_counter}')
                        progress_bars[idx].update(1)
                    else:
                        break
            self.stop()
            self.close()
            save_workers.shutdown(wait=True)
            self.save_meta_data()
            cv2.destroyAllWindows()
            return

        except Exception as e:
            self.console.log(e)
            raise e


def main(args):
    callbacks = {
        CALLBACKS.tag_cb: lambda: get_datetime_tag() if args.tag is None else lambda: args.tag,
        CALLBACKS.save_path_cb: lambda cam_cfg, sys_cfg: osp.join(sys_cfg.base_dir, "r" + cam_cfg.sn[-2:]),
        CALLBACKS.camera_friendly_name: lambda cam_cfg, _: "r" + cam_cfg.sn[-2:]
    }

    sys = new_realsense_camera_system_from_yaml_file(LocalRecordSequence, args.config, callbacks)

    sys.app()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./realsense_config.yaml')
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()
    main(args)
