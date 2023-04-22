import json
import os
from typing import Dict, List

import numpy as np


class RealsenseCameraSequence:
    SN: str = ""
    camera_intrinsic_path: str = ""
    camera_intrinsic: Dict = {}
    valid: bool = False
    width: int = 0
    height: int = 0
    intrinsic_matrix: np.ndarray = np.empty(0)

    def __init__(self, path_to_camera):
        self.path_to_camera = path_to_camera
        self.path_to_depth_stream = os.path.join(self.path_to_camera, "depth")
        self.path_to_color_stream = os.path.join(self.path_to_camera, "color")

        self.SN = os.path.basename(self.path_to_camera)
        self.camera_intrinsic_path = os.path.join(self.path_to_camera, "camera_intrinsic.json")
        if os.path.isfile(self.camera_intrinsic_path):
            try:
                self.camera_intrinsic = json.load(open(self.camera_intrinsic_path, "r"))
                self.width = self.camera_intrinsic['width']
                self.height = self.camera_intrinsic['height']
                self.intrinsic_matrix = self.camera_intrinsic['intrinsic_matrix']
                self.valid = True
            except json.JSONDecodeError as e:
                self.valid = False
            finally:
                self.valid = False

    def __repr__(self):
        return f"<RealsenseCamera, SN={self.SN}>"


class RealsenseCameraSystemSequence:
    def __init__(self, path_to_recording, mask: List[str] = None):
        if mask is None:
            mask = []
        self.path_to_recording = path_to_recording
        self.cameras: List[RealsenseCameraSequence] = []
        self.mask = mask  # ['SN1', 'SN2'], cameras to keep
        self.parse()

    def parse(self):
        camera_path_list = filter(lambda x: os.path.isdir(x), map(lambda x: os.path.join(self.path_to_recording, x), os.listdir(self.path_to_recording)))
        for camera_path in camera_path_list:
            self.cameras.append(RealsenseCameraSequence(path_to_camera=camera_path))

        if self.mask is not None:
            self.cameras = list(filter(lambda x: x.SN in self.mask, self.cameras))


if __name__ == '__main__':
    r = RealsenseCameraSystemSequence("/Users/liyutong/Downloads/bulu-0-0")
    print(r)
