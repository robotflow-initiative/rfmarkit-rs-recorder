# RealsenseRecorder

A set of Python scripts to operate Intel Realsense Cameras. By using multi-thread techniques, this script can record color + depth from up to 4 Realsense Cameras (2 x L515 and 2 x D435)

## Installation

### From Source

```shell
git clone https://github.com/mvig-robotflow/rfimu-realsense-recorder
cd rfimu-realsense-recorder
python setup.py
```

### From PyPi

```shell
python -m pip install markit-realsense-recorder
```

## Usage

### Script Usage

```python
import io

import cv2
import numpy as np
import yaml
from realsense_recorder.common import new_realsense_camera_system_from_config, RealsenseSystemModel

cfg_str = """
realsense:
  cameras:
  - color:
    - exposure: -1
      format: rs.format.bgra8
      fps: 30
      height: 1080
      width: 1920
    depth: [] # Do not user depth 
    endpoint: {}
    imu: []
    product_id: 0B64 # 0B64 for L515
    product_line: L500 # Currently supported models are L500(L515) and D400(D435)
    ref: 1
    sn: f0220485 # SN of target Camera, can get from RealSenseViewer
  system:
    base_dir: ./realsense_data
    frame_queue_size: 100
    interactive: false
    interval_ms: 0
    use_bag: false
"""


def main():
    cfg = yaml.load(io.StringIO(cfg_str), yaml.SafeLoader)
    sys = new_realsense_camera_system_from_config(RealsenseSystemModel, cfg['realsense'], None)
    print(sys.cameras)
    cam = sys.cameras[0]
    cam.open()
    cam.start()
    mtx = np.array(cam.intrinsics_matrix)
    while True:
        color_image, depth_image, ts, sys_ts, frame_counter = cam.get_frames()
        cv2.imshow("frame", color_image)
        key = cv2.waitKey(1)
        if key == 27:
            print('esc break...')
            cv2.destroyAllWindows()
            break

main()
```

### Command Line Usage

To Create and persist record configuration:

```shell
python -m realsense_recorder configure
```

To launch a remote record station that supports REST API

```shell
python -m realsense_recorder run --app=remote_record_seq
```
