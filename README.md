# RealsenseRecorder

## Installation

### From Source

```shell
git clone https://github.com/mvig-robotflow/rfimu-realsense-recorder
cd rfimu-realsense-recorder
python setup.py
```

### From PyPi

```shell
python -m pip install realsense-recorder
```


## Usage

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
    depth: []
    endpoint: {}
    imu: []
    product_id: 0B64
    product_line: L500
    ref: 1
    sn: f0220485
    use_depth: false
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
