import pyrealsense2 as rs
import tqdm

config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

config.enable_record_to_file('./playground.bag')

pipeline = rs.pipeline()
profile = pipeline.start(config)

try:
    with tqdm.tqdm(ncols=0) as pbar:
        while True:
            pipeline.wait_for_frames()
            pbar.update(1)
finally:
    pipeline.stop()
