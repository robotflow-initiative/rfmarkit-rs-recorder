import os.path as osp

from realsense_recorder.apps.local_record_seq import main

class DefaultArgs:
    config: str = "./realsense_config.yaml"
    tag: str = None

main(DefaultArgs)