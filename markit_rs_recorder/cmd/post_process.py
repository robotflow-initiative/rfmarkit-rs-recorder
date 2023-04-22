import argparse
import numpy as np
from typing import Tuple

from markit_rs_recorder.io import compress_record, sync_cameras

# def detect_fiducials(base_dir: str):
#     pass


def worker(base_dir: str):
    try:
        compress_record(base_dir)
        sync_cameras(base_dir)
    except Exception as e:
        print(e)


def main(args):
    worker(args.base_dir)
    pass


def entry_point(argv):
    parser = argparse.ArgumentParser(description='Recorder')
    parser.add_argument('--base_dir', type=str, help='Base directory', required=True)
    args = parser.parse_args(argv)
    main(args)


if __name__ == '__main__':
    import sys

    entry_point(sys.argv[1:])
