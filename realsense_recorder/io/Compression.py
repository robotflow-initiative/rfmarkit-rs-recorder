import datetime
import glob
import os
import os.path as osp
import shutil
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import cv2
import numpy as np
import tqdm
from realsense_recorder.io import get_directory_reader
from rich.console import Console

console = None


def _read_color(path: str) -> Any:
    """
    Read anything, from a path
    :param path:
    :return: numpy.ndarray, RGB frame
    """
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _compress_color_folder(input_folder, n_prefetch):
    reader = get_directory_reader(input_folder, 'color_bmp', num_preload=n_prefetch, read_function=_read_color)
    files_to_remove = []
    with tqdm.tqdm(range(len(reader))) as pbar:
        while not reader.eof:
            frame, meta, _ = reader.next()
            frame_basename_without_ext = osp.splitext(osp.basename(meta['basename']))[0]
            cv2.imwrite(osp.join(input_folder, frame_basename_without_ext + ".jpeg"), frame)
            files_to_remove.append(osp.join(input_folder, frame_basename_without_ext + ".bmp"))
            pbar.update()
    for f in files_to_remove:
        os.remove(f)


def _compress_depth_folder(input_folder, n_prefetch):
    reader = get_directory_reader(input_folder, 'depth_npy', num_preload=n_prefetch)
    files_to_remove = []
    with tqdm.tqdm(range(len(reader))) as pbar:
        while not reader.eof:
            frame, meta, _ = reader.next()
            frame_basename_without_ext = osp.splitext(osp.basename(meta['basename']))[0]
            np.savez_compressed(osp.join(input_folder, frame_basename_without_ext + ".npz"), depth=frame)
            files_to_remove.append(osp.join(input_folder, frame_basename_without_ext + ".npy"))
            pbar.update()
    for f in files_to_remove:
        os.remove(f)


def compress_record(input_recording: str, n_prefetch=16, console=None):
    console = Console() if console is None else console
    console.log(f"input recording: {input_recording}")

    camera_folders = list(
        map(lambda x: os.path.basename(x),
            list(filter(lambda x: os.path.isdir(x),
                        glob.glob(osp.join(input_recording, "*"))
                        )
                 )
            )
    )

    folder_compression_color = [
        osp.join(input_recording, camera_folder, 'color') for camera_folder in camera_folders
    ]

    folder_compression_depth = [
        osp.join(input_recording, camera_folder, 'depth') for camera_folder in camera_folders
    ]

    pool_1 = ProcessPoolExecutor(max_workers=4)
    for _input_folder in folder_compression_color:
        console.log(f"compressing color {_input_folder}")
        pool_1.submit(_compress_color_folder, _input_folder, n_prefetch)

    pool_2 = ProcessPoolExecutor(max_workers=4)
    for _input_folder in folder_compression_depth:
        console.log(f"Compressing depth {_input_folder}")
        pool_2.submit(_compress_depth_folder, _input_folder, n_prefetch)

    pool_1.shutdown(wait=True)
    pool_2.shutdown(wait=True)



if __name__ == '__main__':
    # debug
    # main()
    console = Console()
    compress_record(r"C:\Users\robotflow\Desktop\virat\realsense_data\2023-04-16_213145-2", console=console)
