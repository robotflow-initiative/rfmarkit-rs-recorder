import glob
import os
import threading
from typing import Callable, List, Tuple, Dict, Any

import cv2
import numpy as np


def index_function_0(item: str) -> int:
    """
    Convert a string label to int index
    :param item: label as string
    :return: converted index
    """
    return int(os.path.basename(item).split('_')[0])


def read_function_0(path: str) -> Any:
    """
    Read anything, from a path
    :param path:
    :return: numpy.ndarray, RGB frame
    """
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def read_function_1(path: str) -> Any:
    """
    Read anything, from a path
    :param path:
    :return: loaded numpy.ndarray
    """
    return np.load(path)


def read_function_2(path: str) -> Any:
    """
    Read anything, from a path
    :param path:
    :return: numpy.ndarray, depth frame
    """
    return cv2.imread(path, cv2.CV_16UC1)


def parse_function_0(label: str):
    """
    Parse meta data from label as string
    :param label:
    :return:
    """
    basename = os.path.basename(label)
    meta = os.path.splitext(basename)[0].split('_')
    return {
        "frame_idx": meta[0],
        "timestamp_int": meta[1],
        "timestamp": meta[2],
        "basename": basename
    }


class DirectoryReader:
    def __init__(self, path_to_dir: str,
                 sorting_function: Callable,
                 read_function: Callable,
                 parse_function: Callable,
                 glob_pattern: str,
                 num_preload: int = 0):
        """
        DirectoryReader read small files from a directory using async I/O

        :param path_to_dir: the path to directory
        :param sorting_function: function to sort files
        :param read_function: function to read data
        :param parse_function: function to get metadata
        :param glob_pattern: glob pattern to apply, use to find target files
        :param num_preload: number of preloaded files, set to 0 to disable multi-thread
        """
        self.path_to_folder = path_to_dir
        self.sort_function = sorting_function
        self.read_function = read_function
        self.parse_function = parse_function
        self.glob_pattern = glob_pattern
        self.num_preload = max(0, num_preload)

        self.frame_path_list = []
        self.metadata = []
        self.curr_idx: int = 0
        self.load()
        self.frame_buffer: Dict[int, Tuple[np.ndarray, Dict]] = {}
        self.frame_buffer_lock: threading.Lock = threading.Lock()

    def __len__(self):
        return len(self.frame_path_list)

    def load(self):
        frame_path_list: List[str] = glob.glob(os.path.join(self.path_to_folder, self.glob_pattern))
        self.frame_path_list = sorted(frame_path_list, key=index_function_0)
        self.metadata = list(map(lambda x: self.parse_function(x), self.frame_path_list))
        self.curr_idx = 0

    def reload(self, new_frame_path_list=None, sort=False):
        if new_frame_path_list is not None:
            self.frame_path_list = new_frame_path_list
            if sort:
                self.frame_path_list = sorted(self.frame_path_list, key=index_function_0)

        self.metadata = list(map(lambda x: self.parse_function(x), self.frame_path_list))
        self.curr_idx = 0

    def _is_in_buffer(self, idx: int) -> bool:
        """
        Check if an index is in buffer
        :param idx: the index of frame to judge
        :return:
        """
        return idx in self.frame_buffer.keys()

    def _read_frame_to_buffer(self, idx: int) -> None:
        """
        Read the file correspond to index to buffer
        :param idx: the index of frame to read
        :return:
        """
        if 0 <= idx < len(self.frame_path_list):
            img = self.read_function(self.frame_path_list[idx])
            meta = self.metadata[idx]
        else:
            img, meta = np.empty(0), {}

        with self.frame_buffer_lock:
            self.frame_buffer[idx] = (img, meta, idx)

    def _delete_frame_from_buffer(self, idx: int) -> None:
        """
        Delete an index from buffer
        :param idx: the index of frame to delete
        :return:
        """
        if idx in self.frame_buffer.keys():
            del self.frame_buffer[idx]

    def _cache_frame(self, idx: int) -> None:
        """
        Cache a small file (frame)
        :param idx: the index of frame to cache
        :return:
        """
        with self.frame_buffer_lock:
            if idx in self.frame_buffer.keys():
                return
        t = threading.Thread(target=self._read_frame_to_buffer, args=(idx,))
        t.start()

    def _read_frame(self, idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Read a small file (frame), and return frame with metadata
        :param idx:
        :return:
        """
        if 0 <= idx < len(self.frame_path_list):
            img = self.read_function(self.frame_path_list[idx])
            meta = self.metadata[idx]
            return img, meta, idx
        else:
            return np.empty(0), {}, idx

    def next(self) -> Tuple[np.ndarray, Dict]:
        """
        Read the next small file (frame), cache future frames if num_preload > 0
        :return:
        """
        for n_preload in range(self.num_preload):
            self._cache_frame(self.curr_idx + n_preload)

        img, meta = None, None
        with self.frame_buffer_lock:
            if self._is_in_buffer(self.curr_idx):
                img, meta, idx = self.frame_buffer[self.curr_idx]
                self._delete_frame_from_buffer(self.curr_idx)
        if img is None:
            img, meta, idx = self._read_frame(self.curr_idx)

        self.curr_idx = min(self.__len__(), self.curr_idx + 1)
        return img, meta, idx

    def prev(self) -> Tuple[np.ndarray, Dict]:
        """
        Read the previous small file (frame), cache past frames if num_preload > 0
        :return:
        """
        self.curr_idx = max(-1, self.curr_idx - 1)

        for n_preload in range(self.num_preload):
            self._cache_frame(self.curr_idx - n_preload)

        img, meta, idx = None, None, -1
        with self.frame_buffer_lock:
            if self._is_in_buffer(self.curr_idx):
                img, meta, idx = self.frame_buffer[self.curr_idx]
                self._delete_frame_from_buffer(self.curr_idx)

        if img is None:
            img, meta, idx = self._read_frame(self.curr_idx)

        return img, meta, idx

    @property
    def eof(self):
        """
        If the curr_index exceed the end of file or the start of file, return True
        :return:
        """
        if self.curr_idx >= len(self.frame_path_list) or self.curr_idx < 0:
            return True
        else:
            return False

    def seek(self, idx: int):
        """
        Move curr_idx to idx
        :param idx:
        :return:
        """
        self.curr_idx = max(min(idx, len(self.frame_path_list) - 1), 0)


def get_directory_reader(path_to_folder: str,
                         type: str,
                         num_preload: int = 0,
                         read_function: Callable = None,
                         parse_function: Callable = None) -> DirectoryReader:
    """
    Get directory reader by label
    :param path_to_folder:
    :param type:
    :param num_preload:
    :param read_function:
    :return:
    """
    if type == "color_bmp":
        return DirectoryReader(path_to_folder,
                               sorting_function=index_function_0,
                               read_function=read_function_0 if read_function is None else read_function,
                               parse_function=parse_function_0 if parse_function is None else parse_function,
                               glob_pattern="*.bmp",
                               num_preload=num_preload)
    elif type == "color_png":
        return DirectoryReader(path_to_folder,
                               sorting_function=index_function_0,
                               read_function=read_function_0 if read_function is None else read_function,
                               parse_function=parse_function_0 if parse_function is None else parse_function,
                               glob_pattern="*.png",
                               num_preload=num_preload)
    elif type == "color_jpg":
        return DirectoryReader(path_to_folder,
                               sorting_function=index_function_0,
                               read_function=read_function_0 if read_function is None else read_function,
                               parse_function=parse_function_0 if parse_function is None else parse_function,
                               glob_pattern="*.jpg",
                               num_preload=num_preload)
    elif type == "color_jpeg":
        return DirectoryReader(path_to_folder,
                               sorting_function=index_function_0,
                               read_function=read_function_0 if read_function is None else read_function,
                               parse_function=parse_function_0 if parse_function is None else parse_function,
                               glob_pattern="*.jpeg",
                               num_preload=num_preload)

    elif type == "depth_npy":
        return DirectoryReader(path_to_folder,
                               sorting_function=index_function_0,
                               read_function=read_function_1 if read_function is None else read_function,
                               parse_function=parse_function_0 if parse_function is None else parse_function,
                               glob_pattern="*.npy",
                               num_preload=num_preload)
    elif type == "depth_npz":
        return DirectoryReader(path_to_folder,
                               sorting_function=index_function_0,
                               read_function=read_function_1 if read_function is None else read_function,
                               parse_function=parse_function_0 if parse_function is None else parse_function,
                               glob_pattern="*.npz",
                               num_preload=num_preload)

    elif type == "depth_png":
        return DirectoryReader(path_to_folder,
                               sorting_function=index_function_0,
                               read_function=read_function_2 if read_function is None else read_function,
                               parse_function=parse_function_0 if parse_function is None else parse_function,
                               glob_pattern="*.png",
                               num_preload=num_preload)
    else:
        raise NotImplementedError(f"Type {type} not implemented")


if __name__ == '__main__':
    import time

    s = time.time()
    f = get_directory_reader('/Users/liyutong/Downloads/bulu-0-0/011422071122/color', "color_bmp", num_preload=8)

    # num_preload:
    # - 8 is the optimal value for Apple M1 (700MB/s)
    while not f.eof:
        frame, meta = f.next()

    print(time.time() - s)
