import json
from os import path as osp

import numpy as np
from typing import Tuple


def query_closest_values(query: np.ndarray, source: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    diff = query[:, np.newaxis] - source
    abs_diff = np.abs(diff)
    min_index = np.argmin(abs_diff, axis=1)
    closest_values = source[min_index]
    return closest_values, min_index


def sync_cameras(base_dir: str,
                 drop_first_n_frames: int = 30,
                 sync_method: str = 'backend_t'):
    calibration = json.load(open(osp.join(base_dir, 'calibration.json')))
    recording_metadata = json.load(open(osp.join(base_dir, 'metadata_all.json')))
    cam_ids = list(recording_metadata['metadata'].keys())

    timestamp_map = {
        cam_id: np.array(
            list(
                map(
                    lambda x: x['dev_ts'][sync_method],
                    recording_metadata['metadata'][cam_id]
                )
            )
        ) for cam_id in cam_ids
    }
    frame_counter_map = {
        cam_id: np.array(
            list(
                map(
                    lambda x: x['frame_counter'],
                    recording_metadata['metadata'][cam_id]
                )
            )
        ) for cam_id in cam_ids
    }
    filename_map = {
        cam_id: np.array(
            list(
                map(
                    lambda x: x['frame_basename'],
                    recording_metadata['metadata'][cam_id]
                )
            )
        ) for cam_id in cam_ids
    }

    master_cam_id = list(filter(lambda x: 'to' not in x, calibration['camera_poses'].keys()))[0]
    slave_cam_ids = list(filter(lambda x: x != master_cam_id, calibration['cameras'].keys()))

    closest_value_map = {
        master_cam_id: timestamp_map[master_cam_id]
    }
    closest_index_map = {
        master_cam_id: np.arange(len(timestamp_map[master_cam_id]))
    }
    for cam_id in cam_ids:
        if cam_id != master_cam_id:
            closest_value_map[cam_id], closest_index_map[cam_id] = query_closest_values(timestamp_map[master_cam_id], timestamp_map[cam_id])

    head = 0
    tail = len(timestamp_map[master_cam_id])
    while True:

        if any([
            closest_index_map[cam_id][tail - 1] <= closest_index_map[cam_id][tail - 2] for cam_id in cam_ids
        ]):
            tail -= 1

        else:
            break

    while True:
        if any([
            closest_index_map[cam_id][head] >= closest_index_map[cam_id][head + 1] for cam_id in cam_ids
        ]):
            head += 1
        else:
            break

    while True:
        if not all([frame_counter_map[cam_id][head] > drop_first_n_frames for cam_id in cam_ids]):
            head += 1
        else:
            break

    synced_filenames = {
        cam_id: filename_map[cam_id][closest_index_map[cam_id][head:tail]] for cam_id in cam_ids
    }
    num_selected_frames = sum([len(v) for k, v in synced_filenames.items()])
    num_dropped_frames = sum([len(v) for k, v in timestamp_map.items()]) - num_selected_frames
    sequence_length = tail - head
    selected_frames_dict = {
        "meta": {
            "master_camera": master_cam_id,
            "slave_cameras": slave_cam_ids,
            "num_selected_frames": num_selected_frames,
            "num_dropped_frames": num_dropped_frames,
            'sequence_length': sequence_length,
        },
        "filenames": {
            cam_id: {
                "color": list(map(lambda x: x + ".jpeg", synced_filenames[cam_id])),
                "depth": list(map(lambda x: x + ".npz", synced_filenames[cam_id])),
            } for cam_id in cam_ids
        },
        'sequence_view': [
            {
                cam_id: {
                    'color': synced_filenames[cam_id][i] + '.jpeg',
                    'depth': synced_filenames[cam_id][i] + '.npz'
                } for cam_id in cam_ids
            }
            for i in range(sequence_length)
        ]

    }
    with open(osp.join(base_dir, 'selected_frames.json'), 'w') as f:
        json.dump(selected_frames_dict, f, indent=4)

    pass
