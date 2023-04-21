import copy
import dataclasses
import logging
import os
import os.path as osp
import pickle
from typing import Union, List, Optional, Dict, Tuple

import cv2
import numpy as np
import open3d as o3d
import pupil_apriltags
import json
from scipy.spatial.transform.rotation import Rotation as R
import tqdm

logger = logging.getLogger("realsense_recorder.filters.fiducial")

from realsense_recorder.common.utils import rvec_tvec_to_matrix, R_T_to_matrix

@dataclasses.dataclass()
class AprilTagFiducialDetectHelper:
    marker_length_m: float  # in meter
    april_tag_family: str = "tag25h9"
    camera_matrix: Union[List[List[float]], np.ndarray] = None  # [[ fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    camera_distort: Optional[Union[List[float], np.ndarray]] = None  # [ 0.1927826544288516, -0.34972530095573834, 0.011612480526787846, -0.00393533140166019, -2.9216752723525734 ]
    dof_cache: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
    font: int = cv2.FONT_HERSHEY_SIMPLEX

    def __post_init__(self):
        self.marker_points = np.array([[-self.marker_length_m / 2, self.marker_length_m / 2, 0],
                                       [self.marker_length_m / 2, self.marker_length_m / 2, 0],
                                       [self.marker_length_m / 2, -self.marker_length_m / 2, 0],
                                       [-self.marker_length_m / 2, -self.marker_length_m / 2, 0]])

        self.detector = pupil_apriltags.Detector(
            families=self.april_tag_family,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        if self.camera_distort is not None:
            if isinstance(self.camera_distort, list):
                self.camera_distort = np.array(self.camera_distort)
            self.camera_distort = self.camera_distort.squeeze()
            assert self.camera_distort.shape[0] in [4, 5, 8, 12, 14]

        if self.camera_matrix is not None:
            if isinstance(self.camera_matrix, list):
                self.camera_matrix = np.array(self.camera_matrix)
            assert self.camera_matrix.shape == (3, 3)

        self.dof_cache = {}
        self.dof_freq = {}

    @staticmethod
    def depth2xyz(u, v, depth, K):
        x = (u - K[0, 2]) * depth / K[0, 0]
        y = (v - K[1, 2]) * depth / K[1, 1]
        return np.array([x, y, depth]).reshape(3, 1)

    @staticmethod
    def apply_polygon_mask_color(img: np.ndarray, polygons: List[np.ndarray]):
        mask = np.zeros(img.shape, dtype=img.dtype)
        for polygon in polygons:
            polygon = polygon.reshape(-1, 2).astype(np.int32)
            cv2.fillConvexPoly(mask, polygon, (255, 255, 255))
        res = cv2.bitwise_and(img, mask)
        return res

    @staticmethod
    def apply_polygon_mask_depth(img: np.ndarray, polygons: List[np.ndarray]):
        mask = np.zeros(img.shape, dtype=img.dtype)
        for polygon in polygons:
            polygon = polygon.reshape(-1, 2).astype(np.int32)
            cv2.fillConvexPoly(mask, polygon, (np.iinfo(img.dtype).max))
        res = cv2.bitwise_and(img, mask)
        return res

    @classmethod
    def create_roi_masked_point_cloud(cls,
                                      color_frame: np.ndarray,
                                      depth_frame: np.ndarray,
                                      corners: List[np.ndarray],
                                      camera_matrix: np.ndarray) -> Tuple[Optional[o3d.geometry.PointCloud], Optional[Exception]]:
        color_frame_masked = cls.apply_polygon_mask_color(color_frame, [corner.reshape(-1, 2).astype(np.int32) for corner in corners])
        depth_frame_masked = cls.apply_polygon_mask_depth(depth_frame, [corner.reshape(-1, 2).astype(np.int32) for corner in corners])

        if depth_frame_masked.sum() > 0:
            color_frame_o3d = o3d.geometry.Image(cv2.cvtColor(color_frame_masked.astype(np.uint8), cv2.COLOR_BGRA2RGB))
            depth_frame_o3d = o3d.geometry.Image(depth_frame_masked.astype(np.uint16))
            rgbd_image = o3d.geometry.RGBDImage().create_from_color_and_depth(
                color_frame_o3d,
                depth_frame_o3d,
                4000.,  # MAGIC NUMBER
                2.,  # MAGIC NUMBER
                convert_rgb_to_intensity=False
            )

            pcd = o3d.geometry.PointCloud().create_from_rgbd_image(
                image=rgbd_image,
                intrinsic=o3d.camera.PinholeCameraIntrinsic(
                    depth_frame_masked.shape[1],
                    depth_frame_masked.shape[0],
                    camera_matrix[0][0],
                    camera_matrix[1][1],
                    camera_matrix[0][2],
                    camera_matrix[1][2],
                ),
                extrinsic=np.eye(4),
            )
            if len(pcd.points) < 4:
                return None, Exception("no enough depth")
            else:
                return pcd, None
        else:
            return None, Exception("no depth")

    def reject_false_rotation(self,
                              color_frame,
                              depth_frame,
                              rvec: np.ndarray,
                              tvec: np.ndarray,
                              corners: np.ndarray,
                              debug=False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Exception]]:
        pcd, err = self.create_roi_masked_point_cloud(color_frame, depth_frame, [corners], self.camera_matrix)
        if err is None:
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.marker_length_m, max_nn=30)
            )
            position = np.asarray(pcd.points).mean(axis=0)
            normals = np.asarray(pcd.normals)
            normal_vec = normals.mean(axis=0)
            normal_vec /= np.linalg.norm(normal_vec)

            z_axis = np.array([[0], [0], [1]])
            candidates = [cv2.Rodrigues(x)[0] @ z_axis for x in rvec]
            score = np.array([float(abs(np.dot(x.T, normal_vec))) for x in candidates])

            if debug:
                arrow = o3d.geometry.TriangleMesh().create_arrow(
                    cylinder_radius=self.marker_length_m / 20,
                    cone_radius=self.marker_length_m / 10,
                    cylinder_height=self.marker_length_m / 2,
                    cone_height=self.marker_length_m / 2,
                )
                R, _ = cv2.Rodrigues(np.cross(normal_vec, [0, 0, 1]))
                # arrow.scale(3, center=arrow.get_center())
                arrow.translate(position)
                arrow.rotate(R, center=arrow.get_center())

                o3d.visualization.draw_geometries([o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.3, origin=[0, 0, 0]), pcd, arrow])
                # o3d.visualization.draw_geometries([o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.3, origin=[0, 0, 0]), arrow])

            return rvec[score.argmax()], tvec[score.argmax()], None
        else:
            return None, None, Exception("no depth")

    def preview_one_frame(self, color_frame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        output_color_frame = copy.deepcopy(color_frame)
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(output_color_frame, corners, ids)
            logger.info(f"detected {len(corners)} markers, with ids {ids}")
        return output_color_frame, corners, ids

    # num = 0
    def process_one_frame(self,
                          color_frame,
                          depth_frame,
                          depth_scale: float = 1000.,
                          undistort=True,
                          debug=False):
        undistort = False if self.camera_distort is None else undistort
        if self.camera_matrix is None:
            return None, None, None, Exception("no camera matrix")

        if undistort:
            color_frame_undistort = cv2.undistort(color_frame, self.camera_matrix, self.camera_distort)
            depth_frame_undistort = cv2.undistort(depth_frame, self.camera_matrix, self.camera_distort)
        else:
            color_frame_undistort = color_frame
            depth_frame_undistort = depth_frame
        output_color_frame = copy.deepcopy(color_frame_undistort)
        depth_frame_undistort = cv2.medianBlur(depth_frame_undistort, 5)

        gray = cv2.cvtColor(color_frame_undistort, cv2.COLOR_BGR2GRAY)

        detection_results = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(self.camera_matrix[0, 0], self.camera_matrix[1, 1], self.camera_matrix[0, 2], self.camera_matrix[1, 2]),
            tag_size=self.marker_length_m,
        )

        if len(detection_results):
            res = {}
            for detection_result in detection_results:
                pose_rot = detection_result.pose_R
                pose_t = detection_result.pose_t
                assert len(pose_rot.shape) == len(pose_t.shape) == 2

                rvec = R.from_matrix(pose_rot).as_rotvec()
                tvec = pose_t.reshape((3,))

                res[detection_result.tag_id] = (rvec, tvec, True)

            return res, output_color_frame, depth_frame_undistort, None
        else:
            return None, output_color_frame, depth_frame_undistort, Exception("no marker found")

    def vis_3d(self, detection: Dict[str, Tuple[np.ndarray, np.ndarray]], color_frame=None, depth_frame=None):
        marker_meshes = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])]
        for marker_id, (rvec, tvec, is_unique) in detection.items():
            if is_unique:
                new_mesh = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1, origin=tvec)
                R, _ = cv2.Rodrigues(rvec)
                new_mesh.rotate(R, center=tvec)
                marker_meshes.append(new_mesh)
            else:
                for i in range(len(rvec)):
                    new_mesh = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1, origin=tvec[i])
                    R, _ = cv2.Rodrigues(rvec[i])
                    new_mesh.rotate(R, center=tvec[i])
                    marker_meshes.append(new_mesh)
        if color_frame is not None:
            color_frame_o3d = o3d.geometry.Image(cv2.cvtColor(color_frame.astype(np.uint8), cv2.COLOR_BGRA2RGB))
            depth_frame_o3d = o3d.geometry.Image(depth_frame.astype(np.uint16))

            rgbd_image = o3d.geometry.RGBDImage().create_from_color_and_depth(
                color_frame_o3d,
                depth_frame_o3d,
                4000.,  # MAGIC NUMBER
                2.,  # MAGIC NUMBER
                convert_rgb_to_intensity=False
            )

            pcd = o3d.geometry.PointCloud().create_from_rgbd_image(
                image=rgbd_image,
                intrinsic=o3d.camera.PinholeCameraIntrinsic(
                    depth_frame.shape[1],
                    depth_frame.shape[0],
                    self.camera_matrix[0, 0],
                    self.camera_matrix[1, 1],
                    self.camera_matrix[0, 2],
                    self.camera_matrix[1, 2],
                ),
                extrinsic=np.eye(4),
            )
            marker_meshes.append(pcd)
        o3d.visualization.draw_geometries(marker_meshes)
        return marker_meshes

    def vis_2d(self, detection: Dict[str, Tuple[np.ndarray, np.ndarray]], color_frame):
        for marker_id, (rvec, tvec, is_unique) in detection.items():
            if is_unique:
                cv2.drawFrameAxes(color_frame, self.camera_matrix, self.camera_distort[:5], rvec, tvec, 0.03, 2)
            else:
                cv2.drawFrameAxes(color_frame, self.camera_matrix, self.camera_distort[:5], rvec[0], tvec[1], 0.03, 2)
        cv2.putText(color_frame, "Id: " + str(list(detection.keys())), (0, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        scale = int(color_frame.shape[0] / 480)
        cv2.imshow("frame", color_frame[::scale, ::scale, :])
        key = cv2.waitKey(0)
        return color_frame


def detect_fiducial_marker(base_dir: str,
                           enabled_cameras: List[str] = None,
                           marker_length_m: float = 0.032,
                           april_tag_family: str = "tag25h9",
                           sequence_length: int = None,
                           debug: bool = False,
                           ):
    selected_frames_info = json.load(open(os.path.join(base_dir, "selected_frames.json")))
    sequence_length = selected_frames_info['meta']['sequence_length'] if sequence_length is None else sequence_length
    calibration = json.load(open(os.path.join(base_dir, 'calibration.json')))

    cam_ids = list(calibration['cameras'].keys())
    master_cam_id = selected_frames_info['meta']['master_camera']
    enabled_camera_ids = cam_ids if enabled_cameras is None else enabled_cameras

    transformation_matrix = {
        cam_id: np.eye(4) if cam_id == master_cam_id else R_T_to_matrix(**calibration['camera_poses'][cam_id + '_to_' + master_cam_id]) for cam_id in enabled_camera_ids
    }

    ctxs = [
        AprilTagFiducialDetectHelper(
            marker_length_m=marker_length_m,
            april_tag_family=april_tag_family,
            camera_matrix=np.array(calibration['cameras'][cam]['K']),
            camera_distort=np.array(calibration['cameras'][cam]['dist']),
        ) for cam in enabled_camera_ids
    ]

    # frames_list_dict = {
    #     [
    #         {cam_id: {
    #             'color': selected_frames_info['filenames'][cam_id][idx]['color'],
    #             'depth': selected_frames_info['filenames'][cam_id][idx]['depth'],
    #         } for cam_id in cam_ids}
    #     ]
    #     for idx in range(sequence_length)
    # }

    marker_detection_res = []
    with tqdm.tqdm(total=sequence_length) as pbar:
        for frame_idx in range(0, sequence_length):
            color_frames = [
                cv2.imread(
                    os.path.join(
                        base_dir,
                        cam_id,
                        'color',
                        selected_frames_info['sequence_view'][frame_idx][cam_id]['color']
                    )
                ) for cam_id in enabled_camera_ids
            ]
            depth_frames = [
                np.load(
                    os.path.join(
                        base_dir,
                        cam_id,
                        'depth',
                        selected_frames_info['sequence_view'][frame_idx][cam_id]['depth']
                    )
                )['arr_0'] for cam_id in enabled_camera_ids
            ]

            # color_frames = [ctx.preview_one_frame(x)[0] for x in color_frames]
            marker_curr_res = {}
            for cam_id, color_frame, depth_frame, ctx in zip(enabled_camera_ids, color_frames, depth_frames, ctxs):
                res, processed_color_frame, processed_depth_frame, err = ctx.process_one_frame(
                    color_frame, depth_frame, undistort=True, depth_scale=4000)  # MAGIC NUMBER

                if res is None:
                    continue

                for marker_id, result in res.items():
                    rvec, tvec, _ = result
                    marker_trans_mat = np.matmul(np.linalg.inv(transformation_matrix[cam_id]), rvec_tvec_to_matrix(rvec, tvec))
                    if marker_id not in marker_curr_res.keys():
                        marker_curr_res[marker_id] = [(cam_id, marker_trans_mat[:3,3])]  # rvec, xyz, err
                    else:
                        marker_curr_res[marker_id].append((cam_id, marker_trans_mat[:3,3]))

                if debug:
                    ctx.vis_2d(res, processed_color_frame)
                    ctx.vis_3d(res, processed_color_frame, processed_depth_frame)
                    # cv2.imshow(cam, frame)


            marker_detection_res.append({
                k: list(map(lambda x: (x[0], x[1].tolist()), v)) for k, v in marker_curr_res.items()
            })
            pbar.update()
        # print(marker_curr_res)
        # for marker_id, result in marker_curr_res.items():
        #     if len(result) < 2:
        #         continue
        #     print(frame_idx, marker_id, np.linalg.norm((result[0] - result[1])))
        #     print(i, marker_id, result[0].tolist(), result[1].tolist())
    with open(osp.join(base_dir, "marker_detection_result.json"), "w") as f:
        json.dump(marker_detection_res, f, indent=4)
