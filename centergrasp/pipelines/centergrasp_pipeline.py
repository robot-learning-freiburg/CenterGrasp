import time
import random
import numpy as np
import open3d as o3d
import spatialmath as sm
from typing import List
from centergrasp.cameras import CameraParams
from centergrasp.sapien.sapien_utils import Obs
from centergrasp.rgb.data_structures import RgbdDataNp
from centergrasp.visualization import RerunViewer
from centergrasp.rgb.rgb_inference import RGBInference
from centergrasp.rgb.pred_postprocessing import postprocess_predictions


class CenterGraspPipeline:
    def __init__(
        self,
        rgb_model: str = "12c7ven5",
        seed: int = 123,
        use_icp: bool = True,
        visualize: bool = True,
        camera_params: CameraParams = None,
    ):
        self.use_icp = use_icp
        self.rgb_net = RGBInference(rgb_model)
        self.rng = np.random.default_rng(seed)
        self.visualize = visualize
        self.camera_params = camera_params
        return

    def predict_grasp(self, obs: Obs, confidence_map=None) -> List[sm.SE3]:
        predictions, _ = self._centergrasp_predictions(obs, confidence_map)
        # Get the best grasp for each object
        best_grasps = [pred.grasp_poses[0] for pred in predictions if len(pred.grasp_poses) > 0]
        wTeegoal_list = [self._extract_grasp(grasp, obs.camera_pose) for grasp in best_grasps]
        # Filter out grasps that are outside of the table
        wTeegoal_list = [wTeegoal for wTeegoal in wTeegoal_list if wTeegoal.t[0] > 0.1]
        return wTeegoal_list

    def _centergrasp_predictions(self, obs: Obs, confidence_map=None, num_grasps=1):
        # Predict
        start_time = time.time()
        heatmap_out, full_preds = self.rgb_net.get_full_predictions(
            obs.camera.rgb, obs.camera.depth
        )

        # Postprocessing
        rgbd_data = RgbdDataNp(obs.camera.rgb, obs.camera.depth, heatmap_out, None, None, None)
        postpr_preds, full_pcd = postprocess_predictions(
            rgbd_data,
            full_preds,
            num_grasps,
            self.use_icp,
            confidence_map,
            obs.joint_state,
            obs.wTbase,
            obs.camera_pose,
            self.camera_params,
        )
        inference_time = time.time() - start_time
        random.shuffle(postpr_preds)

        # Visualize
        if self.visualize:
            # qualysis_addr = "127.0.0.1:9876"
            RerunViewer.clear()
            RerunViewer.vis_rgbd_data(rgbd_data)
            RerunViewer.add_o3d_pointcloud("vis/full_pcd", full_pcd, radii=0.0015)
            for idx in range(len(postpr_preds)):
                RerunViewer.visualize_prediction(full_preds[idx], postpr_preds[idx], idx)

        return postpr_preds, inference_time

    def _extract_grasp(self, chosen_grasp, wTcam) -> sm.SE3:
        camTgrasp_1 = sm.SE3(chosen_grasp, check=False)
        camTgrasp_2 = sm.SE3(chosen_grasp, check=False) * sm.SE3.Rz(-np.pi)
        handTcam = sm.SE3.Rz(np.pi / 2)  # Assumes gripper camera
        handTgrasp_1 = handTcam * camTgrasp_1
        handTgrasp_2 = handTcam * camTgrasp_2
        error_rpy_1 = sm.smb.tr2rpy(handTgrasp_1.R, unit="rad", order="zyx", check=False)
        error_rpy_2 = sm.smb.tr2rpy(handTgrasp_2.R, unit="rad", order="zyx", check=False)
        camTgrasp = (
            camTgrasp_1
            if np.linalg.norm(error_rpy_1) <= np.linalg.norm(error_rpy_2)
            else camTgrasp_2
        )
        wTeegoal = wTcam * camTgrasp
        return wTeegoal

    def predict_shape(self, obs: Obs) -> np.ndarray:
        predictions, _ = self._centergrasp_predictions(obs)
        pcs_list = []
        combined_pcs = o3d.geometry.PointCloud()
        for pred in predictions:
            pred.pc_o3d.transform(obs.camera_pose.A)
            pcs_list.append(np.asarray(pred.pc_o3d.points))
            combined_pcs += pred.pc_o3d
        return np.asarray(combined_pcs.points), pcs_list

    def predict_shape_and_grasps(self, obs: Obs):
        predictions, _ = self._centergrasp_predictions(obs, num_grasps=10)
        # Get shape
        pcs_list = []
        combined_pcs = o3d.geometry.PointCloud()
        for pred in predictions:
            pred.pc_o3d.transform(obs.camera_pose.A)
            pcs_list.append(pred.pc_o3d)
            combined_pcs += pred.pc_o3d
        # Get the best grasp for each object
        all_grasps = [
            grasp for pred in predictions for grasp in pred.grasp_poses if len(pred.grasp_poses) > 0
        ]
        wTeegoal_list_all = [self._extract_grasp(grasp, obs.camera_pose) for grasp in all_grasps]
        # Filter out grasps that are outside of the table
        wTeegoal_list_all = [wTeegoal for wTeegoal in wTeegoal_list_all if wTeegoal.t[0] > 0.1]

        best_grasps = [pred.grasp_poses[0] for pred in predictions if len(pred.grasp_poses) > 0]
        wTeegoal_list_best = [self._extract_grasp(grasp, obs.camera_pose) for grasp in best_grasps]
        # Filter out grasps that are outside of the table
        wTeegoal_list_best = [wTeegoal for wTeegoal in wTeegoal_list_best if wTeegoal.t[0] > 0.1]
        return combined_pcs, pcs_list, wTeegoal_list_all, wTeegoal_list_best
