import time
import numpy as np
import spatialmath as sm
from typing import Optional
from centergrasp.configs import ZED2HALF_PARAMS
from centergrasp.sapien.sapien_utils import Obs
from centergrasp.visualization import RerunViewer
from contact_graspnet.inference_class import ContactGraspNetInference
from uois.inference import UOISInference


class ContactGraspNetPipeline:
    def __init__(
        self,
        use_uois=False,
        seed: int = 123,
        visualize: bool = True,
    ) -> None:
        self.use_uois = use_uois
        self.visualize = visualize
        self.depth_K = ZED2HALF_PARAMS.K
        self.rng = np.random.default_rng(seed)
        self.contactgraspnet = ContactGraspNetInference()
        self.segm_net = UOISInference() if self.use_uois else None
        return

    def predict_grasp(self, obs: Obs, confidence_map=None) -> sm.SE3:
        wTeegoal, _, _ = self._get_grasp(obs.camera.rgb, obs.camera.depth, obs.camera_pose)
        return wTeegoal

    def _get_grasp(self, rgb_uint8, depth, wTcam):
        depth = np.squeeze(depth)
        start_time = time.time()
        segmap = self.segm_net.predict(rgb_uint8, depth, self.depth_K) if self.use_uois else None
        pc_full, pc_colors, pred_grasps_cam, scores = self.contactgraspnet.predict(
            rgb_uint8, depth, self.depth_K, segmap
        )
        pred_grasps_cam_np = np.concatenate([arr for arr in pred_grasps_cam.values()])
        scores_np = np.concatenate([arr for arr in scores.values()])
        inference_time = time.time() - start_time

        if len(pred_grasps_cam_np) > 0:
            best_idx = np.argmax(scores_np)
            best_grasp_cam = sm.SE3(pred_grasps_cam_np[best_idx], check=False)
            best_grasp_cam = best_grasp_cam * sm.SE3.Rz(-np.pi / 2)  # Acronym to urdf hand frame
            wTeegoal = [self.extract_grasp(best_grasp_cam, wTcam)]
        else:
            best_grasp_cam = None
            wTeegoal = []
        if self.visualize:
            self.show_vis(
                pc_full,
                pc_colors,
                best_grasp_cam,
                rgb_uint8,
                depth,
            )
        return wTeegoal, inference_time, None

    def extract_grasp(self, chosen_grasp, wTcam):
        camTgrasp_1 = sm.SE3(chosen_grasp, check=False)
        camTgrasp_2 = sm.SE3(chosen_grasp, check=False) * sm.SE3.Rz(-np.pi)
        error_rpy_1 = sm.smb.tr2rpy(camTgrasp_1.R, unit="rad", order="zyx", check=False)
        error_rpy_2 = sm.smb.tr2rpy(camTgrasp_2.R, unit="rad", order="zyx", check=False)
        camTgrasp = (
            camTgrasp_1
            if np.linalg.norm(error_rpy_1) <= np.linalg.norm(error_rpy_2)
            else camTgrasp_2
        )
        wTeegoal = wTcam * camTgrasp
        return wTeegoal

    def show_vis(self, pc_full, pc_colors, grasp: Optional[sm.SE3], rgb, depth):
        RerunViewer.clear()
        RerunViewer.add_rgb("rgb", rgb)
        RerunViewer.add_depth("depth", depth)
        RerunViewer.add_np_pointcloud(
            "vis/contactgn_pcd", points=pc_full, colors_uint8=pc_colors, radii=0.002
        )
        if grasp is not None:
            RerunViewer.add_grasps("vis/contactgn_grasps", [grasp.A])
            RerunViewer.add_axis("vis/grasp_pose", grasp.A, size=0.002)
        return


class ContactGraspNetUOIS(ContactGraspNetPipeline):
    def __init__(self, use_uois=True) -> None:
        super().__init__(use_uois)
        return
