import numpy as np
import spatialmath as sm
from centergrasp.configs import Directories, ZED2HALF_PARAMS
from centergrasp.giga_utils import GigaParams
from centergrasp.sapien.sapien_utils import Obs
from vgn.utils import visual
from vgn.inference.inference_class import GIGAInference, CameraIntrinsic, ModelType


class GigaPipeline:
    def __init__(
        self,
        mode: str = "packed",
        seed: int = 123,
        visualize: bool = True,
        real_robot: bool = True,
    ):
        camera_intrinsic = CameraIntrinsic(
            width=ZED2HALF_PARAMS.width,
            height=ZED2HALF_PARAMS.height,
            fx=ZED2HALF_PARAMS.fx,
            fy=ZED2HALF_PARAMS.fy,
            cx=(ZED2HALF_PARAMS.width - 1) / 2,
            cy=(ZED2HALF_PARAMS.height - 1) / 2,
        )
        model_dir = Directories.GIGA_REPO / "data/models"
        if mode == "packed":
            model_type = ModelType.packed
        elif mode == "pile":
            model_type = ModelType.pile
        else:
            raise ValueError(f"Invalid mode: {mode}")
        self.giga_inference = GIGAInference(model_dir, camera_intrinsic, model_type)
        ws_origin = np.array([0.7, -0.24, 0.41]) if real_robot else GigaParams.ws_origin
        self.wTtask = sm.SE3.Trans(ws_origin)
        self.palmThand = sm.SE3.Trans([0, 0, -0.0624])
        self.rng = np.random.default_rng(seed)
        self.visualize = visualize
        return

    def predict_grasp(self, obs: Obs, confidence_map=None) -> sm.SE3:
        wTcam = sm.SE3(obs.camera_pose, check=False)
        camTtask = wTcam.inv() * self.wTtask
        grasps, scores, inference_time, tsdf_pc, pred_mesh = self.giga_inference.predict(
            obs.camera.depth, camTtask.A, reconstruction=False
        )
        # Grasps are already sorted by score here
        wTeegoal_list = []
        for grasp in grasps:
            grasp_se3 = sm.SE3.Rt(R=grasp.pose.rotation.as_matrix(), t=grasp.pose.translation)
            wTgrasp = self.wTtask * grasp_se3
            wTeegoal = self.extract_grasp(wTgrasp)
            wTeegoal_list.append(wTeegoal)

        if self.visualize:
            grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            self.giga_inference.visualize(
                grasp_mesh_list,
                wTcam,
                self.wTtask,
                tsdf_pc,
                obs.camera.rgb,
                obs.camera.depth,
                pred_mesh,
            )
        return wTeegoal_list

    def extract_grasp(self, wTgrasp_raw: sm.SE3) -> sm.SE3:
        wTgrasp_1 = wTgrasp_raw
        wTgrasp_2 = wTgrasp_1 * sm.SE3.Rz(-np.pi)
        error_rpy_1 = sm.smb.tr2rpy(wTgrasp_1.R, unit="rad", order="zyx", check=False)
        error_rpy_2 = sm.smb.tr2rpy(wTgrasp_2.R, unit="rad", order="zyx", check=False)
        wTgrasp = (
            wTgrasp_1 if np.linalg.norm(error_rpy_1) <= np.linalg.norm(error_rpy_2) else wTgrasp_2
        )
        wTeegoal = wTgrasp * self.palmThand
        return wTeegoal

    def predict_shape(self, obs: Obs) -> np.ndarray:
        wTcam = sm.SE3(obs.camera_pose, check=False)
        camTtask = wTcam.inv() * self.wTtask
        _, _, _, tsdf_pc, pred_mesh = self.giga_inference.predict(
            obs.camera.depth, camTtask.A, reconstruction=True
        )
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] *= GigaParams.ws_size
        translation_matrix = sm.SE3.Trans(np.ones(3) * GigaParams.ws_size / 2).A
        pred_mesh.apply_transform(translation_matrix @ self.wTtask.A @ scale_matrix)
        pred_pc = (
            np.asarray(pred_mesh.sample(2000)) if len(pred_mesh.vertices) > 0 else np.array([])
        )

        if self.visualize:
            self.giga_inference.visualize(
                [], wTcam, self.wTtask, tsdf_pc, obs.camera.rgb, obs.camera.depth, pred_mesh
            )
        return pred_pc, pred_mesh
