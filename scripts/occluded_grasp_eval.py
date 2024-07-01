import numpy as np
import centergrasp.data_utils as data_utils
from centergrasp.sapien.sapien_utils import Obs, CameraObs
from centergrasp.configs import Directories, ZED2HALF_PARAMS
from centergrasp.pipelines.centergrasp_pipeline import CenterGraspPipeline

# Before running the script, remember to comment out the "Filter grasp not aligned with camera"
# line in pred_postprocessing.py

rgb_path = Directories.RGBD / "real/rgb/000015.png"
depth_path = Directories.RGBD / "real/depth/000015.png"
rgb = data_utils.load_rgb_from_file(rgb_path)
depth = data_utils.load_depth_from_file(depth_path)
rgb_model = "12c7ven5"

pipeline = CenterGraspPipeline(rgb_model, camera_params=ZED2HALF_PARAMS)
camera_obs = CameraObs(rgb, depth_real=depth[..., np.newaxis])
postpr_preds, _ = pipeline._centergrasp_predictions(Obs(camera_obs), num_grasps=10000)
assert len(postpr_preds) == 1, "Only one object should be detected"
grasp_poses = postpr_preds[0].grasp_poses

print("Number of grasps detected:", len(grasp_poses))

grasps_on_visible_region = grasp_poses[np.where(grasp_poses[:, 2, 2] > 0)]
grasps_on_nonvisible_region = grasp_poses[np.where(grasp_poses[:, 2, 2] <= 0)]

print("Number of grasps on visible region:", len(grasps_on_visible_region))
print("Number of grasps on non-visible region:", len(grasps_on_nonvisible_region))

ratio_visible = len(grasps_on_visible_region) / len(grasp_poses)
ration_nonvisible = len(grasps_on_nonvisible_region) / len(grasp_poses)

print("Ratio of grasps on visible region:", ratio_visible)
print("Ratio of grasps on non-visible region:", ration_nonvisible)
