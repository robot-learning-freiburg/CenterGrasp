import numpy as np
from tqdm import tqdm
from typing import Tuple
from graspnetAPI import GraspGroup
from centergrasp.configs import Directories
from centergrasp.sapien.sapien_utils import Obs, CameraObs
from centergrasp.graspnet.rgb_data import RGBDReader, KINECT_HALF_PARAMS
from centergrasp.pipelines.centergrasp_pipeline import CenterGraspPipeline


def grasp_panda_to_graspnet(grasp_poses: np.ndarray) -> Tuple[np.ndarray]:
    centers = grasp_poses[:, :3, 3] + (0.0624 + 0.02) * grasp_poses[:, :3, 2]
    rotations = np.zeros_like(grasp_poses[:, :3, :3])
    rotations[..., 0] = grasp_poses[:, :3, 2]
    rotations[..., 1] = grasp_poses[:, :3, 1]
    rotations[..., 2] = -grasp_poses[:, :3, 0]
    rotations = rotations.reshape(-1, 9)
    return rotations, centers


def main(rgb_model: str):
    dump_folder = Directories.EVAL_GRASPNET
    rgbd_reader = RGBDReader(mode="test")
    pipeline = CenterGraspPipeline(rgb_model, camera_params=KINECT_HALF_PARAMS)

    grasp_width = 0.08
    grasp_height = 0.02  # finger thickness
    grasp_depth = 0.05 - 0.02  # finger length (from center)
    for i in tqdm(range(len(rgbd_reader))):
        data = rgbd_reader.get_data_np(i)
        scene_name, img_name = rgbd_reader.get_scene_img_names(i)
        camera_obs = CameraObs(data.rgb, depth_real=data.depth[..., np.newaxis])
        postpr_preds, _ = pipeline._centergrasp_predictions(Obs(camera_obs), num_grasps=4000)
        out_arrays = []
        for postpr_pred in postpr_preds:
            if len(postpr_pred.grasp_poses) == 0:
                continue
            rotations, centers = grasp_panda_to_graspnet(postpr_pred.grasp_poses)
            lowest_score = 1 - (0.01 * len(centers))
            scores = np.linspace(start=1, stop=lowest_score, num=len(centers), endpoint=False)
            scores = scores[..., np.newaxis]
            widths = np.ones((len(centers), 1)) * grasp_width
            heights = np.ones((len(centers), 1)) * grasp_height
            depths = np.ones((len(centers), 1)) * grasp_depth
            obj_ids = np.zeros_like(scores)
            out = np.hstack((scores, widths, heights, depths, rotations, centers, obj_ids))
            out_arrays.append(out)
        out_arrays = np.concatenate(out_arrays) if len(out_arrays) > 0 else np.zeros((1, 17))
        grasp_group = GraspGroup(out_arrays)
        save_path = dump_folder / scene_name / "kinect"
        save_path.mkdir(parents=True, exist_ok=True)
        grasp_group.save_npy(save_path / (img_name + ".npy"))
    return


if __name__ == "__main__":
    main(rgb_model="el6oa23g")
