import pathlib
import argparse
import numpy as np
import rerun as rr
import centergrasp.mesh_utils as mesh_utils
import centergrasp.sgdf.mesh_grasps as mesh_grasps
from centergrasp.giga_utils import MeshPathsLoader
from centergrasp.sapien.sapien_utils import Obs, CameraObs
from centergrasp.pipelines.centergrasp_pipeline import CenterGraspPipeline
from centergrasp.rgb.pred_postprocessing import postprocess_predictions
from centergrasp.rgb.rgb_data import RGBDReader, RGBDReaderReal, RGBDataset
from centergrasp.rgb.data_structures import RgbdDataNp
from centergrasp.rgb.rgb_inference import extract_obj_predictions
from centergrasp.rgb.data_structures import FullObjPred
from centergrasp.sgdf.sgdf_inference import SGDFInference
from centergrasp.visualization import RerunViewer
from centergrasp.configs import ZED2HALF_PARAMS


def get_meshpath(obj_name: str = "", group: str = "packed") -> pathlib.Path:
    mesh_loader = MeshPathsLoader()
    mesh_path = (
        mesh_loader.get_from_name(obj_name, group) if obj_name != "" else mesh_loader.get_random()
    )
    return mesh_path


def vis_object(mesh_path: pathlib.Path, num_grasps: int, transform: np.ndarray, scale: float = 1.0):
    obj_name = mesh_path.stem
    RerunViewer.add_axis("vis/axis", transform)
    # Meshes
    print(mesh_path)
    mesh_trimesh = mesh_utils.load_mesh_trimesh(mesh_path, scale)
    mesh_trimesh.apply_transform(transform)
    RerunViewer.add_mesh_trimesh(f"vis/meshes/{obj_name}", mesh_trimesh)
    # Grasps
    if num_grasps > 0:
        rng = np.random.default_rng()
        grasp_poses_all = mesh_grasps.read_poses_data(mesh_path, scale, frame="ttip")
        if len(grasp_poses_all) == 0:
            return
        grasp_poses = rng.choice(grasp_poses_all, num_grasps, replace=False)
        grasp_poses = transform @ grasp_poses
        RerunViewer.add_grasps(f"vis/grasps/{obj_name}/", grasp_poses)
    return


def vis_object_grasps(num_grasps: int, obj_name: str, group: str):
    mesh_path = get_meshpath(obj_name, group)
    vis_object(mesh_path, num_grasps, transform=np.eye(4))
    return


def get_grid_corners(num_points, distance):
    grid_corners = []
    x_points = int(np.ceil(np.sqrt(num_points)))
    y_points = int(np.floor(np.sqrt(num_points)))
    if x_points * y_points < num_points:
        y_points += 1
    x_width = x_points * distance
    y_width = y_points * distance
    for x in np.linspace(-x_width / 2, x_width / 2, x_points):
        for y in np.linspace(-y_width / 2, y_width / 2, y_points):
            grid_corners.append([x, y, 0.0])
    return grid_corners


def vis_all_objects_grasps(num_grasps: int):
    mesh_paths = MeshPathsLoader.get_list("all")
    grid_corners = get_grid_corners(num_points=len(mesh_paths), distance=0.3)
    for mesh_path in mesh_paths:
        obj_pose = np.eye(4)
        obj_pose[:3, 3] = grid_corners.pop()
        vis_object(mesh_path, num_grasps, transform=obj_pose)
    return


def vis_sgdf_net(sgdf_model: str, obj_name: str = "", group: str = "packed", scale: float = 1.0):
    sgdf_inference = SGDFInference(sgdf_model)
    while True:
        mesh_path = get_meshpath(obj_name, group)
        mesh_trimesh = mesh_utils.load_mesh_trimesh(mesh_path, scale=scale)
        sgdf_prediction = sgdf_inference.predict_from_meshpath(mesh_path=mesh_path, scale=scale)
        RerunViewer.add_axis("vis/axis", np.eye(4))
        RerunViewer.add_mesh_trimesh("vis/mesh", mesh_trimesh)
        RerunViewer.vis_sgdf_prediction("pred", sgdf_prediction)
        input("Press enter to continue...")
    return


def vis_rgbd(start_idx: int, mode: str):
    rgbd_reader = RGBDReader(mode)
    for i in range(start_idx, start_idx + 1000):
        data = rgbd_reader.get_data_np(i)
        rr.set_time_sequence(timeline="frame_idx", sequence=i)
        RerunViewer.vis_rgbd_data(data)
        input("Press enter to continue...")
    return


def vis_sgdf_net_scene(sgdf_model: str, start_idx: int, mode: str):
    rgbd_dataset = RGBDataset(sgdf_model, mode)
    sgdf_inference = SGDFInference(sgdf_model)
    for i in range(start_idx, start_idx + 1000):
        RerunViewer.clear()
        rgb, depth, heatmap_target, pose_target, shape_target, _ = rgbd_dataset[i]
        rgbd_data = RgbdDataNp.from_torch(rgb, depth, heatmap_target)
        rgb_gt_preds = extract_obj_predictions(heatmap_target, pose_target, shape_target)
        sgdf_preds = [
            sgdf_inference.predict_reconstruction(pred.embedding) for pred in rgb_gt_preds
        ]
        full_preds = [
            FullObjPred.from_net_predictions(rgb_pred, sgdf_pred)
            for rgb_pred, sgdf_pred in zip(rgb_gt_preds, sgdf_preds)
        ]
        postpr_preds, full_pcd = postprocess_predictions(rgbd_data, full_preds, use_icp=True)
        rr.set_time_sequence(timeline="frame_idx", sequence=i)
        RerunViewer.vis_rgbd_data(rgbd_data)
        RerunViewer.add_o3d_pointcloud("vis/full_pcd", full_pcd, radii=0.0015)
        for idx in range(len(postpr_preds)):
            RerunViewer.visualize_prediction(full_preds[idx], postpr_preds[idx], idx)
        input("Press enter to continue...")
    return


def vis_full_prediction_sim(rgb_model: str, start_idx: int, mode: str):
    rgbd_reader = RGBDReader(mode)
    pipeline = CenterGraspPipeline(rgb_model, camera_params=ZED2HALF_PARAMS)
    for i in range(start_idx, start_idx + 1000):
        data = rgbd_reader.get_data_np(i)
        camera_obs = CameraObs(data.rgb, depth_real=data.depth[..., np.newaxis])
        _, _ = pipeline._centergrasp_predictions(Obs(camera_obs))
        input("Press enter to continue...")
    return


def vis_full_prediction_real(rgb_model: str, start_idx: int):
    rgbd_reader = RGBDReaderReal()
    pipeline = CenterGraspPipeline(rgb_model, camera_params=ZED2HALF_PARAMS)
    for i in range(start_idx, len(rgbd_reader)):
        data = rgbd_reader.get_data_np(i)
        camera_obs = CameraObs(data.rgb, depth_real=data.depth[..., np.newaxis])
        _, _ = pipeline._centergrasp_predictions(Obs(camera_obs))
        input("Press enter to continue...")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="subcommands", required=True)

    # Visualize a single object with grasps
    grasps_single_object_parser = subparsers.add_parser("grasps-single-object")
    grasps_single_object_parser.add_argument("--num-grasps", type=int, default=10)
    grasps_single_object_parser.add_argument("--obj-name", type=str, default="")
    grasps_single_object_parser.add_argument(
        "--group", type=str, default="packed", choices=["packed", "pile"]
    )
    grasps_single_object_parser.set_defaults(func=vis_object_grasps)

    # Visualize all objects with grasps
    grasps_all_objects_parser = subparsers.add_parser("grasps-all-objects")
    grasps_all_objects_parser.add_argument("--num-grasps", type=int, default=10)
    grasps_all_objects_parser.set_defaults(func=vis_all_objects_grasps)

    # Visualize sgdf net
    sgdf_net_parser = subparsers.add_parser("sgdf-net")
    sgdf_net_parser.add_argument("--sgdf-model", type=str, default="9vkd9370")
    sgdf_net_parser.add_argument("--obj-name", type=str, default="")
    sgdf_net_parser.add_argument("--group", type=str, default="packed", choices=["packed", "pile"])
    sgdf_net_parser.add_argument("--scale", type=float, default=1.0)
    sgdf_net_parser.set_defaults(func=vis_sgdf_net)

    # Visualize sgdf net scene
    sgdf_net_scene_parser = subparsers.add_parser("sgdf-net-scene")
    sgdf_net_scene_parser.add_argument("--sgdf-model", type=str, default="9vkd9370")
    sgdf_net_scene_parser.add_argument("--start-idx", type=int, default=0)
    sgdf_net_scene_parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "valid"]
    )
    sgdf_net_scene_parser.set_defaults(func=vis_sgdf_net_scene)

    # Visualize rgbd from idx
    rgbd_parser = subparsers.add_parser("rgbd")
    rgbd_parser.add_argument("--start-idx", type=int, default=0)
    rgbd_parser.add_argument("--mode", type=str, default="train", choices=["train", "valid"])
    rgbd_parser.set_defaults(func=vis_rgbd)

    # Visualize full prediction sim
    full_prediction_parser = subparsers.add_parser("full-prediction-sim")
    full_prediction_parser.add_argument("--rgb-model", type=str, default="12c7ven5")
    full_prediction_parser.add_argument("--start-idx", type=int, default=0)
    full_prediction_parser.add_argument(
        "--mode", type=str, default="valid", choices=["train", "valid", "real"]
    )
    full_prediction_parser.set_defaults(func=vis_full_prediction_sim)

    # Visualize full prediction real
    full_prediction_parser = subparsers.add_parser("full-prediction-real")
    full_prediction_parser.add_argument("--rgb-model", type=str, default="12c7ven5")
    full_prediction_parser.add_argument("--start-idx", type=int, default=0)
    full_prediction_parser.set_defaults(func=vis_full_prediction_real)

    # Call the function
    RerunViewer()
    args = parser.parse_args()
    func = args.func
    del args.func
    func(**vars(args))
