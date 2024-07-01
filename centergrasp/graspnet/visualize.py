import argparse
import numpy as np
import rerun as rr
from centergrasp.visualization import RerunViewer
from centergrasp.sapien.sapien_utils import Obs, CameraObs
from centergrasp.rgb.data_structures import RgbdDataNp, FullObjPred
from centergrasp.rgb.rgb_inference import extract_obj_predictions
from centergrasp.rgb.pred_postprocessing import postprocess_predictions
from centergrasp.graspnet.sgdf_data import load_trimesh_obj
from centergrasp.pipelines.centergrasp_pipeline import CenterGraspPipeline
from centergrasp.graspnet.sgdf_inference import SGDFInference, SGDFInferenceGT
from centergrasp.graspnet.rgb_data import RGBDReader, RGBDatasetGraspnet, KINECT_HALF_PARAMS


def vis_sgdf_net(sgdf_model: str, start_idx: int):
    sgdf_inference = SGDFInferenceGT(sgdf_model)
    while True:
        mesh_trimesh = load_trimesh_obj(start_idx)
        sgdf_prediction = sgdf_inference.predict_from_objidx(start_idx)
        RerunViewer.add_axis("vis/axis", np.eye(4))
        RerunViewer.add_mesh_trimesh("vis/mesh", mesh_trimesh)
        RerunViewer.vis_sgdf_prediction("pred", sgdf_prediction)
        input("Press enter to continue...")
        start_idx += 1
    return


def vis_rgbd(start_idx: int, mode: str):
    rgbd_reader = RGBDReader(mode)
    for i in range(start_idx, start_idx + 1000):
        data = rgbd_reader.get_data_np(i)
        rr.set_time_sequence(timeline="frame_idx", sequence=i)
        RerunViewer.vis_rgbd_data(data, KINECT_HALF_PARAMS.K)
        input("Press enter to continue...")
    return


def vis_sgdf_net_scene(sgdf_model: str, start_idx: int, mode: str):
    rgbd_dataset = RGBDatasetGraspnet(sgdf_model, mode)
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
        postpr_preds, full_pcd = postprocess_predictions(
            rgbd_data, full_preds, use_icp=True, camera_params=KINECT_HALF_PARAMS
        )
        rr.set_time_sequence(timeline="frame_idx", sequence=i)
        RerunViewer.vis_rgbd_data(rgbd_data)
        RerunViewer.add_o3d_pointcloud("vis/full_pcd", full_pcd, radii=0.0015)
        for idx in range(len(postpr_preds)):
            RerunViewer.visualize_prediction(full_preds[idx], postpr_preds[idx], idx)
        input("Press enter to continue...")
    return


def vis_full_prediction_sim(rgb_model: str, start_idx: int, mode: str):
    rgbd_reader = RGBDReader(mode)
    pipeline = CenterGraspPipeline(rgb_model, camera_params=KINECT_HALF_PARAMS)
    for i in range(start_idx, start_idx + 1000):
        data = rgbd_reader.get_data_np(i)
        camera_obs = CameraObs(data.rgb, depth_real=data.depth[..., np.newaxis])
        _, _ = pipeline._centergrasp_predictions(Obs(camera_obs))
        input("Press enter to continue...")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="subcommands", required=True)

    # Visualize sgdf net
    sgdf_net_parser = subparsers.add_parser("sgdf-net")
    sgdf_net_parser.add_argument("--sgdf-model", type=str, default="6953cfxt")
    sgdf_net_parser.add_argument("--start-idx", type=int, default=0)
    sgdf_net_parser.set_defaults(func=vis_sgdf_net)

    # Visualize rgbd from idx
    rgbd_parser = subparsers.add_parser("rgbd")
    rgbd_parser.add_argument("--start-idx", type=int, default=0)
    rgbd_parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    rgbd_parser.set_defaults(func=vis_rgbd)

    # Visualize sgdf net scene
    sgdf_net_scene_parser = subparsers.add_parser("sgdf-net-scene")
    sgdf_net_scene_parser.add_argument("--sgdf-model", type=str, default="6953cfxt")
    sgdf_net_scene_parser.add_argument("--start-idx", type=int, default=0)
    sgdf_net_scene_parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test"]
    )
    sgdf_net_scene_parser.set_defaults(func=vis_sgdf_net_scene)

    # Visualize full prediction sim
    full_prediction_parser = subparsers.add_parser("full-prediction-sim")
    full_prediction_parser.add_argument("--rgb-model", type=str, default="el6oa23g")
    full_prediction_parser.add_argument("--start-idx", type=int, default=0)
    full_prediction_parser.add_argument(
        "--mode", type=str, default="test", choices=["train", "test", "real"]
    )
    full_prediction_parser.set_defaults(func=vis_full_prediction_sim)

    # Call the function
    RerunViewer()
    args = parser.parse_args()
    func = args.func
    del args.func
    func(**vars(args))
