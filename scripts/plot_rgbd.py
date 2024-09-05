import argparse
import trimesh
import numpy as np
import centergrasp.rgb.pose_utils as pose_utils
from centergrasp.rgb.pred_postprocessing import postprocess_predictions
from centergrasp.cameras import CameraConventions
from centergrasp.configs import ZED2HALF_PARAMS, Directories, COLOR_PALETTE
from centergrasp.rgb.rgb_inference import extract_obj_predictions
from centergrasp.rgb.data_structures import FullObjPred
from centergrasp.data_utils import save_rgb, save_depth_colormap
from centergrasp.rgb.rgb_data import RGBDataset
from centergrasp.rgb.data_structures import RgbdDataNp
from centergrasp.sgdf.sgdf_inference import SGDFInference
from centergrasp.visualization import create_markers_multiple
import centergrasp.rgb.heatmaps as heatmaps
from simnet.lib.transform import Pose
import pyrender


def get_pcd_pyrender(pcd_np, color_idx=0):
    sm = trimesh.creation.uv_sphere(radius=0.002)
    sm.visual.vertex_colors = np.array([*COLOR_PALETTE[color_idx % len(COLOR_PALETTE)], 1.0])
    tfs = np.tile(np.eye(4), (len(pcd_np), 1, 1))
    tfs[:, :3, 3] = pcd_np
    pcd_pyrender = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    return pcd_pyrender


def main(
    sgdf_model: str,
    idx: int,
):
    root_path = Directories.DATA / "centergrasp_g" / "plot_rgbd"
    root_path.mkdir(parents=True, exist_ok=True)
    rgbd_dataset = RGBDataset(sgdf_model, mode="train")
    sgdf_inference = SGDFInference(sgdf_model)

    # Pyrender setup
    renderer = pyrender.OffscreenRenderer(ZED2HALF_PARAMS.width, ZED2HALF_PARAMS.height)
    camera = pyrender.IntrinsicsCamera(
        fx=ZED2HALF_PARAMS.fx,
        fy=ZED2HALF_PARAMS.fy,
        cx=ZED2HALF_PARAMS.cx,
        cy=ZED2HALF_PARAMS.cy,
        znear=0.01,
        zfar=10,
    )
    pyrender_scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
    cam_node = pyrender_scene.add(camera)
    cam_pose = np.eye(4) @ CameraConventions.opencv_T_opengl
    pyrender_scene.set_pose(cam_node, cam_pose)

    # Make data
    rgb, depth, heatmap_target, pose_target, shape_target, _ = rgbd_dataset[idx]
    rgbd_data = RgbdDataNp.from_torch(rgb, depth, heatmap_target)
    rgb_gt_preds = extract_obj_predictions(heatmap_target, pose_target, shape_target)
    sgdf_preds = [sgdf_inference.predict_reconstruction(pred.embedding) for pred in rgb_gt_preds]
    full_preds = [
        FullObjPred.from_net_predictions(rgb_pred, sgdf_pred)
        for rgb_pred, sgdf_pred in zip(rgb_gt_preds, sgdf_preds)
    ]
    postpr_preds, _ = postprocess_predictions(rgbd_data, full_preds, num_grasps=8, use_icp=True)

    obj_idx = 0
    # Add shape
    pcd_list = [np.asarray(pred.pc_o3d.points) for pred in postpr_preds]
    pcd_node = pyrender_scene.add(get_pcd_pyrender(pcd_list[obj_idx], 0))
    rendered_pcd, _ = renderer.render(pyrender_scene)
    pyrender_scene.remove_node(pcd_node)

    # Add grasps
    grasps_list = [grasp_pose for grasp_pose in postpr_preds[obj_idx].grasp_poses]
    grasps_trimesh = create_markers_multiple(grasps_list, color=[0, 255, 0], axis_frame=False)
    grasps_pyrender = [pyrender.Mesh.from_trimesh(mesh, smooth=False) for mesh in grasps_trimesh]
    grasp_nodes = [pyrender_scene.add(grasp) for grasp in grasps_pyrender]
    rendered_grasps, _ = renderer.render(pyrender_scene)
    for node in grasp_nodes:
        pyrender_scene.remove_node(node)

    # Add everything
    _ = [pyrender_scene.add(get_pcd_pyrender(pc, i)) for i, pc in enumerate(pcd_list)]
    grasps_list = [grasp_pose for pred in postpr_preds for grasp_pose in pred.grasp_poses]
    grasps_trimesh = create_markers_multiple(grasps_list, color=[0, 255, 0], axis_frame=False)
    grasps_pyrender = [pyrender.Mesh.from_trimesh(mesh, smooth=False) for mesh in grasps_trimesh]
    grasp_nodes = [pyrender_scene.add(grasp) for grasp in grasps_pyrender]
    rendered_full, _ = renderer.render(pyrender_scene)

    # Crop
    w_crop = 200
    h_crop = 100
    rgb_cropped = rgbd_data.rgb[:-h_crop, w_crop:-w_crop]
    depth_cropped = rgbd_data.depth[:-h_crop, w_crop:-w_crop]
    heatmap_cropped = rgbd_data.heatmap[:-h_crop, w_crop:-w_crop]
    heatmap_vis_cropped = heatmaps.visualize_heatmap(rgb_cropped, heatmap_cropped, with_peaks=True)
    rendered_pcd = rendered_pcd[:-h_crop, w_crop:-w_crop]
    rendered_grasps = rendered_grasps[:-h_crop, w_crop:-w_crop]
    rendered_full = rendered_full[:-h_crop, w_crop:-w_crop]
    heatmap_vis = heatmaps.visualize_heatmap(rgbd_data.rgb, rgbd_data.heatmap, with_peaks=True)
    poses = [pred.pose for pred in full_preds]
    _poses = [Pose(camera_T_object=pose) for pose in poses]
    poses_vis = pose_utils.visualize_poses(rgbd_data.rgb, _poses)

    # Save images
    save_rgb(rgb_cropped, root_path / "rgb_cropped.png")
    save_depth_colormap(depth_cropped, root_path / "depth_cropped.png")
    save_rgb(heatmap_vis_cropped, root_path / "heatmap_cropped.png")
    save_rgb(rendered_pcd, root_path / "pcd.png")
    save_rgb(rendered_grasps, root_path / "grasps.png")
    save_rgb(rendered_full, root_path / "full.png")
    save_rgb(rgbd_data.rgb, root_path / "rgb.png")
    save_depth_colormap(rgbd_data.depth, root_path / "depth.png")
    save_rgb(heatmap_vis, root_path / "heatmap.png")
    save_rgb(poses_vis, root_path / "poses.png")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgdf-model", type=str, default="9vkd9370")
    parser.add_argument("--idx", default=22, type=int)
    args = parser.parse_args()
    main(**vars(args))
