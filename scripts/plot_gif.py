import argparse
import trimesh
import numpy as np
from PIL import Image
import centergrasp.data_utils as data_utils
import centergrasp.rgb.heatmaps as heatmaps
from centergrasp.sapien.sapien_utils import Obs, CameraObs
from centergrasp.cameras import CameraConventions, _normalize, look_at_z
from centergrasp.rgb.rgb_data import RGBDReaderReal
from centergrasp.rgb.pred_postprocessing import get_full_pcd
from centergrasp.visualization import create_markers_multiple, create_markers_multiple_fat
from centergrasp.pipelines.centergrasp_pipeline import CenterGraspPipeline
from centergrasp.configs import ZED2HALF_PARAMS, Directories, COLOR_PALETTE, CAM_POSE_REAL
import pyrender


def get_pcd_pyrender(pcd_np, color_idx=0):
    sphere_trimesh = trimesh.creation.uv_sphere(radius=0.002)
    sphere_trimesh.visual.vertex_colors = np.array(
        [*COLOR_PALETTE[color_idx % len(COLOR_PALETTE)], 1.0]
    )
    tfs = np.tile(np.eye(4), (len(pcd_np), 1, 1))
    tfs[:, :3, 3] = pcd_np
    pcd_pyrender = pyrender.Mesh.from_trimesh(sphere_trimesh, poses=tfs)
    return pcd_pyrender


def main(rgb_model: str, idx: int):
    root_path = Directories.DATA / "centergrasp_g" / "plot_gif"
    root_path.mkdir(parents=True, exist_ok=True)
    data_loader = RGBDReaderReal()
    centergrasp_pipeline = CenterGraspPipeline(rgb_model, visualize=False, camera_params=ZED2HALF_PARAMS)
    cam_pose = CAM_POSE_REAL

    # Pyrender setup
    renderer = pyrender.OffscreenRenderer(ZED2HALF_PARAMS.width, ZED2HALF_PARAMS.height)
    camera = pyrender.IntrinsicsCamera(
        fx=ZED2HALF_PARAMS.f_xy,
        fy=ZED2HALF_PARAMS.f_xy,
        cx=ZED2HALF_PARAMS.cx,
        cy=ZED2HALF_PARAMS.cy,
        znear=0.01,
        zfar=10,
    )
    pyrender_scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=6.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, 0, 2])
    pyrender_scene.add(light, pose=light_pose)
    cam_node = pyrender_scene.add(camera)
    cam_pose_pr = cam_pose.A @ CameraConventions.opencv_T_opengl
    pyrender_scene.set_pose(cam_node, cam_pose_pr)

    # Make data
    rgbd_data = data_loader.get_data_np(idx)
    full_pcd = get_full_pcd(rgbd_data, ZED2HALF_PARAMS)
    full_pcd.transform(cam_pose.A)
    cam_obs = CameraObs(rgbd_data.rgb, depth_real=rgbd_data.depth[..., np.newaxis])
    obs = Obs(cam_obs, camera_pose=cam_pose)
    (
        _,
        pcs_list,
        wTeegoal_list_all,
        wTeegoal_list_best,
    ) = centergrasp_pipeline.predict_shape_and_grasps(obs)
    all_grasps_trimesh = create_markers_multiple(
        [g.A for g in wTeegoal_list_all], color=[0, 255, 0], axis_frame=False
    )
    best_grasps_trimesh = create_markers_multiple_fat(
        [g.A for g in wTeegoal_list_best], color=[0, 255, 0], axis_frame=False
    )
    pcs_list_np = [np.asarray(pc.points) for pc in pcs_list]
    heatmap, _ = centergrasp_pipeline.rgb_net.get_full_predictions(obs.camera.rgb, obs.camera.depth)
    heatmap_vis = heatmaps.visualize_heatmap(obs.camera.rgb, heatmap, with_peaks=True)

    # Convert to pyrender
    full_pcd_pyrender = pyrender.Mesh.from_points(
        np.asarray(full_pcd.points),
        colors=np.power(np.asarray(full_pcd.colors), 2.2),  # Gamma correction
    )
    pcds_pyrender = [get_pcd_pyrender(pcd, i) for i, pcd in enumerate(pcs_list_np)]
    grasps_pyrender = [
        pyrender.Mesh.from_trimesh(grasps, smooth=False) for grasps in all_grasps_trimesh
    ]
    best_grasps_pyrender = [
        pyrender.Mesh.from_trimesh(grasp, smooth=False) for grasp in best_grasps_trimesh
    ]

    scene_center = np.array([pcd.get_center() for pcd in pcs_list]).mean(axis=0)
    radius = np.linalg.norm(scene_center - cam_pose.t)
    v1 = _normalize(cam_pose.t - scene_center)
    v2 = _normalize(np.cross(v1, np.array([0, 0, 1])))
    param = np.linspace(-0.25, 0.25, 40)
    # Parametric Equation of a Circle: https://math.stackexchange.com/a/1184089

    camera_positions = [scene_center + radius * (np.cos(t) * v1 + np.sin(t) * v2) for t in param]
    camera_poses = [
        look_at_z(cam_position, scene_center) @ CameraConventions.opencv_T_opengl
        for cam_position in camera_positions
    ]

    # Render
    def render_gif(scene, camera_poses, filename):
        images = []
        for pose in camera_poses:
            scene.set_pose(cam_node, pose)
            rendered_image, _ = renderer.render(scene)
            images.append(Image.fromarray(rendered_image))
        # Append the same array last to first to make a backward loop
        images.extend(images[::-1])
        out_file = root_path / filename
        images[0].save(out_file, save_all=True, append_images=images[1:], duration=50, loop=0)
        scene.set_pose(cam_node, cam_pose_pr)
        return

    # Multi grasp shape
    mesh_nodes = [pyrender_scene.add(pcd) for pcd in pcds_pyrender]
    grasp_nodes = [pyrender_scene.add(grasp) for grasp in grasps_pyrender]
    rendered_multi_grasps, _ = renderer.render(pyrender_scene)
    best_grasps_trimesh_pil = Image.fromarray(rendered_multi_grasps)
    best_grasps_trimesh_pil.save(root_path / f"{idx:06d}_a.png")
    render_gif(pyrender_scene, camera_poses, f"{idx:06d}_a.gif")
    for grasp_node in grasp_nodes:
        pyrender_scene.remove_node(grasp_node)

    # Single grasp shape
    grasp_nodes = [pyrender_scene.add(grasp) for grasp in best_grasps_pyrender]
    rendered_best_grasps, _ = renderer.render(pyrender_scene)
    best_grasps_trimesh_pil = Image.fromarray(rendered_best_grasps)
    best_grasps_trimesh_pil.save(root_path / f"{idx:06d}_b.png")
    render_gif(pyrender_scene, camera_poses, f"{idx:06d}_b.gif")
    for grasp_node in grasp_nodes:
        pyrender_scene.remove_node(grasp_node)
    for mesh_node in mesh_nodes:
        pyrender_scene.remove_node(mesh_node)

    # Multi grasp full pcd
    full_pcd_node = pyrender_scene.add(full_pcd_pyrender)
    grasp_nodes = [pyrender_scene.add(grasp) for grasp in grasps_pyrender]
    rendered_multi_grasps, _ = renderer.render(pyrender_scene)
    rendered_multi_grasps_pil = Image.fromarray(rendered_multi_grasps)
    rendered_multi_grasps_pil.save(root_path / f"{idx:06d}_c.png")
    render_gif(pyrender_scene, camera_poses, f"{idx:06d}_c.gif")
    for grasp_node in grasp_nodes:
        pyrender_scene.remove_node(grasp_node)

    # Single grasp full pcd
    grasp_nodes = [pyrender_scene.add(grasp) for grasp in best_grasps_pyrender]
    rendered_best_grasps, _ = renderer.render(pyrender_scene)
    rendered_best_grasps_pil = Image.fromarray(rendered_best_grasps)
    rendered_best_grasps_pil.save(root_path / f"{idx:06d}_d.png")
    render_gif(pyrender_scene, camera_poses, f"{idx:06d}_d.gif")
    for grasp_node in grasp_nodes:
        pyrender_scene.remove_node(grasp_node)
    pyrender_scene.remove_node(full_pcd_node)

    data_utils.save_rgb(obs.camera.rgb, root_path / f"{idx:06d}_rgb.png")
    data_utils.save_rgb(heatmap_vis, root_path / f"{idx:06d}_heatmap.png")
    print("Done")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-model", type=str, default="12c7ven5")
    parser.add_argument("--idx", default=0, type=int)
    args = parser.parse_args()
    main(**vars(args))
