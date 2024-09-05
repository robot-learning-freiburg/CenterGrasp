import argparse
import trimesh
import itertools
import numpy as np
import matplotlib.pyplot as plt
from centergrasp import set_seeds
import centergrasp.data_utils as data_utils
from centergrasp.cameras import CameraConventions
from centergrasp.visualization import RerunViewer
from centergrasp.sapien.sapien_utils import Obs, CameraObs
from centergrasp.giga_utils import get_giga_cam_pose_opencv
from centergrasp.pipelines.giga_pipeline import GigaPipeline
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


def main(
    rgb_model: str,
    seed: int,
    headless: bool = True,
):
    if not headless:
        RerunViewer()
    chosen_ideces = {
        "giga_packed": 0,
        "giga_pile": 3,
        "ycb_packed": 0,
        "ycb_pile": 0,
        "real_packed": 2,
        "real_pile": 6,
    }
    real_robot = False
    source_path = Directories.DATA / "centergrasp_g" / "rgbd_evals"
    save_path = Directories.DATA / "centergrasp_g" / "plot_shape"
    save_path.mkdir(parents=True, exist_ok=True)
    set_seeds(seed)
    centergrasp_pl = CenterGraspPipeline(rgb_model, seed, visualize=not headless, camera_params=ZED2HALF_PARAMS)
    giga_pl = GigaPipeline("packed", seed, visualize=not headless, real_robot=real_robot)

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
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, 0, 2])
    pyrender_scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02])
    pyrender_scene.add(light, pose=light_pose)
    cam_node = pyrender_scene.add(camera)

    w_crop = 200
    h_crop = 75
    rgb_list = []
    depth_list = []
    giga_shape_list = []
    centergrasp_shape_list = []
    for env_name in chosen_ideces.keys():
        if real_robot != "real" in env_name:
            real_robot = "real" in env_name
            giga_pl = GigaPipeline("packed", seed, visualize=not headless, real_robot=real_robot)
        rgb_path = source_path / env_name / "rgb" / f"{chosen_ideces[env_name]:08d}.png"
        rgb = data_utils.load_rgb_from_file(rgb_path)
        depth_path = source_path / env_name / "depth" / f"{chosen_ideces[env_name]:08d}.png"
        depth = data_utils.load_depth_from_file(depth_path)
        if "real" in env_name:
            cam_pose = CAM_POSE_REAL
        else:
            cam_pose = get_giga_cam_pose_opencv()
        obs = Obs(CameraObs(rgb, depth_real=depth[..., np.newaxis]), cam_pose)
        _, cg_predicted_pc_list = centergrasp_pl.predict_shape(obs)
        _, giga_predicted_mesh = giga_pl.predict_shape(obs)

        # Render
        cam_pose = obs.camera_pose.A @ CameraConventions.opencv_T_opengl
        pyrender_scene.set_pose(cam_node, cam_pose)
        # Giga
        giga_pyrender_mesh = pyrender.Mesh.from_trimesh(giga_predicted_mesh, smooth=False)
        giga_mesh_node = pyrender_scene.add(giga_pyrender_mesh)
        rendered_giga_shape, _ = renderer.render(pyrender_scene)
        pyrender_scene.remove_node(giga_mesh_node)
        # CenterGrasp
        cg_pcd_nodes = [
            pyrender_scene.add(get_pcd_pyrender(pc, i)) for i, pc in enumerate(cg_predicted_pc_list)
        ]
        rendered_cg_shape, _ = renderer.render(pyrender_scene)
        for pc in cg_pcd_nodes:
            pyrender_scene.remove_node(pc)

        rgb_list.append(obs.camera.rgb[h_crop:-h_crop, w_crop:-w_crop])
        depth_list.append(obs.camera.depth[h_crop:-h_crop, w_crop:-w_crop])
        giga_shape_list.append(rendered_giga_shape[h_crop:-h_crop, w_crop:-w_crop])
        centergrasp_shape_list.append(rendered_cg_shape[h_crop:-h_crop, w_crop:-w_crop])
        # Save images
        data_utils.save_rgb(rgb_list[-1], save_path / f"rgb_{env_name}.png")
        data_utils.save_depth_colormap(depth_list[-1], save_path / f"depth_{env_name}.png")
        data_utils.save_rgb(giga_shape_list[-1], save_path / f"giga_shape_{env_name}.png")
        data_utils.save_rgb(centergrasp_shape_list[-1], save_path / f"cg_shape_{env_name}.png")

    # Final Plot
    n_rows = 3
    n_cols = len(chosen_ideces.keys())
    aspect_ratio = (rgb_list[0].shape[1] * n_cols) / (rgb_list[0].shape[0] * n_rows)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(4 * aspect_ratio + 0.2, 4))
    for col in range(n_cols):
        ax[0, col].imshow(rgb_list[col])
        ax[1, col].imshow(giga_shape_list[col])
        ax[2, col].imshow(centergrasp_shape_list[col])
    for row, col in itertools.product(range(n_rows), range(n_cols)):
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
        for spine in ax[row, col].spines.values():
            spine.set_visible(False)
    plt.subplots_adjust(wspace=0.02, hspace=0)
    ax[0, 0].set_ylabel("RGB \nObservation")
    ax[1, 0].set_ylabel("GIGA \nShape")
    ax[2, 0].set_ylabel("CenterGrasp\n Shape")
    ax[2, 0].set_xlabel("Giga Objects Packed")
    ax[2, 1].set_xlabel("Giga Objects Pile")
    ax[2, 2].set_xlabel("YCB Objects Packed")
    ax[2, 3].set_xlabel("YCB Objects Pile")
    ax[2, 4].set_xlabel("Real Objects Packed")
    ax[2, 5].set_xlabel("Real Objects Pile")
    plt.savefig(save_path / "shape_plot.png", dpi=400, bbox_inches="tight")
    plt.savefig(save_path / "shape_plot.pdf", dpi=400, bbox_inches="tight")
    # plt.show()
    print("Done")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-model", default="12c7ven5", type=str)
    parser.add_argument("--seed", default=123, type=int)
    args = parser.parse_args()
    main(**vars(args))
