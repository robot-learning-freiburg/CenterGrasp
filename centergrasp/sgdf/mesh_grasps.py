import pathlib
import numpy as np
import centergrasp.mesh_utils as mesh_utils
from centergrasp.configs import Directories
from centergrasp.franka_gripper import FrankaGripperO3d
from centergrasp.o3d_live_vis import O3dLiveVisualizer


def get_fpath_from_mesh_path(mesh_path: pathlib.Path, kind: str, scale: float) -> pathlib.Path:
    assert kind in ["poses", "scores"]
    assert scale in [0.7, 0.85, 1.0]
    scale_flag = f"{int(scale*100):02d}"
    obj_name = mesh_path.stem
    group = mesh_path.parents[1].stem
    assert group in ["packed", "pile"]
    fpath = Directories.GRASPS / kind / group / f"{obj_name}_{scale_flag}.npy"
    return fpath


def check_exists(mesh_path: pathlib.Path, scale: float) -> bool:
    poses_fpath = get_fpath_from_mesh_path(mesh_path, "poses", scale)
    scores_fpath = get_fpath_from_mesh_path(mesh_path, "scores", scale)
    return poses_fpath.exists() and scores_fpath.exists()


def write_poses_data(mesh_path: pathlib.Path, poses: np.ndarray, scale: float):
    fpath = get_fpath_from_mesh_path(mesh_path, "poses", scale)
    smaller_data = poses[:, :3, :] if len(poses) > 0 else np.array([])
    np.save(fpath, smaller_data, allow_pickle=False)
    return


def write_scores_data(mesh_path: pathlib.Path, scores: np.ndarray, scale: float):
    fpath = get_fpath_from_mesh_path(mesh_path, "scores", scale)
    np.save(fpath, scores, allow_pickle=False)


def read_poses_data(mesh_path: pathlib.Path, scale: float, frame: str) -> np.ndarray:
    assert frame in ["hand", "palm", "ttip"]
    fpath = get_fpath_from_mesh_path(mesh_path, "poses", scale)
    data = np.load(fpath, allow_pickle=False)
    if len(data) == 0:
        return data
    homog_part = np.zeros((len(data), 1, 4))
    homog_part[..., -1] = 1
    poses_data = np.concatenate((data, homog_part), axis=1)
    if frame == "palm":
        poses_data[..., :3, 3] -= 0.041 * poses_data[..., :3, 2]
    elif frame == "hand":
        poses_data[..., :3, 3] -= 0.1034 * poses_data[..., :3, 2]
    return poses_data


def read_scores_data(mesh_path: pathlib.Path, scale: float) -> np.ndarray:
    fpath = get_fpath_from_mesh_path(mesh_path, "scores", scale)
    data = np.load(fpath, allow_pickle=False)
    return data


# TODO: add load with threshold for score


def generate_grasps(mesh_path: pathlib.Path, scale: float, vis_flag: bool = False):
    """
    Saves grasp poses expressed as the ttip pose in the obj frame.
    """
    vis = O3dLiveVisualizer(vis_flag)

    # Get object mesh
    obj_mesh = mesh_utils.load_mesh_o3d(mesh_path, scale, mm2m=False)
    vis.add_and_render(obj_mesh)

    # Get PointCloud
    pc = obj_mesh.sample_points_poisson_disk(number_of_points=1000, init_factor=5)
    pc.estimate_normals()
    pc.normalize_normals()

    # Gripper Model
    gripper_mesh = FrankaGripperO3d(vis)

    # Search Grid
    point_indeces = np.arange(len(pc.points))
    angle_step = 15  # degrees

    out_grasps = []
    out_scores = []
    # Iterate over widths
    gripper_mesh.set_gripper_width(0.08)

    # Iterate over points
    for pc_idx in point_indeces:
        point, normal = pc.points[pc_idx], pc.normals[pc_idx]
        gripper_mesh.align_to_surface(point, normal)
        in_gripper_points, in_gripper_normals = mesh_utils.points_inside_mesh(
            gripper_mesh.meshes["cylinder_mesh"], pc
        )
        gripper_mesh.center_gripper(in_gripper_points)

        # Calculate Score
        gr_normals = np.array([gripper_mesh.vector_w_to_grasp(n) for n in in_gripper_normals])
        score = np.mean([n[1] ** 2 for n in gr_normals])
        if score < 0.5 or np.isnan(score) or score is None:
            continue
        score = np.clip(score, a_min=0.001, a_max=0.999)

        # Iterate over rotations
        for _ in range(360 // angle_step):
            gripper_mesh.step_rotation(angle_step)

            # Check collision
            collision = mesh_utils.is_colliding(
                meshes=[
                    gripper_mesh.meshes["hand"],
                    gripper_mesh.meshes["left_finger"],
                    gripper_mesh.meshes["right_finger"],
                ],
                pc=pc,
            )
            if collision:
                continue
            out_grasps.append(gripper_mesh.wTgrasp.A)
            out_scores.append(score)
    return np.array(out_grasps), np.array(out_scores)


if __name__ == "__main__":
    mesh_path = Directories.GIGA_REPO / "data/urdfs/packed/train/MelforBottle_800_tex_visual.obj"
    out = generate_grasps(mesh_path, scale=1.0, vis_flag=True)
