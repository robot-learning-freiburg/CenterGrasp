import os
import pathlib
import trimesh
import argparse
import numpy as np
import multiprocessing
from tqdm import tqdm
from typing import Tuple
from graspnetAPI import GraspNet
from graspnetAPI.utils.utils import generate_views
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from sklearn.model_selection import train_test_split
from centergrasp.configs import Directories
from centergrasp.sgdf.sgdf_data import SgdfData
import centergrasp.sgdf.sgdf_utils as sgdf_utils

# To work on headless machine
os.environ["PYOPENGL_PLATFORM"] = "egl"
try:
    from mesh_to_sdf import sample_sdf_near_surface  # noqa: E402
except Exception as e:
    print(e)


GRASPNET = GraspNet(root=Directories.GRASPNET, split="custom")
TOT_NUM_OBJECTS = 88


def load_trimesh_obj(obj_idx: int) -> trimesh.Trimesh:
    return GRASPNET.loadObjTrimesh(obj_idx)[0]


def graspnet_to_panda(rotations: np.ndarray) -> np.ndarray:
    """Converts GraspNet rotations (N, 3, 3) to panda_hand rotation."""
    rotations_panda = np.zeros_like(rotations)
    rotations_panda[..., 0] = -rotations[..., 2]
    rotations_panda[..., 1] = rotations[..., 1]
    rotations_panda[..., 2] = rotations[..., 0]
    return rotations_panda


def generate_sdf(obj_idx: int, number_of_points: int = 250000) -> Tuple[np.ndarray, np.ndarray]:
    trimesh_obj = load_trimesh_obj(obj_idx)
    points, sdf = sample_sdf_near_surface(trimesh_obj, number_of_points, transform_back=True)
    points = points.astype("float64")
    sdf = sdf.astype("float64")  # type: ignore
    return points, sdf


def load_graspnet_grasps(obj_idx: int, fric_coef_thresh=0.4) -> np.ndarray:
    num_views, num_angles, num_depths = 300, 12, 4
    template_views = generate_views(num_views)
    template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
    template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])

    sampled_points, offsets, fric_coefs = GRASPNET.loadGraspLabels(obj_idx)[obj_idx]
    point_inds = np.arange(sampled_points.shape[0])
    num_points = len(point_inds)
    target_points = sampled_points[:, np.newaxis, np.newaxis, np.newaxis, :]
    target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])
    views = np.tile(template_views, [num_points, 1, 1, 1, 1])
    angles = offsets[:, :, :, :, 0]
    depths = offsets[:, :, :, :, 1]
    widths = offsets[:, :, :, :, 2]

    mask1 = (fric_coefs <= fric_coef_thresh) & (fric_coefs > 0)
    target_points = target_points[mask1]  # (N, 3)
    views = views[mask1]
    angles = angles[mask1]
    depths = depths[mask1]
    widths = widths[mask1]
    fric_coefs = fric_coefs[mask1]

    rot = batch_viewpoint_params_to_matrix(-views, angles)  # (N, 3, 3)
    rot_panda = graspnet_to_panda(rot)
    grasp_poses_data = np.concatenate(
        (rot_panda, target_points[..., np.newaxis]), axis=-1
    )  # (N, 3, 4)
    scores = 1.1 - fric_coefs
    return grasp_poses_data, scores


def read_poses_data(obj_idx: int, frame: str) -> np.ndarray:
    assert frame in ["hand", "palm"]
    grasp_data, _ = load_graspnet_grasps(obj_idx)
    homog_part = np.zeros((len(grasp_data), 1, 4))
    homog_part[..., -1] = 1
    poses_data = np.concatenate((grasp_data, homog_part), axis=1)
    if frame == "palm":
        poses_data[..., :3, 3] -= 0.02 * poses_data[..., :3, 2]
    elif frame == "hand":
        poses_data[..., :3, 3] -= 0.0824 * poses_data[..., :3, 2]
    return poses_data


def get_sgdf(obj_idx: int) -> Tuple[SgdfData, SgdfData]:
    # SDF
    xyz_points, sdf = generate_sdf(obj_idx, number_of_points=100000)

    # GDF
    successful_grasps = read_poses_data(obj_idx, frame="palm")
    if len(successful_grasps) == 0:
        successful_grasps = np.eye(4).reshape(1, 4, 4)
    v_vecs = np.array(sgdf_utils.generate_gt_v(xyz_points, successful_grasps)).reshape(-1, 15)

    # Split
    points_train, points_valid, sdf_train, sdf_valid, v_vecs_train, v_vecs_valid = train_test_split(
        xyz_points, sdf, v_vecs, test_size=0.15, random_state=42
    )

    sgdf_train = SgdfData(points_train, sdf_train, v_vecs_train)  # type: ignore
    sgdf_valid = SgdfData(points_valid, sdf_valid, v_vecs_valid)  # type: ignore
    return sgdf_train, sgdf_valid


def get_fpath_from_obj_idx(obj_idx: int, mode: str) -> pathlib.Path:
    assert mode in ["train", "valid"]
    fpath = Directories.SGDF_GRASPNET / mode / f"{obj_idx:03d}.npy"
    return fpath


def check_file_exists(obj_idx: int) -> bool:
    train_fpath = get_fpath_from_obj_idx(obj_idx, "train")
    valid_fpath = get_fpath_from_obj_idx(obj_idx, "valid")
    return train_fpath.exists() and valid_fpath.exists()


def write_sgdf_data(obj_idx: int, mode: str, sgdf_data: SgdfData):
    fpath = get_fpath_from_obj_idx(obj_idx, mode)
    # Vector format: [px, py, pz, sdf, v_ax, v_ay, v_az, v_bx, v_by, v_bz, v_cx, v_cy, v_cz, v_dx, v_dy, v_dz, v_ex, v_ey, v_ez]  # noqa: E501
    vec = np.hstack((sgdf_data.xyz_points, sgdf_data.sdf[..., np.newaxis], sgdf_data.v_vecs))
    np.save(fpath, vec, allow_pickle=False)
    return


def save_sgdf_data(obj_idx: int):
    if check_file_exists(obj_idx):
        return
    sgdf_train_data, sgdf_valid_data = get_sgdf(obj_idx)
    write_sgdf_data(obj_idx, "train", sgdf_train_data)
    write_sgdf_data(obj_idx, "valid", sgdf_valid_data)
    return


def main(num_workers: int):
    for mode in ["train", "valid"]:
        (Directories.SGDF_GRASPNET / mode).mkdir(parents=True, exist_ok=True)
    # Use single worker for debugging
    if num_workers < 2:
        for obj_idx in tqdm(range(TOT_NUM_OBJECTS)):
            save_sgdf_data(obj_idx)
        return

    # Use multiple workers otherwise
    # multiprocessing.set_start_method("forkserver")  # Required for open3d
    with multiprocessing.Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(save_sgdf_data, list(range(TOT_NUM_OBJECTS))), total=TOT_NUM_OBJECTS
        ):
            pass
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    main(**vars(args))
