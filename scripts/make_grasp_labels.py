import pathlib
import argparse
import itertools
import multiprocessing
from tqdm import tqdm
from centergrasp.configs import Directories
from centergrasp.giga_utils import MeshPathsLoader
import centergrasp.sgdf.mesh_grasps as mesh_grasps


def save_grasps_labels(mesh_path: pathlib.Path):
    for scale in tqdm([0.7, 0.85, 1.0]):
        print(f"{mesh_path} ({scale})")
        if mesh_grasps.check_exists(mesh_path, scale):
            continue
        grasp_poses, scores = mesh_grasps.generate_grasps(mesh_path, scale)
        mesh_grasps.write_poses_data(mesh_path, grasp_poses, scale)
        mesh_grasps.write_scores_data(mesh_path, scores, scale)
    return


def main(num_workers: int):
    for kind, mode in itertools.product(["poses", "scores"], ["packed", "pile"]):
        (Directories.GRASPS / kind / mode).mkdir(parents=True, exist_ok=True)
    full_mesh_paths = MeshPathsLoader.get_list("all")

    # Use single worker for debugging
    if num_workers < 2:
        for mesh_path in tqdm(full_mesh_paths):
            save_grasps_labels(mesh_path)
        return

    # Use multiple workers otherwise
    multiprocessing.set_start_method("forkserver")  # Required for open3d
    with multiprocessing.Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(save_grasps_labels, full_mesh_paths), total=len(full_mesh_paths)
        ):
            pass
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    main(**vars(args))
