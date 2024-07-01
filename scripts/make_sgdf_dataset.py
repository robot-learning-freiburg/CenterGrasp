import pathlib
import itertools
import argparse
from tqdm import tqdm
import multiprocessing
import centergrasp.sgdf.sgdf_data as sgdf_data
from centergrasp.configs import Directories
from centergrasp.giga_utils import MeshPathsLoader


def save_sgdf_data(mesh_path: pathlib.Path):
    for scale in tqdm([0.7, 0.85, 1.0]):
        print(f"{mesh_path} ({scale})")
        if sgdf_data.check_exists(mesh_path, scale):
            continue
        sgdf_train_data, sgdf_valid_data = sgdf_data.get_sgdf(mesh_path, scale)
        sgdf_data.write_sgdf_data(mesh_path, "train", sgdf_train_data, scale)
        sgdf_data.write_sgdf_data(mesh_path, "valid", sgdf_valid_data, scale)
    return


def main(num_workers: int):
    for mode, group in itertools.product(["train", "valid"], ["packed", "pile"]):
        (Directories.SGDF / mode / group).mkdir(parents=True, exist_ok=True)
    full_mesh_paths = MeshPathsLoader.get_list("all")

    # Use single worker for debugging
    if num_workers < 2:
        for a in tqdm(full_mesh_paths):
            save_sgdf_data(a)
        return

    # Use multiple workers otherwise
    multiprocessing.set_start_method("forkserver")  # Required for open3d
    with multiprocessing.Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(save_sgdf_data, full_mesh_paths), total=len(full_mesh_paths)
        ):
            pass
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    main(**vars(args))
