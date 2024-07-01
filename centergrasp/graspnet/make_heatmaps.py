import os
import pathlib
import multiprocessing
from tqdm import tqdm
from graspnetAPI import GraspNet
from centergrasp.configs import Directories
import centergrasp.data_utils as data_utils
from centergrasp.rgb.heatmaps import heatmap_from_segmentation
from centergrasp.graspnet.rgb_data import get_obj_poses

GRASPNET = GraspNet(root=Directories.GRASPNET, camera="kinect", split="train")


def make_heatmap(i: int):
    paths = GRASPNET.loadData(i)
    segLabelPath = paths[2]
    poses_path = segLabelPath.replace("label", "annotations").replace(".png", ".xml")
    bmask_path = pathlib.Path(
        (segLabelPath)
        .replace(str(Directories.GRASPNET), str(Directories.RGBD_GRASPNET))
        .replace("label", "bmask")
        .replace(".png", ".json")
    )
    heatmap_path = pathlib.Path(
        (segLabelPath)
        .replace(str(Directories.GRASPNET), str(Directories.RGBD_GRASPNET))
        .replace("label", "heatmap")
    )
    if bmask_path.exists() and heatmap_path.exists():
        print(f"Skipping {i}, it already exists")
        return
    bmask_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    segm = data_utils.load_rgb_from_file(segLabelPath)
    obj_indices, _ = get_obj_poses(poses_path)
    shifted_indices = obj_indices + 1  # Shift by 1, since 0 is background
    heatmap, simple_bmasks = heatmap_from_segmentation(segm, shifted_indices)
    data_utils.save_binary_masks(simple_bmasks, bmask_path)
    data_utils.save_rgb(heatmap, heatmap_path)


def main():
    num_images = len(GRASPNET)
    num_workers = os.cpu_count() // 2
    with multiprocessing.Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(make_heatmap, range(num_images)), total=num_images):
            pass
    return


if __name__ == "__main__":
    main()
    # make_heatmap(2412)
