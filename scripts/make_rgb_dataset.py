import argparse
import functools
from tqdm import tqdm
import multiprocessing
from typing import Optional
from multiprocessing.synchronize import Lock as LockBase
from centergrasp.configs import Directories
from centergrasp.giga_utils import GigaScenesLoader
from centergrasp.sapien.scenes import SceneImgRenderer
from centergrasp.rgb.rgb_data import write_rgbd_data, get_rgbd_paths, check_exists


def split_range_chunks(range_: range, num_chunks: int) -> list[range]:
    """Yield successive n-sized chunks from range."""
    step = len(range_) / num_chunks
    subranges = [range(round(step * i), round(step * (i + 1))) for i in range(num_chunks)]
    return subranges


def make_dataset(
    scenes_range: range,
    scenes_loader: GigaScenesLoader,
    headless: bool,
    raytracing: bool,
    mode: str,
    imgs_per_scene: int,
    lock: Optional[LockBase] = None,
):
    if mode == "valid":
        scenes_range = range(scenes_range.start, scenes_range.stop, 100)
        imgs_per_scene = 1
    scene_renderer = SceneImgRenderer(headless, raytracing, imgs_per_scene, lock)
    for scene_idx in tqdm(scenes_range):
        rgbd_paths = get_rgbd_paths(mode, scene_idx, imgs_per_scene)
        if check_exists(rgbd_paths):
            continue
        scene_objects = scenes_loader.load_all_objs_idx(scene_idx)
        gt_data_list, objs_info = scene_renderer.make_data(scene_objects)
        write_rgbd_data(rgbd_paths, gt_data_list, objs_info)
    return


def main(headless: bool, raytracing: bool, mode: str, imgs_per_scene: int, num_workers: int):
    (Directories.RGBD / mode / "rgb").mkdir(parents=True, exist_ok=True)
    (Directories.RGBD / mode / "depth").mkdir(parents=True, exist_ok=True)
    (Directories.RGBD / mode / "pose").mkdir(parents=True, exist_ok=True)
    (Directories.RGBD / mode / "segm").mkdir(parents=True, exist_ok=True)
    scenes_loader = GigaScenesLoader()
    scenes_range = range(scenes_loader.num_all_scenes)
    # Use single worker for debugging
    if num_workers < 2:
        make_dataset(scenes_range, scenes_loader, headless, raytracing, mode, imgs_per_scene)
        return

    # Use multiple workers otherwise
    multiprocessing.set_start_method("forkserver")  # Required for open3d
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    subranges = split_range_chunks(scenes_range, num_workers)
    func = functools.partial(
        make_dataset,
        scenes_loader=scenes_loader,
        headless=headless,
        raytracing=raytracing,
        mode=mode,
        imgs_per_scene=imgs_per_scene,
        lock=lock,
    )
    with multiprocessing.Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(func, subranges), total=len(subranges)):
            pass
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--raytracing", action="store_true")
    parser.add_argument("--imgs_per_scene", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    main(**vars(args))
