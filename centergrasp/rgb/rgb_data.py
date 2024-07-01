import json
import torch
import pathlib
import numpy as np
from typing import List, Tuple
import centergrasp.se3_utils as se3_utils
import centergrasp.data_utils as data_utils
from centergrasp.cameras import CameraConventions
from centergrasp.sapien.scenes import GTData
from centergrasp.configs import Directories, DEVICE
from centergrasp.sgdf.sgdf_data import SgdfPathsLoader
from centergrasp.rgb.data_structures import RgbdPaths, RgbdPathsSingle, RgbdDataNp
from centergrasp.sgdf.training_deep_sgdf import load_embeddings


def get_rgbd_paths(mode: str, scene_idx: int, num_cams: int) -> RgbdPaths:
    root_path = Directories.RGBD / mode
    file_names = [f"{scene_idx:08d}_{cam_idx:04d}" for cam_idx in range(num_cams)]
    rgbd_paths = RgbdPaths(
        rgb=[root_path / "rgb" / f"{file_names[idx]}.png" for idx in range(num_cams)],
        depth=[root_path / "depth" / f"{file_names[idx]}.png" for idx in range(num_cams)],
        poses=[root_path / "pose" / f"{file_names[idx]}.npy" for idx in range(num_cams)],
        binary_masks=[root_path / "segm" / f"{file_names[idx]}_bm.json" for idx in range(num_cams)],
        heatmap=[root_path / "segm" / f"{file_names[idx]}_heatmap.png" for idx in range(num_cams)],
        info=root_path / "segm" / f"{scene_idx:08d}_info.json",
    )
    return rgbd_paths


def check_exists(rgbd_paths: RgbdPaths) -> bool:
    rgb = all([p.exists() for p in rgbd_paths.rgb])
    depth = all([p.exists() for p in rgbd_paths.depth])
    poses = all([p.exists() for p in rgbd_paths.poses])
    binary_masks = all([p.exists() for p in rgbd_paths.binary_masks])
    heatmap = all([p.exists() for p in rgbd_paths.heatmap])
    info = rgbd_paths.info.exists()
    return rgb and depth and poses and binary_masks and heatmap and info


def write_rgbd_data(rgbd_paths: RgbdPaths, gt_data_list: List[GTData], objs_info: List[dict]):
    data_utils.save_dict_as_json(objs_info, rgbd_paths.info)
    for cam_idx, gt_data in enumerate(gt_data_list):
        data_utils.save_rgb(gt_data.rgb, rgbd_paths.rgb[cam_idx])
        data_utils.save_depth(gt_data.depth, rgbd_paths.depth[cam_idx])
        np.save(rgbd_paths.poses[cam_idx], gt_data.camTposes, allow_pickle=False)
        data_utils.save_binary_masks(gt_data.binary_masks, rgbd_paths.binary_masks[cam_idx])
        data_utils.save_rgb(gt_data.heatmap, rgbd_paths.heatmap[cam_idx])
    return


class RGBDReader:
    def __init__(self, mode: str = "train") -> None:
        self.rgb_paths = self.get_rgb_paths(mode)
        self.rgb_paths_to_idx = {rgb_path: idx for idx, rgb_path in enumerate(self.rgb_paths)}
        return

    @staticmethod
    def get_rgb_paths(mode: str) -> List[pathlib.Path]:
        assert mode in ["train", "valid"]
        rgb_paths = sorted((Directories.RGBD / mode / "rgb").iterdir())
        return rgb_paths

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx: int) -> RgbdPathsSingle:
        rgb_path = self.rgb_paths[idx]
        scene_idx = int(rgb_path.stem.split("_")[0])
        info_path = rgb_path.parents[1] / "segm" / f"{scene_idx:08d}_info.json"
        paths = RgbdPathsSingle(
            rgb=rgb_path,
            depth=pathlib.Path(str(rgb_path).replace("/rgb/", "/depth/")),
            poses=pathlib.Path(str(rgb_path).replace("/rgb/", "/pose/").replace(".png", ".npy")),
            binary_masks=pathlib.Path(
                str(rgb_path).replace("/rgb/", "/segm/").replace(".png", "_bm.json")
            ),
            heatmap=pathlib.Path(
                str(rgb_path).replace("/rgb/", "/segm/").replace(".png", "_heatmap.png")
            ),
            info=info_path,
        )
        return paths

    def get_data_np(self, idx: int) -> RgbdDataNp:
        paths = self[idx]
        rgb = data_utils.load_rgb_from_file(paths.rgb)
        depth = data_utils.load_depth_from_file(paths.depth)
        poses = np.load(paths.poses, allow_pickle=False)
        binary_masks = data_utils.load_binary_masks(paths.binary_masks)
        heatmap = data_utils.load_rgb_from_file(paths.heatmap)
        info = json.load(open(paths.info))
        # Transform obj poses from sapien camera frame to opencv camera frame
        poses = CameraConventions.opencv_T_robotics @ poses
        return RgbdDataNp(rgb, depth, heatmap, binary_masks, poses, info)

    def get_random(self) -> RgbdPathsSingle:
        return self[np.random.randint(len(self))]

    def get_idx_from_path(self, rgb_path: pathlib.Path) -> int:
        return self.rgb_paths_to_idx[rgb_path]


class RGBDReaderReal:
    def __init__(self):
        self.rgb_paths = self.get_rgb_paths()
        return

    @staticmethod
    def get_rgb_paths() -> List[pathlib.Path]:
        rgb_paths = sorted((Directories.RGBD / "real" / "rgb").iterdir())
        return rgb_paths

    def __len__(self):
        return len(self.rgb_paths)

    def get_data_np(self, idx: int) -> RgbdDataNp:
        rgb_path = self.rgb_paths[idx]
        depth_path = pathlib.Path(str(rgb_path).replace("/rgb/", "/depth/"))
        rgb = data_utils.load_rgb_from_file(rgb_path)
        depth = data_utils.load_depth_from_file(depth_path)
        return RgbdDataNp(rgb, depth, None, None, None, None)


class RGBDataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(self, embedding_ckpt: str, mode="train"):
        if mode not in ["train", "valid"]:
            raise ValueError("mode must be either 'train' or 'valid'")
        self.mode = mode
        self.sgdf_paths_loader = SgdfPathsLoader(mode)
        self.rgbd_reader = RGBDReader(mode)
        # Needs to be on CPU for dataloading
        self.embeddings_matrix = load_embeddings(embedding_ckpt).cpu()
        self.code_length = self.embeddings_matrix.shape[1]
        # self.obj_index_inv_dict = get_obj_idx_invariance_dict()
        # self.obj_rotation_dict = get_obj_rotation_dict()
        return

    def __len__(self):
        return len(self.rgbd_reader)

    def __getitem__(self, idx):
        data_np = self.rgbd_reader.get_data_np(idx)
        pose_target, shape_target, invariance_map = self.make_targets(
            data_np.poses, data_np.binary_masks, data_np.info
        )
        # Torch shape order: [channels, height, width]
        rgb = data_utils.img_np_to_torch(data_np.rgb)
        depth = torch.from_numpy(data_np.depth).unsqueeze(0)
        heatmap_target = data_utils.img_np_to_torch(data_np.heatmap)
        pose_target = torch.from_numpy(pose_target).permute(2, 0, 1)
        shape_target = torch.from_numpy(shape_target).permute(2, 0, 1)
        invariance_map = torch.from_numpy(invariance_map)
        return rgb, depth, heatmap_target, pose_target, shape_target, invariance_map

    def make_targets(
        self, poses: np.ndarray, binary_masks: np.ndarray, info: List[dict]
    ) -> Tuple[np.ndarray]:
        height = binary_masks.shape[1]
        width = binary_masks.shape[2]
        pose_target = np.zeros((height, width, 12), dtype=np.float32)
        invariance_map = np.zeros((height, width), dtype=np.uint8)
        shape_target = np.zeros((height, width, self.code_length), dtype=np.float32)
        for obj_idx, obj_info in enumerate(info):
            sgdf_idx = self.sgdf_paths_loader.get_idx_from_meshpath(
                pathlib.Path(obj_info["mesh_path"]), obj_info["scale"][0]
            )
            pose = poses[obj_idx]
            pose_flat = se3_utils.pose_4x4_to_flat(pose)
            shape = self.embeddings_matrix[sgdf_idx]
            invariance = 0  # TODO
            pose_target[binary_masks[obj_idx] == 1] = pose_flat
            shape_target[binary_masks[obj_idx] == 1] = shape
            invariance_map[binary_masks[obj_idx] == 1] = invariance
        # downsample
        pose_target = pose_target[::8, ::8, :]
        shape_target = shape_target[::8, ::8, :]
        invariance_map = invariance_map[::8, ::8]
        return pose_target, shape_target, invariance_map

    def get_data(self, idx, to_device):
        rgb, depth, heatmap_target, pose_target, shape_target, invariance_map = self[idx]
        if to_device:
            rgb = data_utils.make_single_batch(rgb, 3).to(DEVICE)
            depth = data_utils.make_single_batch(depth, 1).to(DEVICE)
            heatmap_target = heatmap_target.to(DEVICE)
            pose_target = pose_target.to(DEVICE)
            shape_target = shape_target.to(DEVICE)
            invariance_map = invariance_map.to(DEVICE)
        return rgb, depth, heatmap_target, pose_target, shape_target, invariance_map


if __name__ == "__main__":
    dataset = RGBDataset("9vkd9370")
    data = dataset[1]
