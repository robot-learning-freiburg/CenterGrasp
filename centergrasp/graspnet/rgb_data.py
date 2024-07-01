import torch
import pathlib
import numpy as np
from typing import Tuple, List
from graspnetAPI import GraspNet
from graspnetAPI.utils.utils import parse_posevector
from graspnetAPI.utils.xmlhandler import xmlReader
import centergrasp.se3_utils as se3_utils
import centergrasp.data_utils as data_utils
from centergrasp.cameras import CameraParams
from centergrasp.configs import Directories, DEVICE
from centergrasp.rgb.data_structures import RgbdDataNp
from centergrasp.sgdf.training_deep_sgdf import load_embeddings

KINECT_K = np.array(
    [[631.54864502, 0.0, 638.43517329], [0.0, 631.20751953, 366.49904066], [0.0, 0.0, 1.0]]
)
KINECT_PARAMS = CameraParams(width=1280, height=720, K=KINECT_K)
KINECT_HALF_PARAMS = KINECT_PARAMS.downsampled(2).cropped(8, 0)


def get_obj_poses(pose_fpath: str) -> Tuple[np.ndarray]:
    scene_reader = xmlReader(pose_fpath)
    posevector = scene_reader.getposevectorlist()
    obj_idx, poses = zip(*[parse_posevector(vec) for vec in posevector])
    obj_idx = np.array(obj_idx)
    poses = np.array(poses)
    # Sort by obj_idx, to align with the saved binary masks
    ind_argsort = np.argsort(obj_idx)
    obj_idx = obj_idx[ind_argsort]
    poses = poses[ind_argsort]
    return obj_idx, poses


class RGBDReader:
    def __init__(self, mode: str = "train") -> None:
        self.camera = "kinect"
        self.graspnet_api = GraspNet(root=Directories.GRASPNET, camera=self.camera, split=mode)
        return

    def __len__(self):
        return len(self.graspnet_api)

    def get_scene_img_names(self, idx: int) -> Tuple[str]:
        rgbPath = pathlib.Path(self.graspnet_api.loadData(idx)[0])
        scene_name = rgbPath.parents[2].name
        img_name = rgbPath.stem
        return scene_name, img_name

    def get_data_np(self, idx: int, half: bool = True) -> RgbdDataNp:
        paths = self.graspnet_api.loadData(idx)
        rgbPath = paths[0]
        depthPath = paths[1]
        segLabelPath = paths[2]
        poses_path = rgbPath.replace("/rgb/", "/annotations/").replace(".png", ".xml")
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

        rgb = data_utils.load_rgb_from_file(rgbPath)
        depth = data_utils.load_depth_from_file(depthPath)
        obj_indices, poses = get_obj_poses(poses_path)
        binary_masks = data_utils.load_binary_masks(bmask_path)
        heatmap = data_utils.load_rgb_from_file(heatmap_path)
        if half:
            # Downsample
            rgb = np.ascontiguousarray(rgb[::2, ::2, :])
            depth = depth[::2, ::2]
            binary_masks = binary_masks[:, ::2, ::2]
            heatmap = heatmap[::2, ::2]
            # Crop
            rgb = rgb[4:-4, ...]
            depth = depth[4:-4, ...]
            binary_masks = binary_masks[:, 4:-4, :]
            heatmap = heatmap[4:-4, :]
        return RgbdDataNp(rgb, depth, heatmap, binary_masks, poses, obj_indices)


class RGBDatasetGraspnet(torch.utils.data.Dataset):  # type: ignore
    def __init__(self, embedding_ckpt: str, mode="train"):
        if mode not in ["train", "test"]:
            raise ValueError("mode must be either 'train' or 'test'")
        self.mode = mode
        self.rgbd_reader = RGBDReader(mode)
        # Needs to be on CPU for dataloading
        self.embeddings_matrix = load_embeddings(embedding_ckpt).cpu()
        self.code_length = self.embeddings_matrix.shape[1]
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
        self, poses: np.ndarray, binary_masks: np.ndarray, info: List[int]
    ) -> Tuple[np.ndarray]:
        height = binary_masks.shape[1]
        width = binary_masks.shape[2]
        pose_target = np.zeros((height, width, 12), dtype=np.float32)
        invariance_map = np.zeros((height, width), dtype=np.uint8)
        shape_target = np.zeros((height, width, self.code_length), dtype=np.float32)
        for i, obj_idx in enumerate(info):
            pose = poses[i]
            pose_flat = se3_utils.pose_4x4_to_flat(pose)
            shape = self.embeddings_matrix[obj_idx]
            invariance = 0  # TODO
            pose_target[binary_masks[i] == 1] = pose_flat
            shape_target[binary_masks[i] == 1] = shape
            invariance_map[binary_masks[i] == 1] = invariance
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
    dataset = RGBDatasetGraspnet(embedding_ckpt="6953cfxt")
    data = dataset[1]
