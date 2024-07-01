import pathlib
import torch
import numpy as np
import open3d as o3d
from typing import List
from dataclasses import dataclass
import centergrasp.data_utils as data_utils
from centergrasp.sgdf.data_structures import SGDFPrediction


@dataclass
class ObjPredictionTh:
    pose: torch.Tensor
    embedding: torch.Tensor
    bmask: np.ndarray


@dataclass
class FullObjPred:
    pose: np.ndarray
    bmask: np.ndarray
    pc_o3d: o3d.geometry.PointCloud
    grasp_poses: np.ndarray
    grasp_confidences: np.ndarray

    @classmethod
    def from_net_predictions(cls, rgb_pred: ObjPredictionTh, sgdf_pred: SGDFPrediction):
        pose = rgb_pred.pose.detach().cpu().numpy()
        bmask = rgb_pred.bmask
        pc_o3d = sgdf_pred.pc_o3d.transform(pose)
        grasp_poses = pose @ sgdf_pred.grasp_poses
        grasp_confidences = sgdf_pred.grasp_confidences
        return cls(pose, bmask, pc_o3d, grasp_poses, grasp_confidences)


@dataclass
class PostprObjPred:
    pc_o3d: o3d.geometry.PointCloud
    grasp_poses: np.ndarray


@dataclass
class RgbdPaths:
    rgb: List[pathlib.Path]
    depth: List[pathlib.Path]
    binary_masks: List[pathlib.Path]
    heatmap: List[pathlib.Path]
    poses: List[pathlib.Path]
    info: pathlib.Path


@dataclass
class RgbdPathsSingle:
    rgb: pathlib.Path
    depth: pathlib.Path
    binary_masks: pathlib.Path
    heatmap: pathlib.Path
    poses: pathlib.Path
    info: pathlib.Path


@dataclass
class RgbdDataNp:
    rgb: np.ndarray
    depth: np.ndarray
    heatmap: np.ndarray
    binary_masks: np.ndarray
    poses: np.ndarray
    info: List[dict]

    @classmethod
    def from_torch(cls, rgb_th: torch.Tensor, depth_th: torch.Tensor, heatmap_th: torch.Tensor):
        rgb = data_utils.img_torch_to_np(rgb_th)
        depth = depth_th.squeeze(0).detach().cpu().numpy()
        heatmap = data_utils.img_torch_to_np(heatmap_th)
        return cls(rgb, depth, heatmap, None, None, None)
