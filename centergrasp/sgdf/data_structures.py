import torch
import numpy as np
import open3d as o3d
from dataclasses import dataclass
import centergrasp.data_utils as data_utils


@dataclass
class SGDFPrediction:
    pc_o3d: o3d.geometry.PointCloud
    grasp_poses: np.ndarray
    grasp_confidences: np.ndarray

    @classmethod
    def from_torch(cls, pc_th: torch.Tensor, grasps_th: torch.Tensor, confs_th: torch.Tensor):
        pc, grasps, confs = data_utils.th_to_np(pc_th, grasps_th, confs_th)
        pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
        # Transform from palm to hand frame
        grasps[..., :3, 3] -= 0.0624 * grasps[..., :3, 2]
        return cls(pc_o3d, grasps, confs)
