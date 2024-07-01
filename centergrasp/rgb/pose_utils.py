import torch
import roma
import numpy as np
from typing import List
import cv2


from centergrasp import cameras
from centergrasp.rgb.heatmaps import extract_peaks_from_heatmap
from simnet.lib.transform import Pose
from simnet.lib.net.post_processing import pose_outputs


class MaskedPoseLoss(torch.nn.Module):
    def __init__(self, centroid_threshold=0.25, downscale_factor=8):
        super().__init__()
        self.centroid_threshold = centroid_threshold
        self.downscale_factor = downscale_factor

    def forward(self, output, target, valid_mask, invariance_map):
        """
        Args:
            output: (B, 12, H, W)
            target: (B, 12, H, W)
            valid_mask: (B, H, W)
            invariance_map: (B, H, W)
        Returns:
            loss: (1,)
        """
        small_mask = valid_mask[:, :: self.downscale_factor, :: self.downscale_factor]
        valid_count = torch.sum(small_mask > self.centroid_threshold)
        loss = pose_image_loss(output, target, invariance_map)
        # weight the loss by the heatmap values
        loss = loss * small_mask
        # threshold the loss outside of the mask
        loss[small_mask <= self.centroid_threshold] = 0.0
        if valid_count == 0:
            return torch.sum(loss)
        return torch.sum(loss) / valid_count


def pose_image_loss(
    pose_image_out: torch.Tensor, pose_image_target: torch.Tensor, invariance_map: torch.Tensor
):
    """
    Args:
        pose_image_out: (B, 12, H, W)
        pose_image_target: (B, 12, H, W)
        invariance_map: (B, H, W)
    Returns:
        loss: (B, H, W)
    """
    pose_image_out = pose_image_out.permute(0, 2, 3, 1)
    pose_image_target = pose_image_target.permute(0, 2, 3, 1)
    pose_image_out_sq = pose_flat_to_matrix_th(pose_image_out)
    pose_image_target_sq = pose_flat_to_matrix_th(pose_image_target)
    loss_map = pose_loss(pose_image_out_sq, pose_image_target_sq, invariance_map)
    return loss_map


def pose_loss(output: torch.Tensor, target: torch.Tensor, invariance_map: torch.Tensor):
    """
    Args:
        output: (B, H, W, 4, 4) or (B, H, W, 3, 4)
        target: (B, H, W, 4, 4) or (B, H, W, 3, 4)
        invariance_map: (B, H, W)
    Returns:
        loss: (B, H, W)
    """
    unique_invariances = torch.unique(invariance_map)
    loss = torch.zeros_like(invariance_map, dtype=torch.float32)
    output_kp = pose_to_keypoints(output)
    # No invariance
    # Base pose
    target_0_kp = pose_to_keypoints(target)
    loss_0 = torch.linalg.vector_norm(target_0_kp - output_kp, dim=-1).mean(dim=-1)
    loss = torch.where(invariance_map == 0, loss_0, loss)
    # Rotation Z invariance
    if 1 in unique_invariances:
        # ignore x and y axes
        loss_0b = torch.linalg.vector_norm(target_0_kp - output_kp, dim=-1)[..., [0, 3]].mean(
            dim=-1
        )
        loss = torch.where(invariance_map == 1, loss_0b, loss)
    # Plane XZ + YZ invariance
    if (2 in unique_invariances) or (3 in unique_invariances):
        # 180 around z
        target_1 = torch.clone(target)
        target_1[..., :3, 0] *= -1
        target_1[..., :3, 1] *= -1
        target_1_kp = pose_to_keypoints(target_1)
        loss_1 = torch.linalg.vector_norm(target_1_kp - output_kp, dim=-1).mean(dim=-1)
        loss = torch.where(invariance_map == 2, torch.minimum(loss_0, loss_1), loss)
    # Origin invariance
    if 3 in unique_invariances:
        # 180 around x
        target_2 = torch.clone(target)
        target_2[..., :3, 1] *= -1
        target_2[..., :3, 2] *= -1
        target_2_kp = pose_to_keypoints(target_2)
        loss_2 = torch.linalg.vector_norm(target_2_kp - output_kp, dim=-1).mean(dim=-1)
        # 180 around y
        target_3 = torch.clone(target)
        target_3[..., :3, 0] *= -1
        target_3[..., :3, 2] *= -1
        target_3_kp = pose_to_keypoints(target_3)
        loss_3 = torch.linalg.vector_norm(target_3_kp - output_kp, dim=-1).mean(dim=-1)
        loss = torch.where(
            invariance_map == 3,
            torch.minimum(torch.minimum(loss_0, loss_1), torch.minimum(loss_2, loss_3)),
            loss,
        )
    return loss


def pose_to_keypoints(pose: torch.Tensor):
    """
    Args:
        pose: (..., 4, 4) or (..., 3, 4)
    Returns:
        keypoints: (..., 4, 3)
    """
    base_shape = pose.shape[:-2]
    rot_weight = torch.tensor([0.1], device=pose.device, dtype=pose.dtype)
    keypoints = torch.zeros((*base_shape, 4, 3), device=pose.device)
    keypoints[..., 0, :] = pose[..., :3, 3]
    keypoints[..., 1, :] = pose[..., :3, 3] + rot_weight * pose[..., :3, 0]
    keypoints[..., 2, :] = pose[..., :3, 3] + rot_weight * pose[..., :3, 1]
    keypoints[..., 3, :] = pose[..., :3, 3] + rot_weight * pose[..., :3, 2]
    return keypoints


def pose_image_procrustes(pose_image: torch.Tensor):
    """
    pose_image: (B, 12, H, W)
    """
    pose_image = pose_image.permute(0, 2, 3, 1)
    base_shape = pose_image.shape[:-1]
    trans_image = pose_image[:, :, :, 9:]
    rot_image = pose_image[:, :, :, :9].reshape(*base_shape, 3, 3)
    rot_image = roma.special_procrustes(rot_image)
    rot_image = rot_image.reshape(*base_shape, 9)
    pose_image = torch.cat((rot_image, trans_image), dim=-1)
    pose_image = pose_image.permute(0, 3, 1, 2)
    return pose_image


def pose_flat_procrustes(pose_flat: torch.Tensor):
    pose_matrix = pose_flat_to_matrix_th(pose_flat)
    pose_matrix = roma.special_procrustes(pose_matrix)
    pose_flat = pose_matrix_to_flat_th(pose_matrix)
    return pose_flat


def pose_matrix_to_flat_th(pose_matrix: torch.Tensor):
    pose_flat = torch.cat(
        (
            pose_matrix[:3, :3].reshape(-1),
            pose_matrix[:3, 3],
        )
    )
    return pose_flat


def pose_matrix_to_flat_np(pose_matrix: np.ndarray):
    pose_flat = np.concatenate(
        (
            pose_matrix[:3, :3].reshape(-1),
            pose_matrix[:3, 3],
        )
    )
    return pose_flat


def pose_flat_to_matrix_th(pose_flat: torch.Tensor):
    """
    Args:
        pose_flat: (..., 12)
    Returns:
        pose_matrix: (..., 4, 4)
    """
    base_shape = pose_flat.shape[:-1]
    pose_matrix = torch.eye(4).to(pose_flat.device).expand((*base_shape, 4, 4)).clone()
    # Rotation
    pose_matrix[..., :3, :3] = pose_flat[..., :9].reshape(*base_shape, 3, 3)
    # Translation
    pose_matrix[..., :3, 3] = pose_flat[..., 9:]
    return pose_matrix


def pose_flat_to_matrix_np(pose_flat: np.ndarray):
    pose_matrix = np.eye(4)
    # Rotation
    pose_matrix[:3, :3] = pose_flat[:9].reshape((3, 3))
    # Translation
    pose_matrix[:3, 3] = pose_flat[9:]
    return pose_matrix


def poses_to_numpy(poses_map):
    poses_np = poses_map.detach().cpu().numpy()
    return poses_np


def get_poses(heatmap, posemap):
    # TODO Make HPs accessible from outside?
    peaks = extract_peaks_from_heatmap(heatmap)
    # Transpose posemap
    # TODO Check? Should this be (2, 1, 0) since the peaks are (y, x)
    poses = pose_outputs.extract_abs_pose_from_peaks(np.copy(peaks), posemap.transpose((1, 2, 0)))
    return poses


def overlay_pose(rgb_img, pose: np.ndarray, K: np.ndarray, scale: float = 0.1):
    # TODO Check that they follow the correct notation
    # BGR; x: Red, y: Green, z: Blue
    COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    root_camera = pose[:, 3]
    root_image = cameras.project(np.expand_dims(root_camera, -1), K)

    directions = np.concatenate(
        (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) * scale, np.ones((3, 1))),
        axis=1,
    )
    directions_camera = pose @ directions.T
    directions_image = cameras.project(directions_camera, K)

    # Add roots
    # cv2.circle(
    #     rgb_img, center=(int(root_image[0]), int(root_image[1])), radius=5, color=(0, 0, 255)
    # )
    for idx in range(3):
        direction = directions_image[:, idx]
        rgb_img = cv2.arrowedLine(
            rgb_img,
            pt1=(int(root_image[0]), int(root_image[1])),
            pt2=(int(direction[0]), int(direction[1])),
            color=COLORS[idx],
            thickness=3,
        )
    return rgb_img


def visualize_poses(rgb_img, poses: List[Pose], K: np.ndarray):
    super_imposed_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    for pose in poses:
        super_imposed_img = overlay_pose(super_imposed_img, pose.camera_T_object, K)
    rgb_out = cv2.cvtColor(super_imposed_img, cv2.COLOR_BGR2RGB)
    return rgb_out
