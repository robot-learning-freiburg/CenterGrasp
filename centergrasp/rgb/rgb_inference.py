import torch
import numpy as np
from typing import List, Tuple
import centergrasp.rgb.heatmaps as heatmaps
import centergrasp.rgb.pose_utils as pose_utils
import centergrasp.sgdf.sgdf_inference as sgdf_inference
from centergrasp.rgb.data_structures import ObjPredictionTh, FullObjPred
from centergrasp.rgb.training_centergrasp import load_rgb_model
from centergrasp.configs import DEVICE


def extract_obj_predictions(
    heatmap_th: torch.Tensor, posemap_th: torch.Tensor, shapemap_th: torch.Tensor
) -> List[ObjPredictionTh]:
    heatmap_np = heatmap_th.detach().cpu().numpy()
    peaks = heatmaps.extract_peaks_from_heatmap(heatmap_np)
    binary_masks = heatmaps.binary_masks_from_heatmap(heatmap_np, peaks, threshold=0.25)
    obj_predictions = []
    for i, peak in enumerate(peaks):
        peak_downsampled = peak // 8
        pose = posemap_th[:, peak_downsampled[0], peak_downsampled[1]]
        pose = pose_utils.pose_flat_to_matrix_th(pose)
        shape = shapemap_th[:, peak_downsampled[0], peak_downsampled[1]]
        binary_mask = binary_masks[i]
        obj_predictions.append(ObjPredictionTh(pose, shape, binary_mask))
    return obj_predictions


class RGBInference:
    def __init__(self, rgb_model: str):
        self.lit_rgb_model, rgb_specs = load_rgb_model(rgb_model)
        self.sgdf_inference_net = sgdf_inference.SGDFInference(rgb_specs["EmbeddingCkptPath"])
        self.lit_rgb_model.eval()
        return

    def np_to_torch(self, rgb_uint8_np: np.ndarray, depth_np: np.ndarray) -> Tuple[torch.Tensor]:
        rgb_np = rgb_uint8_np.astype(np.float32) / 255
        rgb_np = rgb_np.transpose((2, 0, 1))
        depth_np = depth_np.transpose((2, 0, 1))
        rgb_th = torch.from_numpy(rgb_np).to(DEVICE).unsqueeze(0)
        depth_th = torch.from_numpy(depth_np).to(DEVICE).unsqueeze(0)
        return rgb_th, depth_th

    def predict(
        self, rgb: torch.Tensor, depth: torch.Tensor
    ) -> Tuple[torch.Tensor, List[ObjPredictionTh]]:
        with torch.no_grad():
            heatmap_out, abs_pose_out, latent_emb_out = self.lit_rgb_model(rgb, depth)
            heatmap_out = heatmap_out.squeeze(0)
            abs_pose_out = abs_pose_out.squeeze(0)
            latent_emb_out = latent_emb_out.squeeze(0)
            obj_predictions = extract_obj_predictions(heatmap_out, abs_pose_out, latent_emb_out)
        return heatmap_out, obj_predictions

    def get_full_predictions(
        self, rgb_uint8_np: np.ndarray, depth_np: np.ndarray
    ) -> Tuple[np.ndarray, List[FullObjPred]]:
        rgb_th, depth_th = self.np_to_torch(rgb_uint8_np, depth_np)
        heatmap_out, obj_predictions = self.predict(rgb_th, depth_th)
        sgdf_preds = [
            self.sgdf_inference_net.predict_reconstruction(pred.embedding)
            for pred in obj_predictions
        ]
        full_preds = [
            FullObjPred.from_net_predictions(rgb_pred, sgdf_pred)
            for rgb_pred, sgdf_pred in zip(obj_predictions, sgdf_preds)
        ]
        heatmap_np = heatmap_out.detach().cpu().numpy()
        return heatmap_np, full_preds
