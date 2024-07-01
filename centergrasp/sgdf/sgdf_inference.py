import torch
import pathlib
import numpy as np
from typing import Tuple
import centergrasp.data_utils as data_utils
from centergrasp.configs import DEVICE
from centergrasp.sgdf.grid import Grid3D
import centergrasp.sgdf.sgdf_data as sgdf_data
from centergrasp.sgdf.data_structures import SGDFPrediction
from centergrasp.sgdf.training_deep_sgdf import load_sgdf_model


class SGDFInference:
    def __init__(self, sgdf_model: str):
        self.lit_model, sgdf_specs = load_sgdf_model(sgdf_model)
        self.lit_model.eval()
        self.sgdf_specs = sgdf_specs
        return

    def predict_reconstruction(
        self, embeddings: torch.Tensor, grid_density: int = 64, grid_half_dim: float = 0.2
    ) -> SGDFPrediction:
        grid_3d = Grid3D(density=grid_density, grid_dim=grid_half_dim, device=str(DEVICE))
        sdf_th, grasp_poses_th, confidence_th = self.predict(grid_3d.points, embeddings)
        pointcloud_th = grid_3d.get_masked_surface_iso(sdf_th, threshold=0.001)
        sgdf_prediction = SGDFPrediction.from_torch(pointcloud_th, grasp_poses_th, confidence_th)
        return sgdf_prediction

    def predict(
        self, points: torch.Tensor, shape_embedding: torch.Tensor, distance_threshold: float = 0.005
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Note: Using no grad here saves memory but disallows us to us the iso-surface projection
        with torch.no_grad():
            if len(points.shape) == 2:
                points = points.unsqueeze(0)
            if len(shape_embedding.shape) == 1:
                shape_embedding = shape_embedding.unsqueeze(0)
            embeddings_expanded = shape_embedding.unsqueeze(1).expand(-1, points.shape[1], -1)
            sdf, grasp_poses = self.lit_model.predict(points, embeddings_expanded)
            distance = torch.linalg.vector_norm(grasp_poses[:, :3, 3] - points.squeeze(0), dim=-1)

            # Filter out grasps that are too far away
            indeces = torch.nonzero(distance < distance_threshold).squeeze(-1)
            distance = distance[indeces]
            grasp_poses = grasp_poses[indeces]

            # The confidence is the negative distance
            confidence = -distance
        return sdf, grasp_poses, confidence

    def predict_th2np(
        self, points: torch.Tensor, shape_embedding: torch.Tensor, distance_threshold: float = 0.005
    ) -> Tuple[np.ndarray]:
        sdf, grasp_poses, confidence = self.predict(points, shape_embedding, distance_threshold)
        return data_utils.th_to_np(sdf, grasp_poses, confidence)

    def predict_np2th(
        self, points: np.ndarray, shape_embedding_np: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = torch.from_numpy(points, dtype=torch.float32)
        points = points.to(DEVICE).unsqueeze(0)
        shape_embedding = torch.from_numpy(shape_embedding_np).to(DEVICE).unsqueeze(0)
        sdf, grasp_poses, confidence = self.predict(points, shape_embedding)
        return sdf, grasp_poses, confidence

    def predict_np2np(
        self, points: np.ndarray, shape_embedding_np: np.ndarray
    ) -> Tuple[np.ndarray]:
        sdf, grasp_poses, confidence = self.predict_np2th(points, shape_embedding_np)
        return data_utils.th_to_np(sdf, grasp_poses, confidence)

    def get_embeddings_np(self) -> np.ndarray:
        embeddings_np = (
            self.lit_model.embeddings.weight.detach().cpu().numpy()
        )  # Number of Objects x Code Dimensions
        return embeddings_np


class SGDFInferenceGT(SGDFInference):
    def __init__(self, **kwargs):
        self.sgdf_path_loader = sgdf_data.SgdfPathsLoader()
        super().__init__(**kwargs)
        return

    def get_embedding_from_fpath(self, fpath: pathlib.Path) -> torch.Tensor:
        idx = self.sgdf_path_loader.get_idx(fpath)
        idx_th = torch.tensor(idx).to(DEVICE)
        shape_embedding = self.lit_model.embeddings(idx_th)
        return shape_embedding.unsqueeze(0)

    def get_embedding_from_meshpath(self, mesh_path: pathlib.Path, scale: float) -> torch.Tensor:
        idx = self.sgdf_path_loader.get_idx_from_meshpath(mesh_path, scale)
        idx_th = torch.tensor(idx).to(DEVICE)
        shape_embedding = self.lit_model.embeddings(idx_th)
        return shape_embedding.unsqueeze(0)

    def predict_from_meshpath(self, mesh_path: pathlib.Path, scale: float) -> SGDFPrediction:
        shape_embedding = self.get_embedding_from_meshpath(mesh_path, scale)
        return self.predict_reconstruction(shape_embedding)
