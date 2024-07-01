import pathlib
import torch
import numpy as np
from typing import Optional
import centergrasp.sgdf.sgdf_data as sgdf_data
from centergrasp.sgdf.sgdf_data import SgdfData, SgdfPathsLoader


def query_sgdf_samples(file_path: pathlib.Path, num_points: Optional[int] = None) -> SgdfData:
    sgdf = sgdf_data.read_sgdf_data(file_path)
    if num_points is not None:
        # Subsample points
        indices = np.random.choice(len(sgdf.sdf), size=num_points, replace=False)
        sgdf.xyz_points = sgdf.xyz_points[indices]
        sgdf.sdf = sgdf.sdf[indices]
        sgdf.v_vecs = sgdf.v_vecs[indices]
    return sgdf


class SGDFDataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(self, points_per_obj: int, mode: str = "train") -> None:
        super().__init__()
        if mode not in ["train", "valid"]:
            raise ValueError("mode must be either 'train' or 'valid'")
        self.file_paths = SgdfPathsLoader.get_list(mode)
        self.points_per_obj = points_per_obj
        # Since we only sample ~8% of the points
        self.len_scaling = 8 if mode == "train" else 1
        return

    def get_num_objects(self):
        return len(self.file_paths)

    def __len__(self):
        return self.get_num_objects() * self.len_scaling

    def __getitem__(self, idx):
        idx = idx // self.len_scaling
        file_path = self.file_paths[idx]
        sgdf = query_sgdf_samples(file_path, num_points=self.points_per_obj)
        points_th = torch.tensor(sgdf.xyz_points, dtype=torch.float32)
        sdfs_th = torch.tensor(sgdf.sdf, dtype=torch.float32)
        gdfs_th = torch.tensor(sgdf.v_vecs, dtype=torch.float32).reshape(-1, 5, 3)
        return idx, points_th, sdfs_th, gdfs_th


if __name__ == "__main__":
    dataset = SGDFDataset(10000)
    data = dataset[15]
