import pathlib
import random
import numpy as np
from typing import List, Tuple
from typing import Optional
from dataclasses import dataclass
from centergrasp.configs import Directories
from sklearn.model_selection import train_test_split
from centergrasp.sgdf.sdf import generate_sdf
import centergrasp.sgdf.mesh_grasps as mesh_grasps
import centergrasp.sgdf.sgdf_utils as sgdf_utils
from centergrasp.giga_utils import MeshPathsLoader


@dataclass
class SgdfData:
    xyz_points: np.ndarray
    sdf: np.ndarray
    v_vecs: np.ndarray


class SgdfPathsLoader:
    def __init__(self, mode: str = "train", num_obj: Optional[int] = None) -> None:
        assert mode in ["train", "valid"]
        self.file_paths = self.get_list(mode, num_obj)
        self.fpath_to_idx = {fpath: idx for idx, fpath in enumerate(self.file_paths)}
        self.mode = mode
        return

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return self.file_paths[idx]

    @staticmethod
    def get_list(mode: str, num_obj: Optional[int] = None) -> List[pathlib.Path]:
        assert mode in ["train", "valid"]
        full_mesh_paths = MeshPathsLoader.get_list("all")
        if num_obj is not None:
            full_mesh_paths = full_mesh_paths[:num_obj]
        file_paths = [
            get_fpath_from_meshpath(p, mode, scale)
            for p in full_mesh_paths
            for scale in [0.7, 0.85, 1.0]
        ]
        return file_paths

    def get_random(self) -> pathlib.Path:
        return random.choice(self.file_paths)

    def get_idx(self, fpath: pathlib.Path) -> int:
        return self.fpath_to_idx[fpath]

    def get_idx_from_meshpath(self, mesh_path: pathlib.Path, scale: float) -> int:
        fpath = get_fpath_from_meshpath(mesh_path, self.mode, scale)
        return self.get_idx(fpath)


def get_fpath_from_meshpath(mesh_path: pathlib.Path, mode: str, scale: float) -> pathlib.Path:
    assert mode in ["train", "valid"]
    scale_flag = f"{int(scale*100):02d}"
    obj_name = mesh_path.stem
    group = mesh_path.parents[1].stem
    fpath = Directories.SGDF / mode / group / f"{obj_name}_{scale_flag}.npy"
    return fpath


def check_exists(mesh_path: pathlib.Path, scale: float) -> bool:
    train_fpath = get_fpath_from_meshpath(mesh_path, "train", scale)
    valid_fpath = get_fpath_from_meshpath(mesh_path, "valid", scale)
    return train_fpath.exists() and valid_fpath.exists()


def write_sgdf_data(mesh_path: pathlib.Path, mode: str, sgdf_data: SgdfData, scale: float):
    fpath = get_fpath_from_meshpath(mesh_path, mode, scale)
    # Vector format: [px, py, pz, sdf, v_ax, v_ay, v_az, v_bx, v_by, v_bz, v_cx, v_cy, v_cz, v_dx, v_dy, v_dz, v_ex, v_ey, v_ez]  # noqa: E501
    vec = np.hstack((sgdf_data.xyz_points, sgdf_data.sdf[..., np.newaxis], sgdf_data.v_vecs))
    np.save(fpath, vec, allow_pickle=False)
    return


def read_sgdf_from_meshpath(mesh_path: pathlib.Path, mode: str, scale: float) -> SgdfData:
    fpath = get_fpath_from_meshpath(mesh_path, mode, scale)
    return read_sgdf_data(fpath)


def read_sgdf_data(fpath: pathlib.Path) -> SgdfData:
    # Vector format: [px, py, pz, sdf, v_ax, v_ay, v_az, v_bx, v_by, v_bz, v_cx, v_cy, v_cz, v_dx, v_dy, v_dz, v_ex, v_ey, v_ez]  # noqa: E501
    vec = np.load(fpath, allow_pickle=False)
    xyz_points = vec[:, :3]
    sdf = vec[:, 3]
    v_vecs = vec[:, 4:]
    return SgdfData(xyz_points, sdf, v_vecs)


def get_sgdf(mesh_path: pathlib.Path, scale: float) -> Tuple[SgdfData, SgdfData]:
    # SDF
    xyz_points, sdf = generate_sdf(mesh_path, scale, number_of_points=100000)

    # GDF
    successful_grasps = mesh_grasps.read_poses_data(mesh_path, scale, frame="palm")
    if len(successful_grasps) == 0:
        successful_grasps = np.eye(4).reshape(1, 4, 4)
    v_vecs = np.array(sgdf_utils.generate_gt_v(xyz_points, successful_grasps)).reshape(-1, 15)

    # Split
    points_train, points_valid, sdf_train, sdf_valid, v_vecs_train, v_vecs_valid = train_test_split(
        xyz_points, sdf, v_vecs, test_size=0.15, random_state=42
    )

    sgdf_train = SgdfData(points_train, sdf_train, v_vecs_train)  # type: ignore
    sgdf_valid = SgdfData(points_valid, sdf_valid, v_vecs_valid)  # type: ignore
    return sgdf_train, sgdf_valid
