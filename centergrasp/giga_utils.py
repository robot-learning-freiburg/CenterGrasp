import pathlib
import random
import numpy as np
from typing import List
import spatialmath as sm
from dataclasses import dataclass
from centergrasp.cameras import look_at_x
from centergrasp.configs import Directories
from centergrasp.cameras import CameraConventions
from centergrasp.mesh_utils import SceneObject, get_volume_from_mesh, get_mass_from_urdf

# Giga camera params
GIGA_CAM_WIDTH = 640
GIGA_CAM_HEIGHT = 480
GIGA_CAM_K = np.array([[540.0, 0.0, 320.0], [0.0, 540.0, 240.0], [0.0, 0.0, 1.0]])


@dataclass
class GigaParams:
    ws_origin = np.array([0.3, -0.15, 0.0])
    ws_size = 0.3
    ws_resolution = 40
    voxel_size = ws_size / ws_resolution
    ws_center = ws_origin + np.array([ws_size / 2, ws_size / 2, ws_size / 2])
    ws_size_array = np.array([ws_size, ws_size, ws_size])


def get_giga_cam_pose() -> np.ndarray:
    origin = GigaParams.ws_origin + np.array([GigaParams.ws_size / 1.5, GigaParams.ws_size / 2, 0])
    theta = np.pi / 4.0
    phi = -np.pi / 2.0
    radius = 2.0 * GigaParams.ws_size
    eye = (
        origin
        + np.r_[
            radius * np.cos(theta) * np.cos(phi),
            radius * np.cos(theta) * np.sin(phi),
            radius * np.sin(theta),
        ]
    )
    cam_pose = look_at_x(eye=eye, target=origin)
    return cam_pose


def get_giga_cam_pose_opencv() -> sm.SE3:
    transform = sm.SE3(CameraConventions.robotics_T_opencv, check=False)
    cam_pose = sm.SE3(get_giga_cam_pose(), check=False) * transform
    return cam_pose


class MeshPathsLoader:
    HEAVY_OBJECTS = ["jar_poisson_004_visual", "lemon_poisson_000_visual"]

    def __init__(self, group: str = "all", mode: str = "train") -> None:
        self.mesh_paths_list = self.get_list(group, mode)
        self.mesh_paths_map = {}
        for path in self.mesh_paths_list:
            obj_name = path.stem
            group = path.parents[1].stem
            self.mesh_paths_map[(obj_name, group)] = path
        return

    def __len__(self):
        return len(self.mesh_paths_list)

    @staticmethod
    def get_list(group: str, mode: str = "train") -> List[pathlib.Path]:
        assert group in ["packed", "pile", "all"]
        assert mode in ["train", "test"]
        if group == "all":
            return MeshPathsLoader.get_list("packed") + MeshPathsLoader.get_list("pile")
        dir = Directories.GIGA_REPO / f"data/urdfs/{group}/{mode}/"
        mesh_paths = [f for f in dir.iterdir() if f.is_file() and f.name.endswith("_visual.obj")]
        mesh_paths.sort()
        if len(mesh_paths) == 0:
            raise Exception("The provided path does not contain any mesh files")
        return mesh_paths

    def get_from_name(self, obj_name: str, group: str) -> pathlib.Path:
        assert group in ["packed", "pile"]
        return self.mesh_paths_map[(obj_name, group)]

    def get_random(self) -> pathlib.Path:
        return random.choice(self.mesh_paths_list)

    def meshpath_to_sceneobj(
        self, meshpath: pathlib.Path, pose: np.ndarray = np.eye(4)
    ) -> SceneObject:
        visual_path = str(meshpath)
        collision_path = visual_path.replace("visual", "collision")
        urdf_path = visual_path.replace("_visual.obj", ".urdf")
        scale = np.random.uniform(0.7, 1.0) * np.ones(3)
        mass = get_mass_from_urdf(urdf_path)
        volume = get_volume_from_mesh(collision_path)
        density = mass / volume if mass is not None else 1000.0
        name = meshpath.stem
        if name in self.HEAVY_OBJECTS:
            density /= 3
        return SceneObject(
            pathlib.Path(visual_path), pathlib.Path(collision_path), pose, scale, density, name
        )


class GigaScenesLoader:
    PACKED_TRAIN_RAW = Directories.GIGA / "data_packed_train_raw"
    PILE_TRAIN_RAW = Directories.GIGA / "data_pile_train_raw"
    PACKED_TRAIN_PROC = Directories.GIGA / "data_packed_train_processed_dex_noise"
    PILE_TRAIN_PROC = Directories.GIGA / "data_pile_train_processed_dex_noise"

    def __init__(self) -> None:
        self.packed_scene_ids = [
            p.stem for p in sorted((self.PACKED_TRAIN_PROC / "scenes").iterdir())
        ]
        self.pile_scene_ids = [p.stem for p in sorted((self.PILE_TRAIN_PROC / "scenes").iterdir())]
        pass

    def __len__(self):
        return self.num_all_scenes

    @property
    def num_packed_scenes(self) -> int:
        return len(self.packed_scene_ids)

    @property
    def num_pile_scenes(self) -> int:
        return len(self.pile_scene_ids)

    @property
    def num_all_scenes(self) -> int:
        return self.num_packed_scenes + self.num_pile_scenes

    def _rounded_scale(self, scale: float, options=[0.7, 0.85, 1.0]):
        return min(options, key=lambda x: abs(x - scale))

    def mesh_data_to_sceneobj(self, mesh_data: np.ndarray) -> SceneObject:
        visual_path: str = mesh_data[0]
        scale: np.ndarray = self._rounded_scale(scale=mesh_data[1]) * np.ones(3)
        pose4x4: np.ndarray = mesh_data[2]
        visual_fpath = Directories.GIGA_REPO / visual_path
        collision_fpath = Directories.GIGA_REPO / visual_path.replace("visual", "collision")
        urdf_path = Directories.GIGA_REPO / visual_path.replace("_visual.obj", ".urdf")
        name = urdf_path.stem
        mass = get_mass_from_urdf(urdf_path)
        volume = get_volume_from_mesh(collision_fpath)
        density = mass / volume if mass is not None else 1000.0
        return SceneObject(visual_fpath, collision_fpath, pose4x4, scale, density, name)

    def load_packed_objs(self, scene_id: str) -> List[SceneObject]:
        path = self.PACKED_TRAIN_RAW / "mesh_pose_list" / f"{scene_id}.npz"
        mesh_pose_list = np.load(path, allow_pickle=True)["pc"]
        objs = [self.mesh_data_to_sceneobj(x) for x in mesh_pose_list]
        return objs

    def load_packed_objs_idx(self, idx: int) -> List[SceneObject]:
        scene_id = self.packed_scene_ids[idx]
        return self.load_packed_objs(scene_id)

    def load_pile_objs(self, scene_id: str) -> List[SceneObject]:
        path = self.PILE_TRAIN_RAW / "mesh_pose_list" / f"{scene_id}.npz"
        mesh_pose_list = np.load(path, allow_pickle=True)["pc"]
        objs = [self.mesh_data_to_sceneobj(x) for x in mesh_pose_list]
        return objs

    def load_pile_objs_idx(self, idx: int) -> List[SceneObject]:
        scene_id = self.pile_scene_ids[idx]
        return self.load_pile_objs(scene_id)

    def load_all_objs_idx(self, idx: int) -> List[SceneObject]:
        if idx < self.num_packed_scenes:
            return self.load_packed_objs_idx(idx)
        else:
            return self.load_pile_objs_idx(idx - self.num_packed_scenes)


if __name__ == "__main__":
    mesh_loader = MeshPathsLoader(mode="train")
    print(f"There are {len(mesh_loader)} meshes in the dataset")
    scene_loader = GigaScenesLoader()
    print(f"There are {len(scene_loader)} scenes in the dataset")
    print("ready")
