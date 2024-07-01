import pathlib
import trimesh
import numpy as np
import open3d as o3d
import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass
class SceneObject:
    visual_fpath: pathlib.Path
    collision_fpath: pathlib.Path
    pose4x4: np.ndarray = np.eye(4)
    scale: np.ndarray = np.ones(3)
    density: float = 1000.0
    name: str = ""


def load_mesh_o3d(path: pathlib.Path, scale: float, mm2m: bool) -> o3d.geometry.TriangleMesh:
    obj_mesh = o3d.io.read_triangle_mesh(str(path))
    obj_mesh.scale(scale, center=(0, 0, 0))
    if mm2m:
        obj_mesh.scale(0.001, center=(0, 0, 0))
    obj_mesh.compute_vertex_normals()
    return obj_mesh


def load_mesh_trimesh(path: pathlib.Path, scale: float) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh")
    mesh.apply_scale(scale)
    return mesh


def points_inside_mesh(meshes, pc):
    occupancy_values = compute_occupancy(meshes, pc).numpy()
    indeces = [i for i, elem in enumerate(occupancy_values) if elem == 1]
    return np.asarray(pc.points)[indeces], np.asarray(pc.normals)[indeces]


def is_colliding(meshes, pc):
    occupancy_values = compute_occupancy(meshes, pc).numpy()
    collision = any(occupancy_values == 1)
    return collision


def distance_to_scene(points, raycasting_scene):
    points = np.asarray(list(points), dtype=np.float32)
    distances = raycasting_scene.compute_distance(points).numpy()
    return min(distances)


def is_colliding_with_scene(raycasting_scene, pc):
    points = np.asarray(pc.points, dtype=np.float32)
    occupancy_values = raycasting_scene.compute_occupancy(points).numpy()
    collision = any(occupancy_values == 1)
    return collision


def compute_occupancy(meshes, pc):
    scene = compute_raycasting_scene(meshes)
    points = np.asarray(pc.points, dtype=np.float32)
    return scene.compute_occupancy(points)


def compute_raycasting_scene(meshes):
    scene = o3d.t.geometry.RaycastingScene()
    if isinstance(meshes, list):
        for mesh in meshes:
            t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            _ = scene.add_triangles(t_mesh)
    elif isinstance(meshes, dict):
        for mesh in meshes.values():
            t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            _ = scene.add_triangles(t_mesh)
    else:
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(meshes)
        _ = scene.add_triangles(t_mesh)
    return scene


def get_volume_from_mesh(mesh_path: pathlib.Path, scale: np.ndarray = np.ones(3)) -> float:
    mesh = trimesh.load(mesh_path)
    volume = np.prod(scale) * mesh.volume
    return volume


def get_mass_from_urdf(urdf_path: pathlib.Path) -> float:
    # Parse the URDF XML file
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    # Find the <inertial> element
    inertial_element = root.find(".//inertial")
    if inertial_element is not None:
        mass = float(inertial_element.find("mass").get("value"))
    else:
        mass = None
        print("Inertial element not found in the URDF XML file.")
    return mass


@dataclass
class AmbientCGTexture:
    path: pathlib.Path

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def color_fpath(self) -> pathlib.Path:
        path = self.path / f"{self.name}_2K_Color.jpg"
        return path if path.exists() else None

    @property
    def roughness_fpath(self) -> pathlib.Path:
        path = self.path / f"{self.name}_2K_Roughness.jpg"
        return path if path.exists() else None

    @property
    def metallic_fpath(self) -> pathlib.Path:
        path = self.path / f"{self.name}_2K_Metallic.jpg"
        return path if path.exists() else None

    @property
    def normal_fpath(self) -> pathlib.Path:
        path = self.path / f"{self.name}_2K_NormalGL.jpg"
        return path if path.exists() else None

    @property
    def displacement_fpath(self) -> pathlib.Path:
        path = self.path / f"{self.name}_2K_Displacement.jpg"
        return path if path.exists() else None

    @property
    def ambient_occlusion_fpath(self) -> pathlib.Path:
        path = self.path / f"{self.name}_2K_AmbientOcclusion.jpg"
        return path if path.exists() else None
