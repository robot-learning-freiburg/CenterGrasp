import os
import pathlib
import numpy as np
from typing import Tuple

# To work on headless machine
os.environ["PYOPENGL_PLATFORM"] = "egl"
try:
    from mesh_to_sdf import sample_sdf_near_surface  # noqa: E402
except Exception as e:
    print(e)
from centergrasp.mesh_utils import load_mesh_trimesh  # noqa: E402


def generate_sdf(
    mesh_fpath: pathlib.Path, scale: float, number_of_points: int = 250000
) -> Tuple[np.ndarray, np.ndarray]:
    trimesh_obj = load_mesh_trimesh(mesh_fpath, scale)
    points, sdf = sample_sdf_near_surface(trimesh_obj, number_of_points, transform_back=True)
    points = points.astype("float64")
    sdf = sdf.astype("float64")  # type: ignore
    return points, sdf
