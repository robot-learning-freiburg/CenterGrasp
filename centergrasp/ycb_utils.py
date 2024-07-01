import json
import gzip
import random
import pathlib
import numpy as np
from typing import Union
from centergrasp.configs import Directories
from centergrasp.mesh_utils import SceneObject


def load_json(filename: Union[str, pathlib.Path]):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "rt")
    elif filename.endswith(".json"):
        f = open(filename, "rt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    ret = json.loads(f.read())
    f.close()
    return ret


class YCBPathsLoader:
    MODEL_JSON = Directories.YCB / "mani_skill2_ycb/info_pick_v0.json"
    HEAVY_OBJECTS = ["006_mustard_bottle"]

    def __init__(self) -> None:
        if not self.MODEL_JSON.exists():
            raise FileNotFoundError(
                f"json file ({self.MODEL_JSON}) is not found."
                "To download default json:"
                "`python -m mani_skill2.utils.download_asset pick_clutter_ycb`."
            )
        self.model_db: dict[str, dict] = load_json(self.MODEL_JSON)
        self.mesh_names = sorted(list(self.model_db.keys()))
        self.mesh_paths = [
            Directories.YCB / f"mani_skill2_ycb/models/{n}/textured.obj" for n in self.mesh_names
        ]

    def __len__(self):
        return len(self.model_db)

    def get_random(self) -> pathlib.Path:
        return random.choice(self.mesh_paths)

    def meshpath_to_sceneobj(
        self, meshpath: pathlib.Path, pose: np.ndarray = np.eye(4)
    ) -> SceneObject:
        name = meshpath.parent.stem
        visual_path = str(meshpath)
        collision_path = visual_path.replace("textured.obj", "collision.obj")
        scale = self.model_db[name].get("scales", 1) * np.ones(3)
        density = self.model_db[name].get("density", 1000)
        if name in self.HEAVY_OBJECTS:
            density /= 3
        return SceneObject(
            pathlib.Path(visual_path), pathlib.Path(collision_path), pose, scale, density, name
        )
