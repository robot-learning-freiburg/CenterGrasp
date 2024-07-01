from tqdm import tqdm
from centergrasp.configs import Directories
import centergrasp.data_utils as data_utils
from centergrasp.giga_utils import GigaScenesLoader

if __name__ == "__main__":
    scenes_loader = GigaScenesLoader()
    out = {}
    for scene_idx in tqdm(range(scenes_loader.num_all_scenes)):
        scene_objects = scenes_loader.load_all_objs_idx(scene_idx)
        for obj in scene_objects:
            if obj.name not in out:
                out[obj.name] = {
                    "fpath": str(obj.visual_fpath),
                    "scales": [obj.scale.tolist()],
                }
            else:
                out[obj.name]["scales"].append(obj.scale.tolist())
    data_utils.save_dict_as_json(out, Directories.ROOT / "taxonomy.json")
