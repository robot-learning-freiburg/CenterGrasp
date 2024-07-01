import numpy as np
from tqdm import tqdm
from graspnetAPI import GraspGroup
from graspnetAPI import GraspNetEval
from centergrasp.configs import Directories
from centergrasp.graspnet.rgb_data import RGBDReader

dump_folder = Directories.EVAL_GRASPNET_OLD
ge_k = GraspNetEval(root=Directories.GRASPNET, camera="kinect", split="test")
rgbd_reader = RGBDReader(mode="test")

for i in tqdm(range(len(rgbd_reader))):
    scene_name, img_name = rgbd_reader.get_scene_img_names(i)
    path = dump_folder / scene_name / "kinect" / (img_name + ".npy")
    saved_array = np.load(path)
    if len(saved_array) == 0:
        print(f"Fixing {path}")
        out_arrays = np.zeros((1, 17))
        grasp_group = GraspGroup(out_arrays)
        grasp_group.save_npy(path)
