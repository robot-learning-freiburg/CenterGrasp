import numpy as np
from tqdm import tqdm
from centergrasp.configs import Directories

root_path = Directories.GRASPS / "poses" / "packed"
num_grasps = []
for fpath in tqdm(root_path.iterdir()):
    data = np.load(fpath, allow_pickle=False)
    num_grasps.append(len(data))

print("Average number of grasps per object: ", np.mean(num_grasps))
