from tqdm import tqdm
from centergrasp.rgb.rgb_data import RGBDataset
from centergrasp.rgb.training_centergrasp import load_rgb_config

specs, _ = load_rgb_config()
train_set = RGBDataset(specs["EmbeddingCkptPath"], mode="train")

for idx in tqdm(range(len(train_set))):
    train_set[idx]
