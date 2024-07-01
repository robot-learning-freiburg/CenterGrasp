import pathlib
from typing import List
from torch.utils.data import Dataset
from centergrasp.sgdf.sgdf_dataset import SGDFDataset
from centergrasp.graspnet.sgdf_data import TOT_NUM_OBJECTS, get_fpath_from_obj_idx


def get_fpath_list(mode: str) -> List[pathlib.Path]:
    assert mode in ["train", "valid"]
    file_paths = [get_fpath_from_obj_idx(i, mode) for i in range(TOT_NUM_OBJECTS)]
    return file_paths


class SGDFDatasetGraspnet(SGDFDataset):
    def __init__(self, points_per_obj: int, mode: str = "train") -> None:
        Dataset.__init__(self)
        if mode not in ["train", "valid"]:
            raise ValueError("mode must be either 'train' or 'valid'")
        self.file_paths = get_fpath_list(mode)
        self.points_per_obj = points_per_obj
        # Since we only sample ~8% of the points
        self.len_scaling = 8 if mode == "train" else 1
        return


if __name__ == "__main__":
    dataset = SGDFDatasetGraspnet(10000)
    data = dataset[15]
