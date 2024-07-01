import re
import cv2
import json
import torch
import pathlib
import functools
import dataclasses
import numpy as np
from PIL import Image
from typing import Tuple
import pycocotools.mask as mask_tools


def save_rgb(rgb: np.ndarray, path: pathlib.Path):
    if rgb.dtype == np.float32 or rgb.dtype == np.float64:
        rgb = (rgb * 255).astype(np.uint8)
    rgb_pil = Image.fromarray(rgb)
    rgb_pil.save(path)
    return


def save_depth(depth: np.ndarray, path: pathlib.Path):
    if depth.dtype == np.float32 or depth.dtype == np.float64:
        depth = (depth * 1000).astype(np.uint16)
    cv2.imwrite(str(path), depth)
    return


def save_depth_colormap(depth: np.ndarray, path: pathlib.Path):
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth, alpha=255 / 1.5), cv2.COLORMAP_JET
    )
    cv2.imwrite(str(path), depth_colormap)


def load_rgb_from_file(path: pathlib.Path) -> np.ndarray:
    rgb_uint8 = np.array(Image.open(path))
    return rgb_uint8


def load_depth_from_file(path: pathlib.Path) -> np.ndarray:
    depth = np.array(Image.open(path), dtype=np.float32) / 1000
    return depth


def save_binary_masks(masks: np.ndarray, path: pathlib.Path):
    # For efficient storage, the masks are encoded using RLE.
    encoded_masks = [mask_tools.encode(np.asfortranarray(mask)) for mask in masks]
    for mask in encoded_masks:
        mask["counts"] = mask["counts"].decode("utf-8")
    save_dict_as_json(encoded_masks, path)
    return


def load_binary_masks(path: pathlib.Path) -> np.ndarray:
    with open(path, "r") as fp:
        data = json.load(fp)
    decoded_masks = np.array([mask_tools.decode(mask) for mask in data])
    return decoded_masks


def save_dict_as_json(data: dict, path: pathlib.Path):
    with open(path, "w") as fp:
        json.dump(data, fp)
    return


def th_to_np(*x: torch.Tensor) -> Tuple[np.ndarray]:
    return tuple([t.detach().cpu().numpy() for t in x])


def img_torch_to_np(img_torch: torch.Tensor) -> np.ndarray:
    """
    Input: torch tensor of type float32
    Output: numpy array of type uint8
    """
    img_np = img_torch.detach().cpu().numpy()
    if len(img_np.shape) > 2:
        img_np = img_np.transpose(1, 2, 0)
    img_np = np.array(img_np * 255, dtype=np.uint8)
    return img_np


def img_np_to_torch(img_np: np.ndarray) -> torch.Tensor:
    """
    Input: numpy array of type uint8
    Output: torch tensor of type float32
    """
    img_th = torch.tensor(img_np / 255, dtype=torch.float32)
    if len(img_np.shape) > 2:
        img_th = img_th.permute(2, 0, 1)
    return img_th


@dataclasses.dataclass
@functools.total_ordering
class CkptTracker:
    ckpt_path: pathlib.Path
    epoch: int
    step: int

    def __eq__(self, __o: "CkptTracker") -> bool:
        return self.epoch == __o.epoch and self.step == __o.step

    def __lt__(self, __o: "CkptTracker") -> bool:
        return self.epoch < __o.epoch and self.step < __o.step


def get_checkpoint_path(ckpt_name, ckpt_folder_name):
    ckpt_path = pathlib.Path(__file__).parents[1].resolve() / ckpt_folder_name / ckpt_name
    if ckpt_path.is_dir():  # We only passed a directory -->
        all_ckpts = list(ckpt_path.glob("*.ckpt"))
        ckpt_trackers = []

        for ckpt in all_ckpts:
            e_str, epoch, s_str, step = re.split("[=-]", ckpt.stem)
            assert e_str == "epoch" and s_str == "step"
            ckpt_trackers.append(CkptTracker(ckpt_path=ckpt, epoch=int(epoch), step=int(step)))

        ckpt_path = max(ckpt_trackers).ckpt_path
        print(f"Using {ckpt_path}")
    return ckpt_path
