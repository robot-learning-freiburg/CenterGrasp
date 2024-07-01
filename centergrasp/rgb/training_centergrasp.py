# type: ignore

import wandb
import torch
import pathlib
import json
import argparse
import numpy as np
import torchvision
import pytorch_lightning as pl
import centergrasp.data_utils as data_utils
from centergrasp.configs import ZED2HALF_PARAMS
from centergrasp.rgb.heatmaps import visualize_heatmap
from centergrasp.rgb.pose_utils import get_poses, visualize_poses, poses_to_numpy, MaskedPoseLoss
from simnet.lib.net import common, losses
from simnet.lib.net.functions.learning_rate import (
    lambda_learning_rate_poly,
    lambda_warmup,
)
from centergrasp.configs import Directories, DEVICE

_mse_loss = losses.MSELoss()
_mask_l1_loss = losses.MaskedL1Loss()
_pose_loss = MaskedPoseLoss()


def load_net_config(path: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    common.add_train_args(parser)
    with open(path, "r") as f:
        args = []
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line == "":
                continue
            if "=" not in line:
                continue
            key, val = line.split("=")
            args += [f"{key}", val]
    net_config = parser.parse_args(args)
    return net_config


def load_rgb_config():
    specs_path = Directories.CONFIGS / "rgb_train_specs.json"
    specs = json.load(open(specs_path))
    _config_path = Directories.CONFIGS / "rgb_train_config.json"
    if not _config_path.is_file():
        _config_path = Directories.CONFIGS / "rgb_train_config_default.json"
    _config = json.load(open(_config_path))
    # Config for compatibility with simnet
    config_path = Directories.CONFIGS / "rgb_config.txt"
    net_config = load_net_config(config_path)
    specs["net_config"] = net_config
    return specs, _config


def load_rgb_model(ckpt_name: str, ckpt_folder_name: str = "ckpt_rgb", **kwargs):
    ckpt_path = data_utils.get_checkpoint_path(ckpt_name, ckpt_folder_name)
    rgb_specs, _ = load_rgb_config()
    lit_model = LitCenterGraspModel.load_from_checkpoint(
        ckpt_path, **kwargs, map_location="cpu"
    ).to(DEVICE)
    return lit_model, rgb_specs


class LitCenterGraspModel(pl.LightningModule):
    def __init__(self, **specs):
        super().__init__()
        self.save_hyperparameters()
        self.specs = specs
        if "K" in specs:
            self.K = specs["K"]
        else:
            self.K = ZED2HALF_PARAMS.K
            vars(specs["net_config"])["img_width"] = ZED2HALF_PARAMS.width
            vars(specs["net_config"])["img_height"] = ZED2HALF_PARAMS.height
        self.epochs = specs["NumEpochs"]
        model_path = (
            pathlib.Path(__file__).parents[2] / "centergrasp/rgb" / specs["net_config"].model_file
        )
        self.model = common.get_model(model_path, specs["net_config"])
        self.init_input_normalization()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=specs["color_jitter_brightness"],
            contrast=specs["color_jitter_contrast"],
            saturation=specs["color_jitter_saturation"],
            hue=specs["color_jitter_hue"],
        )
        return

    def init_input_normalization(self):
        if self.specs["normalize_rgb"]:
            self.rgb_norm = torchvision.transforms.Normalize(
                mean=self.specs["rgb_mean"], std=self.specs["rgb_std"]
            )
        if self.specs["normalize_depth"]:
            self.depth_norm = torchvision.transforms.Normalize(
                mean=self.specs["depth_mean"], std=self.specs["depth_std"]
            )
        return

    def norm_input(self, rgb, depth):
        if self.specs["normalize_rgb"]:
            rgb = self.rgb_norm(rgb)
        if self.specs["normalize_depth"]:
            depth = self.depth_norm(depth)
        return rgb, depth

    def augment_input(self, rgb, depth):
        if self.specs["augment_color_jitter"]:
            rgb = self.color_jitter(rgb)
        return rgb, depth

    def forward(self, rgb, depth):
        rgb, depth = self.norm_input(rgb, depth)
        image = torch.cat([rgb, depth], dim=-3)
        return self.model(image)

    def log_predictions(self, rgb, heatmap_out, posemap_out, prefix):
        if self.logger is not None:
            with torch.no_grad():
                rgb_np = data_utils.img_torch_to_np(rgb)
                # Project poses
                heatmap_np = heatmap_out.detach().cpu().numpy()
                heatmap_pred_vis = visualize_heatmap(np.copy(rgb_np), heatmap_np, with_peaks=True)
                posemap_np = poses_to_numpy(posemap_out)
                poses = get_poses(heatmap_np, posemap_np)
                pose_pred_vis = visualize_poses(rgb_np, poses, self.K)
                llog = {}
                llog[f"{prefix}/pose"] = wandb.Image(pose_pred_vis, caption=prefix)
                llog[f"{prefix}/heatmap"] = wandb.Image(heatmap_pred_vis, caption=prefix)
                llog["trainer/global_step"] = self.global_step
                wandb.log(llog)
        return

    def compute_loss(
        self,
        heatmap_out,
        abs_pose_out,
        latent_emb_out,
        heatmap_target,
        pose_target,
        shape_target,
        invariance_map,
    ):
        heatmap_loss = _mse_loss(heatmap_out, heatmap_target)
        pose_loss = _pose_loss(abs_pose_out, pose_target, heatmap_target, invariance_map)
        shape_loss = _mask_l1_loss(latent_emb_out, shape_target, heatmap_target)
        loss = (
            self.specs["heatmap_loss_w"] * heatmap_loss
            + self.specs["pose_loss_w"] * pose_loss
            + self.specs["shape_loss_w"] * shape_loss
        )
        return loss, heatmap_loss, pose_loss, shape_loss

    def log_loss(self, loss, heatmap_loss, pose_loss, shape_loss, prefix):
        if self.logger is not None:
            llog = {}
            llog[f"{prefix}/loss/total"] = loss
            llog[f"{prefix}/loss/heatmap"] = heatmap_loss
            llog[f"{prefix}/loss/pose"] = pose_loss
            llog[f"{prefix}/loss/shape"] = shape_loss
            llog["trainer/global_step"] = self.global_step
            wandb.log(llog)
        return

    def training_step(self, batch, batch_idx):
        prefix = "train"
        rgb, depth, heatmap_target, pose_target, shape_target, invariance_map = batch
        rgb, depth = self.augment_input(rgb, depth)
        heatmap_out, abs_pose_out, latent_emb_out = self.forward(rgb, depth)
        loss, heatmap_loss, pose_loss, shape_loss = self.compute_loss(
            heatmap_out,
            abs_pose_out,
            latent_emb_out,
            heatmap_target,
            pose_target,
            shape_target,
            invariance_map,
        )
        self.log_loss(loss, heatmap_loss, pose_loss, shape_loss, prefix)
        return loss

    def validation_step(self, batch, batch_idx):
        prefix = "valid"
        rgb, depth, heatmap_target, pose_target, shape_target, invariance_map = batch
        heatmap_out, abs_pose_out, latent_emb_out = self.forward(rgb, depth)
        loss, heatmap_loss, pose_loss, shape_loss = self.compute_loss(
            heatmap_out,
            abs_pose_out,
            latent_emb_out,
            heatmap_target,
            pose_target,
            shape_target,
            invariance_map,
        )
        self.log_loss(loss, heatmap_loss, pose_loss, shape_loss, prefix)
        if batch_idx == 0:
            for i in range(min([4, len(rgb)])):
                self.log_predictions(rgb[i], heatmap_out[i], abs_pose_out[i], prefix)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.specs["net_config"].optim_learning_rate
        )
        lr_lambda = lambda_learning_rate_poly(self.epochs, self.specs["net_config"].optim_poly_exp)
        if (
            self.specs["net_config"].optim_warmup_epochs is not None
            and self.specs["net_config"].optim_warmup_epochs > 0
        ):
            lr_lambda = lambda_warmup(self.specs["net_config"].optim_warmup_epochs, 0.2, lr_lambda)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]
