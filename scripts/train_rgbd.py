import os
import tyro
import random
import numpy as np
from typing import Optional

# pyright: reportGeneralTypeIssues=false


def main(
    log_wandb: bool = False,
    gpu_num: int = 0,
    seed: int = 12345,
    resume_ckpt: Optional[str] = None,
    dataset: str = "giga",
):
    assert dataset in ["giga", "graspnet"]
    # Need to do this before importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from centergrasp.configs import Directories
    from centergrasp.rgb.training_centergrasp import LitCenterGraspModel, load_rgb_config
    from centergrasp.rgb.rgb_data import RGBDataset
    from centergrasp.configs import ZED2HALF_PARAMS
    try:
        from centergrasp.graspnet.rgb_data import RGBDatasetGraspnet, KINECT_HALF_PARAMS
    except ImportError:
        print("Graspnet API not available")
    
    # Seeds
    os.environ["PYTHONHASHSEED"] = str(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get args
    specs, _config = load_rgb_config()
    cam_params = KINECT_HALF_PARAMS if dataset == "graspnet" else ZED2HALF_PARAMS
    specs["K"] = cam_params.K
    vars(specs["net_config"])["img_width"] = cam_params.width
    vars(specs["net_config"])["img_height"] = cam_params.height

    # Data
    dataset_cls = RGBDataset if dataset == "giga" else RGBDatasetGraspnet
    train_set = dataset_cls(specs["EmbeddingCkptPath"], mode="train")
    valid_set = dataset_cls(
        specs["EmbeddingCkptPath"], mode="valid" if dataset == "giga" else "test"
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=_config["num_workers"],
        batch_size=_config["batch_size"],
        shuffle=True,
        persistent_workers=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        num_workers=_config["num_workers"],
        batch_size=_config["batch_size"],
        shuffle=False,
        persistent_workers=True,
    )

    # Model
    lit_model = LitCenterGraspModel(**specs)

    # Loggers
    logger = False
    enable_checkpointing = False
    callbacks = []
    if log_wandb:
        logger = WandbLogger(project="[CenterGrasp] RGB", entity="robot-learning-lab")
        logger.watch(lit_model)  # type: ignore
        enable_checkpointing = True
        checkpoints_path = Directories.ROOT / "ckpt_rgb" / logger.version
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_path, every_n_epochs=1)
        callbacks.append(checkpoint_callback)
        logger.experiment.config.update(specs)
        logger.experiment.config.update(_config)
        logger.experiment.config.update({"seed": seed})

    # Resume Ckpt
    resume_ckpt_path = (
        Directories.ROOT / "ckpt_rgb" / resume_ckpt if resume_ckpt is not None else None
    )

    # Training
    trainer = pl.Trainer(
        accelerator=_config["accelerator"],
        devices=1,
        max_epochs=specs["NumEpochs"],
        logger=logger,
        enable_checkpointing=enable_checkpointing,
        callbacks=callbacks,
    )

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=resume_ckpt_path,
    )


tyro.cli(main)
