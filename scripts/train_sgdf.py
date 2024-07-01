import os
import tyro
import random
import numpy as np

# pyright: reportGeneralTypeIssues=false


def main(log_wandb: bool = False, gpu_num: int = 0, dataset: str = "giga"):
    # Need to do this before importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    assert dataset in ["giga", "graspnet"]

    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from centergrasp.configs import Directories
    from centergrasp.sgdf.sgdf_dataset import SGDFDataset
    from centergrasp.graspnet.sgdf_dataset import SGDFDatasetGraspnet
    from centergrasp.sgdf.training_deep_sgdf import LitSGDFModel, load_sgdf_config, EmbeddingLogger

    # Seeds
    seed = 12345
    os.environ["PYTHONHASHSEED"] = str(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    specs, _config = load_sgdf_config()

    # Data
    dataset_cls = SGDFDataset if dataset == "giga" else SGDFDatasetGraspnet
    train_set = dataset_cls(
        points_per_obj=_config["points_per_obj"],
        mode="train",
    )
    valid_set = dataset_cls(
        points_per_obj=_config["points_per_obj"],
        mode="valid",
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
    specs["points_per_obj"] = _config["points_per_obj"]
    specs["num_train_objects"] = train_set.get_num_objects()
    lit_model = LitSGDFModel(**specs)

    # Loggers
    logger = False
    enable_checkpointing = False
    callbacks = []
    if log_wandb:
        logger = WandbLogger(project="[CenterGrasp] SGDF", entity="robot-learning-lab")
        logger.watch(lit_model)  # type: ignore
        enable_checkpointing = True
        checkpoints_path = Directories.ROOT / "ckpt_sgdf" / logger.version
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoints_path, every_n_epochs=specs["ckpt_every_n_epochs"]
        )
        callbacks.append(checkpoint_callback)
        callbacks.append(EmbeddingLogger(logger))

        logger.experiment.config.update(specs)
        logger.experiment.config.update(_config)
        logger.experiment.config.update({"dataset": dataset})

    # Training
    trainer = pl.Trainer(
        accelerator=_config["accelerator"],
        devices=1,
        max_epochs=specs["NumEpochs"],
        logger=logger,
        enable_checkpointing=enable_checkpointing,
        callbacks=callbacks,
    )

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


tyro.cli(main)
