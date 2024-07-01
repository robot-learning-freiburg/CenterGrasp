import tqdm
import wandb
import pathlib
import shutil
from centergrasp.configs import Directories


RUNS_DICT = {
    "real_packed": "robot-learning-lab/[CenterGrasp] Real-packed/bs8jr0cw",
    "real_pile": "robot-learning-lab/[CenterGrasp] Real-pile/bs8jr0cw",
}


def rename_image_files(path: pathlib.Path):
    image_files = sorted([image_path for image_path in path.iterdir()])
    for idx in range(len(image_files)):
        if image_files[idx].suffix != ".png":
            raise ValueError("This file is not an image!")
        right_name = f"{idx:08d}.png"
        if str(image_files[idx].stem) != right_name:
            print(f"Renaming {image_files[idx]} with {image_files[idx].parent / right_name}")
            image_files[idx].rename(image_files[idx].parent / right_name)
    return


wandb_path = Directories.DATA / "centergrasp_g" / "wandb"
wandb_path.mkdir(parents=True, exist_ok=True)
root_path = Directories.DATA / "centergrasp_g" / "rgbd_evals"
root_path.mkdir(parents=True, exist_ok=True)
api = wandb.Api()

for env_name, run_name in RUNS_DICT.items():
    run = api.run(run_name)
    history = run.scan_history()
    print(f"Downloading {env_name} images...")
    for row in tqdm.tqdm(history):
        if row["CenterGraspPipeline/rgb"] is not None:
            rgb_file = run.file(row["CenterGraspPipeline/rgb"]["path"])
            depth_file = run.file(row["CenterGraspPipeline/depth"]["path"])
            confidence_file = run.file(row["CenterGraspPipeline/confidence"]["path"])
            rgb_file.download(wandb_path / env_name, exist_ok=True)
            depth_file.download(wandb_path / env_name, exist_ok=True)
            confidence_file.download(wandb_path / env_name, exist_ok=True)

    # Copy to our folder structure
    print(f"Copying {env_name} images...")
    current_path = wandb_path / env_name / "media" / "images" / "CenterGraspPipeline"
    (root_path / env_name / "depth").mkdir(parents=True, exist_ok=True)
    (root_path / env_name / "rgb").mkdir(parents=True, exist_ok=True)
    current_img_path_list = sorted(current_path.iterdir())
    for i, cur_img_path in enumerate(current_img_path_list):
        img_name = cur_img_path.name
        if "depth" in img_name:
            shutil.copy2(cur_img_path, root_path / env_name / "depth" / f"{i:08d}.png")
        if "rgb" in img_name:
            shutil.copy2(cur_img_path, root_path / env_name / "rgb" / f"{i:08d}.png")
    rename_image_files(path=root_path / env_name / "depth")
    rename_image_files(path=root_path / env_name / "rgb")
