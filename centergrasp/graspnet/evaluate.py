import wandb
import argparse
import numpy as np
from tqdm import tqdm
from graspnetAPI import GraspNetEval
from centergrasp.configs import Directories


def main(method: str, split: str, top_k: int, log_wandb: bool = False):
    dump_folder = Directories.EVAL_GRASPNET
    ge_k = GraspNetEval(root=Directories.GRASPNET, camera="kinect", split=split)
    wandb.init(
        project="[CenterGrasp] SimGraspNet",
        entity="robot-learning-lab",
        config={"method": method, "split": split, "top_k": top_k},
        mode="online" if log_wandb else "disabled",
    )

    AP_04_list = []
    AP_08_list = []
    AP_list = []
    for scene_id in tqdm(ge_k.sceneIds):
        acc = ge_k.eval_scene(scene_id=scene_id, dump_folder=dump_folder, TOP_K=top_k, vis=False)
        np_acc = np.array(acc) * 100
        AP_04 = np.mean(np_acc[..., 1])
        AP_08 = np.mean(np_acc[..., 3])
        AP = np.mean(np_acc)
        AP_04_list.append(AP_04)
        AP_08_list.append(AP_08)
        AP_list.append(AP)

        log_data = {
            "AP_04": AP_04,
            "AP_08": AP_08,
            "AP": AP,
        }
        wandb.log(log_data)

    split_AP_04 = np.mean(AP_04_list)
    split_AP_08 = np.mean(AP_08_list)
    split_AP = np.mean(AP_list)
    wandb.run.summary["split_AP_04"] = split_AP_04
    wandb.run.summary["split_AP_08"] = split_AP_08
    wandb.run.summary["split_AP"] = split_AP
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="centergrasp", choices=["centergrasp"])
    parser.add_argument("--split", type=str, choices=["test_seen", "test_similar", "test_novel"])
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--log-wandb", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
