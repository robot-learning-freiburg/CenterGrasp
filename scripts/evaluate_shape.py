import argparse
import wandb
import tqdm
import torch
import numpy as np
from typing import List
from dataclasses import dataclass, field
from sklearn.neighbors import NearestNeighbors
from kaolin.metrics.pointcloud import sided_distance
from centergrasp import set_seeds
from centergrasp.configs import DEVICE
from centergrasp.visualization import RerunViewer
from centergrasp.pipelines.giga_pipeline import GigaPipeline
from centergrasp.pipelines.centergrasp_pipeline import CenterGraspPipeline
from centergrasp.sapien.sapien_utils import CameraObsConfig
from centergrasp.sapien.sapien_envs import ENV_DICT, PickEnv
from centergrasp.configs import ZED2HALF_PARAMS


def chamfer_distance(x, y, metric="l2", direction="bi"):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default `l2`
        metric to use for distance computation.
        Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}  # noqa E501
    """
    assert x.shape[1] == 3 and y.shape[1] == 3
    x = x[~np.isnan(x).any(axis=1), :]
    y = y[~np.isnan(y).any(axis=1), :]

    if direction == "y_to_x":
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric).fit(
            x
        )
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == "x_to_y":
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric).fit(
            y
        )
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == "bi":
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric).fit(
            x
        )
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric).fit(
            y
        )
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: 'y_x', 'x_y', 'bi'")

    return chamfer_dist


def pointcloud_iou(pred_points, gt_points, radius=0.01):
    # Source: https://github.com/NVIDIAGameWorks/kaolin/issues/271
    # $ pip install kaolin==0.14.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-{TORCH_VER}_cu{CUDA_VER}.html  # noqa E501
    pred_distances = torch.sqrt(sided_distance(gt_points, pred_points)[0])
    gt_distances = torch.sqrt(sided_distance(pred_points, gt_points)[0])

    fp = (gt_distances > radius).float().sum()
    tp = (gt_distances <= radius).float().sum()
    precision = tp / (tp + fp)
    tp = (pred_distances <= radius).float().sum()
    fn = (pred_distances > radius).float().sum()
    recall = tp / (tp + fn)

    f_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return f_score, iou


@dataclass
class Metrics:
    cd_bi: List[float] = field(default_factory=list)
    iou: List[float] = field(default_factory=list)

    @property
    def avg_cd_bi(self):
        return np.mean(self.cd_bi) * 1000

    @property
    def avg_iou(self):
        return np.mean(self.iou)


def main(
    rgb_model: str,
    env: str,
    seed: int,
    num_episodes: int,
    headless: bool,
    log_wandb: bool,
):
    set_seeds(seed)
    giga_mode = "packed"
    method_names = ["centergrasp", "centergrasp_noicp", "giga"]
    methods = [
        CenterGraspPipeline(rgb_model, seed, visualize=not headless, camera_params=ZED2HALF_PARAMS),
        CenterGraspPipeline(rgb_model, seed, visualize=not headless, use_icp=False, camera_params=ZED2HALF_PARAMS),
        GigaPipeline(giga_mode, seed, visualize=not headless, real_robot=False),
    ]
    metrics = [Metrics() for _ in range(len(methods))]
    camera_config = CameraObsConfig(rgb=True, depth_gt=True)
    environment: PickEnv = ENV_DICT[env](
        camera_obs_config=camera_config,
        seed=seed,
        num_episodes=num_episodes,
        headless=headless,
    )
    if not headless:
        RerunViewer()
    wandb.init(
        project="[CenterGrasp] SimEvalShape",
        entity="robot-learning-lab",
        config={"env": env, "seed": seed, "num_episodes": num_episodes},
        mode="online" if log_wandb else "disabled",
    )
    for episode in tqdm.tqdm(range(num_episodes)):
        obs = environment.get_obs()
        gt_pc = environment.get_gt_pc()
        predictions = [method.predict_shape(obs)[0] for method in methods]
        if not headless:
            RerunViewer.add_np_pointcloud("vis/gt_pc", gt_pc)
            for method_name, prediction in zip(method_names, predictions):
                RerunViewer.add_np_pointcloud(f"vis/{method_name}_pred_pc", prediction)
        # Calculate metrics
        for i, prediction in enumerate(predictions):
            cd_bi = chamfer_distance(gt_pc, prediction, direction="bi")
            _, iou = pointcloud_iou(
                pred_points=torch.tensor(prediction, device=DEVICE).unsqueeze(0),
                gt_points=torch.tensor(gt_pc, device=DEVICE).unsqueeze(0),
            )
            metrics[i].cd_bi.append(cd_bi)
            metrics[i].iou.append(iou.item())
        # Log
        log_data = {
            "episode": episode,
            "cd_bi_centergrasp": metrics[0].cd_bi[-1],
            "iou_centergrasp": metrics[0].iou[-1],
            "cd_bi_centergrasp_noicp": metrics[1].cd_bi[-1],
            "iou_centergrasp_noicp": metrics[1].iou[-1],
            "cd_bi_giga": metrics[2].cd_bi[-1],
            "iou_giga": metrics[2].iou[-1],
        }
        wandb.log(log_data)
        environment.reset()

    for method_name, metric in zip(method_names, metrics):
        print(f"{method_name} bi:\t{metric.avg_cd_bi:.1f}")
        print(f"{method_name} iou:\t{metric.avg_iou}")
    wandb.run.summary["avg_bi_centergrasp"] = metrics[0].avg_cd_bi
    wandb.run.summary["avg_iou_centergrasp"] = metrics[0].avg_iou
    wandb.run.summary["avg_bi_centergrasp_noicp"] = metrics[1].avg_cd_bi
    wandb.run.summary["avg_iou_centergrasp_noicp"] = metrics[1].avg_iou
    wandb.run.summary["avg_bi_giga"] = metrics[2].avg_cd_bi
    wandb.run.summary["avg_iou_giga"] = metrics[2].avg_iou
    wandb.finish()
    print("Done")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-model", default="12c7ven5", type=str)
    parser.add_argument(
        "--env",
        default="giga_packed",
        type=str,
        choices=list(ENV_DICT.keys()),
    )
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--num-episodes", default=200, type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--log-wandb", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
