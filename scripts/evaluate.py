import argparse
import wandb
import tqdm
from centergrasp import set_seeds
from centergrasp.visualization import RerunViewer
from centergrasp.sapien.behaviors import PickBehavior, PickBehaviorGripper
from centergrasp.pipelines.planners import MPLibPlanner
from centergrasp.pipelines.giga_pipeline import GigaPipeline
from centergrasp.pipelines.centergrasp_pipeline import CenterGraspPipeline
from centergrasp.sapien.sapien_utils import CameraObsConfig
from centergrasp.sapien.sapien_envs import ENV_DICT
from centergrasp.sapien.robots import ROBOTS_DICT


def main(
    rgb_model: str,
    robot_type: str,
    env: str,
    method: str,
    seed: int,
    num_episodes: int,
    headless: bool,
    log_wandb: bool,
):
    set_seeds(seed)
    # giga_mode = "packed" if env == "giga_packed" else "pile"
    giga_mode = "packed"
    if method == "centergrasp":
        pipeline = CenterGraspPipeline(rgb_model, seed, visualize=not headless)
    elif method == "centergrasp_noicp":
        pipeline = CenterGraspPipeline(rgb_model, seed, visualize=not headless, use_icp=False)
    elif method == "giga":
        pipeline = GigaPipeline(giga_mode, seed, visualize=not headless, real_robot=False)
    else:
        raise ValueError(f"Invalid method: {method}")
    wandb.init(
        project="[CenterGrasp] SimEvalYCB",
        entity="robot-learning-lab",
        config={
            "robot_type": robot_type,
            "env": env,
            "method": method,
            "seed": seed,
            "num_episodes": num_episodes,
        },
        mode="online" if log_wandb else "disabled",
    )

    camera_config = CameraObsConfig(rgb=True, depth_gt=True)
    planner = MPLibPlanner()
    robot = ROBOTS_DICT[robot_type]()
    environment = ENV_DICT[env](
        camera_obs_config=camera_config,
        seed=seed,
        num_episodes=num_episodes,
        headless=headless,
        sapien_robot=robot,
    )
    if robot_type == "franka":
        behavior = PickBehavior(environment, pipeline, planner)
    elif robot_type == "gripper":
        behavior = PickBehaviorGripper(environment, pipeline, planner)
    else:
        raise ValueError(f"Invalid robot type: {robot_type}")
    if not headless:
        RerunViewer()

    successes = 0
    grasp_attempts = 0
    total_objects = 0
    max_aborted_runs = 2
    max_consecutive_failures = 3
    for episode in tqdm.tqdm(range(num_episodes)):
        aborted_runs = 0
        consecutive_failures = 0
        total_objects += environment.num_objs
        while not (
            environment.episode_is_complete()
            or consecutive_failures >= max_consecutive_failures
            or aborted_runs >= max_aborted_runs
        ):
            info = behavior.run()
            print(f"Run complete: {info['run_complete']}")
            print(f"Grasp success: {info['grasp_success']}")
            if not info["run_complete"]:
                aborted_runs += 1
                continue
            grasp_attempts += 1
            if info["grasp_success"]:
                successes += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1

        success_rate = successes / grasp_attempts if grasp_attempts > 0 else 0
        declutter_rate = successes / total_objects
        print("Episode complete!")
        print(f"Current Success Rate: {success_rate}")
        print(f"Current Declutter Rate: {declutter_rate}")
        environment.reset()
        log_data = {
            "epidsode": episode,
            "success_rate": success_rate,
            "declutter_rate": declutter_rate,
        }
        wandb.log(log_data)
    wandb.finish()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-model", default="12c7ven5", type=str)
    parser.add_argument(
        "--robot-type",
        default="franka",
        type=str,
        choices=list(ROBOTS_DICT.keys()),
    )
    parser.add_argument(
        "--env",
        default="giga_packed",
        type=str,
        choices=list(ENV_DICT.keys()),
    )
    parser.add_argument(
        "--method",
        default="centergrasp",
        type=str,
        choices=["centergrasp", "centergrasp_noicp", "giga"],
    )
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--num-episodes", default=200, type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--log-wandb", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
