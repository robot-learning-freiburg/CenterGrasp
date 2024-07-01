import argparse
from centergrasp import set_seeds
from centergrasp.configs import Directories
import centergrasp.data_utils as data_utils
from centergrasp.sapien.sapien_utils import CameraObsConfig
from centergrasp.sapien.sapien_envs import ENV_DICT, PickEnv


def main(
    rgb_model: str,
    seed: int,
):
    num_episodes = 20
    set_seeds(seed)
    root_path = Directories.DATA / "centergrasp_g" / "rgbd_evals"
    root_path.mkdir(parents=True, exist_ok=True)
    camera_config = CameraObsConfig(rgb=True, depth_gt=True)
    for env_type, env_cls in ENV_DICT.items():
        environment: PickEnv = env_cls(
            camera_obs_config=camera_config,
            seed=seed,
            num_episodes=num_episodes,
            headless=True,
            raytracing=True,
        )
        for episode in range(num_episodes):
            obs = environment.get_obs()
            rgb_save_path = root_path / f"{env_type}" / "rgb"
            rgb_save_path.mkdir(parents=True, exist_ok=True)
            depth_save_path = root_path / f"{env_type}" / "depth"
            depth_save_path.mkdir(parents=True, exist_ok=True)
            data_utils.save_rgb(obs.camera.rgb, rgb_save_path / f"{episode:08d}.png")
            data_utils.save_depth(obs.camera.depth, depth_save_path / f"{episode:08d}.png")
            environment.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-model", default="12c7ven5", type=str)
    parser.add_argument("--seed", default=123, type=int)
    args = parser.parse_args()
    main(**vars(args))
