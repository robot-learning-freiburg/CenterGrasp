import trimesh
import numpy as np
import open3d as o3d
from typing import List
import spatialmath as sm
import sapien.core as sapien
from abc import ABC, abstractmethod
from centergrasp.giga_utils import GigaParams
import centergrasp.mesh_utils as mesh_utils
from centergrasp.cameras import CameraConventions, CameraParams
from centergrasp.rgb.data_structures import RgbdDataNp
from centergrasp.rgb.pred_postprocessing import get_full_pcd
from centergrasp.giga_utils import MeshPathsLoader, get_giga_cam_pose
from centergrasp.configs import ZED2HALF_PARAMS, Directories, WSConfigs
from centergrasp.sapien.robots import SapienRobot
import centergrasp.ycb_utils as ycb_utils
import centergrasp.sapien.sapien_utils as sapien_utils
from centergrasp.sapien.sapien_utils import CameraObsConfig, Obs, Trajectory


def pc_from_box(center: np.ndarray, half_size: np.ndarray) -> np.ndarray:
    box = o3d.geometry.TriangleMesh.create_box(
        width=half_size[0] * 2, height=half_size[1] * 2, depth=half_size[2] * 2
    )
    left_bottom_corner = center - half_size
    box.translate(left_bottom_corner)
    pc = np.asarray(box.sample_points_uniformly(number_of_points=2000).points)
    return pc


class TableEnv:
    def __init__(
        self,
        camera_obs_config: CameraObsConfig = CameraObsConfig(),
        physics_fps: int = 240,
        render_fps: int = 30,
        raytracing: bool = False,
        headless: bool = False,
        sapien_robot: SapienRobot = None,
        camera_params: CameraParams = ZED2HALF_PARAMS,
    ):
        if physics_fps % render_fps != 0:
            raise ValueError(f"{physics_fps=} must be a multiple of {render_fps=}")
        self.physics_dt = 1 / physics_fps
        self.render_each = physics_fps // render_fps
        self.camera_obs_config = camera_obs_config
        self.camera_params = camera_params
        if raytracing:
            sapien_utils.enable_raytracing()
        self.engine, self.renderer, self.scene = sapien_utils.init_sapien(headless, self.physics_dt)
        if not headless:
            self.viewer = sapien_utils.init_viewer(self.scene, self.renderer, show_axes=False)
        sapien_utils.init_default_material(self.scene)
        sapien_utils.init_lights(self.scene)
        self.scene.add_ground(-0.75)
        self.table = sapien_utils.add_table(
            self.scene,
            half_size=WSConfigs.table_half_size,
            position=WSConfigs.table_position,
        )
        self.table_pc = pc_from_box(WSConfigs.table_position, WSConfigs.table_half_size)
        self.sapien_objs: List[sapien.Actor] = []
        side_camera = sapien_utils.add_camera(self.scene, self.camera_params, name="side_camera")
        side_cam_pose = get_giga_cam_pose()
        side_camera.set_pose(sapien.Pose.from_transformation_matrix(side_cam_pose))
        self.camera = side_camera
        self.headless = headless
        self.physics_fps = physics_fps
        self.sapien_robot = sapien_robot
        if sapien_robot is not None:
            sapien_robot.initialize(self.scene)
        return

    @property
    def static_scene_pc(self) -> np.ndarray:
        return self.table_pc

    def _render_step(self, idx: int):
        if idx % self.render_each == 0:
            self.scene.update_render()
            if not self.headless:
                self.viewer.render()
        return

    def add_vis_marker(self, pose: sm.SE3, name: str = "marker"):
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.01] * 3, color=[0, 1, 0])
        marker = builder.build_kinematic(name=name)
        marker.set_pose(sapien.Pose.from_transformation_matrix(pose.A))
        marker.hide_visual()
        return marker

    def remove_actor(self, actor: sapien.Actor):
        self.scene.remove_actor(actor)
        return

    def get_obs(self) -> Obs:
        transform = sm.SE3(CameraConventions.robotics_T_opencv, check=False)
        obs = Obs()
        obs.camera = sapien_utils.get_camera_obs(self.camera, self.camera_obs_config)
        obs.joint_state = self.sapien_robot.get_joint_state() if self.sapien_robot else None
        obs.camera_pose = self.get_camera_pose() * transform
        return obs

    def get_camera_pose(self) -> sm.SE3:
        cam_pose_np = self.camera.get_pose().to_transformation_matrix()
        cam_pose = sm.SE3(cam_pose_np, check=False)
        return cam_pose

    def get_scene_pc(self, obs: Obs) -> np.ndarray:
        rgb_data = RgbdDataNp(obs.camera.rgb, obs.camera.depth, None, None, None, None)
        full_pcd = get_full_pcd(rgb_data, self.camera_params, project_valid_depth_only=True)
        full_pcd.transform(obs.camera_pose)
        ws_pcd = full_pcd.crop(WSConfigs().ws_aabb)
        return np.asarray(ws_pcd.points)

    def step_once(self, idx: int = 0):
        if self.sapien_robot is not None:
            self.sapien_robot.update_qf()
        self.scene.step()
        self._render_step(idx)
        return

    def step_physics(self, seconds: float = 1.0):
        n_steps = int(seconds * self.physics_fps)
        for i in range(n_steps):
            self.step_once(i)
        return

    def open_gripper(self):
        self.sapien_robot.open_gripper()
        self.step_physics(1)
        return

    def close_gripper(self):
        self.sapien_robot.close_gripper()
        self.step_physics(1)
        return

    def execute_traj(self, trajectory: Trajectory):
        for i in range(len(trajectory.position)):
            self.sapien_robot.set_arm_targets(trajectory.position[i], trajectory.velocity[i])
            self.step_once(i)
        return

    def move_to_qpos(self, qpos: np.ndarray):
        self.sapien_robot.set_qpos_target(qpos)
        while np.mean(np.abs(self.sapien_robot.get_joint_state() - qpos)) > 1e-3:
            self.step_physics(0.1)
        return

    def reset_robot(self):
        self.sapien_robot.reset()
        self.step_once()
        return


class PickEnv(TableEnv, ABC):
    def __init__(
        self,
        seed: int,
        num_episodes: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.episode_idx = -1
        self.num_episodes = num_episodes
        self.objs: List[sapien.Actor] = []
        self.rng = np.random.default_rng(seed)
        self.reset()
        return

    @abstractmethod
    def _load_actors(self):
        return

    def reset(self):
        for obj in self.objs:
            self.scene.remove_actor(obj)
        self.objs: List[sapien.Actor] = []
        self.scene_objs: List[mesh_utils.SceneObject] = []
        self.episode_idx += 1
        self._load_actors()
        self.num_objs = len(self.objs)
        if self.sapien_robot is not None:
            self.reset_robot()
        self.step_physics(0.5)
        return

    def eval_complete(self):
        return self.episode_idx + 1 >= self.num_episodes

    def evaluate_success(self) -> bool:
        dropped_obj_idx, dropped_obj = next(
            ((idx, obj) for idx, obj in enumerate(self.objs) if obj.get_pose().p[2] < 0.0),
            (None, None),
        )
        if dropped_obj is not None:
            success = True
            self.scene.remove_actor(dropped_obj)
            self.objs.pop(dropped_obj_idx)
        else:
            success = False
        return success

    def episode_is_complete(self) -> bool:
        done = len(self.objs) == 0
        return done

    def get_gt_pc(self) -> np.ndarray:
        pc_list = []
        for obj, scene_obj in zip(self.objs, self.scene_objs):
            pose = obj.get_pose().to_transformation_matrix()
            mesh_fpath = scene_obj.visual_fpath
            mesh_trimesh = trimesh.load(mesh_fpath)
            mesh_trimesh.apply_transform(pose)
            pc = np.asarray(mesh_trimesh.sample(1000))
            pc_list.append(pc)
        combined_pc = np.concatenate(pc_list, axis=0)
        return combined_pc


class PickGigaPackedEnv(PickEnv):
    def __init__(self, **kwargs):
        self.loader = MeshPathsLoader(group="packed", mode="test")
        super().__init__(**kwargs)
        return

    def _random_pose(self, obj_height: float):
        ws_origin = GigaParams.ws_origin
        ws_size = GigaParams.ws_size
        x = self.rng.uniform(ws_origin[0] + 0.05, ws_origin[0] + ws_size - 0.05)
        y = self.rng.uniform(ws_origin[1] + 0.05, ws_origin[1] + ws_size - 0.05)
        z = obj_height / 2 + 0.001
        angle = self.rng.uniform(0.0, 2.0 * np.pi)
        rot = sm.SO3.RPY((0, 0, angle), order="zyx")
        pose = sm.SE3.Rt(R=rot, t=[x, y, z]).A
        return pose

    def _check_aabb_intersection(
        self, aabb1: trimesh.primitives.Box, aabb_list: List[trimesh.primitives.Box]
    ) -> bool:
        """
        Returns true if there is an intersection between aabb1 and any of the aabbs in aabb_list.
        Not checking for the z coordinate, since they are all at the same height.
        """
        for aabb2 in aabb_list:
            if (
                aabb1.bounds[0][0] < aabb2.bounds[1][0]
                and aabb1.bounds[1][0] > aabb2.bounds[0][0]
                and aabb1.bounds[0][1] < aabb2.bounds[1][1]
                and aabb1.bounds[1][1] > aabb2.bounds[0][1]
            ):
                return True
        return False

    def _load_actors(self):
        attempts = 0
        max_attempts = 12
        aabb_list = []
        obj_count = self.rng.poisson(4) + 1
        while len(self.objs) < obj_count and attempts < max_attempts:
            mesh_path = self.loader.get_random()
            mesh_trimesh = trimesh.load(mesh_path)
            obj_height = mesh_trimesh.bounding_box.extents[2]
            pose_mat = self._random_pose(obj_height)
            mesh_trimesh.apply_transform(pose_mat)
            aabb = mesh_trimesh.bounding_box
            if self._check_aabb_intersection(aabb, aabb_list):
                attempts += 1
                continue
            aabb_list.append(aabb)
            scene_obj = self.loader.meshpath_to_sceneobj(mesh_path, pose_mat)
            material = sapien_utils.random_render_material(self.renderer)
            obj = sapien_utils.add_object_dynamic(self.scene, scene_obj, material)
            self.objs.append(obj)
            self.scene_objs.append(scene_obj)
            self.step_physics(0.1)
        return


class PickGigaPileEnv(PickEnv):
    def __init__(self, **kwargs):
        self.loader = MeshPathsLoader(group="pile", mode="test")
        super().__init__(**kwargs)
        return

    def _load_box(self) -> sapien.Actor:
        box_mesh = Directories.GIGA_REPO / "data/urdfs/setup/box.obj"
        box_pose = sm.SE3.Trans(GigaParams.ws_origin) * sm.SE3.Rx(np.pi / 2)
        box_scale = 1.3 * np.array([0.01, 0.005, 0.01])
        box = mesh_utils.SceneObject(box_mesh, box_mesh, box_pose.A, box_scale, name="box")
        box_sapien = sapien_utils.add_object_nonconvex(self.scene, box)
        self.step_physics(1.0)
        return box_sapien

    def _load_actors(self):
        box_sapien = self._load_box()
        obj_count = self.rng.poisson(4) + 1
        for _ in range(obj_count):
            mesh_path = self.loader.get_random()
            scene_obj = self.loader.meshpath_to_sceneobj(mesh_path)
            material = sapien_utils.random_render_material(self.renderer)
            obj = sapien_utils.add_object_dynamic(self.scene, scene_obj, material)
            position = GigaParams.ws_center + np.array([0.0, 0.0, 0.2])
            quat = self.rng.uniform(-1, 1, 4)
            obj.set_pose(sapien.Pose(position, quat))
            self.objs.append(obj)
            self.scene_objs.append(scene_obj)
            self.step_physics(1.0)
        self.scene.remove_actor(box_sapien)
        return


class PickYCBPackedEnv(PickGigaPackedEnv):
    def __init__(self, **kwargs):
        self.loader = ycb_utils.YCBPathsLoader()
        super(PickGigaPackedEnv, self).__init__(**kwargs)
        return


class PickYCBPileEnv(PickGigaPileEnv):
    def __init__(self, **kwargs):
        self.loader = ycb_utils.YCBPathsLoader()
        super(PickGigaPileEnv, self).__init__(**kwargs)
        return


def build_actor_ycb(
    model_id: str,
    scene: sapien.Scene,
    scale: float = 1.0,
    density=1000,
):
    builder = scene.create_actor_builder()
    model_dir = Directories.YCB / "mani_skill2_ycb/models" / model_id

    collision_file = str(model_dir / "collision.obj")
    builder.add_multiple_collisions_from_file(
        filename=collision_file,
        scale=[scale] * 3,
        density=density,
    )

    visual_file = str(model_dir / "textured.obj")
    builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

    actor = builder.build()
    return actor


class PickYCBEnv(PickEnv):
    EPISODE_JSON = Directories.YCB / "pick_clutter/ycb_train_5k.json.gz"
    MODEL_JSON = Directories.YCB / "mani_skill2_ycb/info_pick_v0.json"
    HEAVY_OBJECTS = ["006_mustard_bottle"]

    def __init__(
        self,
        num_episodes: int,
        **kwargs,
    ):
        if not self.EPISODE_JSON.exists():
            raise FileNotFoundError(
                f"Episode json ({self.EPISODE_JSON}) is not found."
                "To download default json:"
                "`python -m mani_skill2.utils.download_asset pick_clutter_ycb`."
            )
        self.episodes: List[dict] = ycb_utils.load_json(self.EPISODE_JSON)
        self.model_db: dict[str, dict] = ycb_utils.load_json(self.MODEL_JSON)
        if num_episodes > len(self.episodes):
            raise ValueError("num_episodes is larger than the number of episodes in the json.")
        super().__init__(num_episodes=num_episodes, **kwargs)
        return

    def _load_model(self, model_id, model_scale=1.0):
        density = self.model_db[model_id].get("density", 1000)
        if model_id in self.HEAVY_OBJECTS:
            density /= 2
        obj = build_actor_ycb(
            model_id,
            self.scene,
            scale=model_scale,
            density=density,
        )
        obj.name = model_id
        obj.set_damping(0.1, 0.1)
        return obj

    def _load_actors(self):
        episode = self.episodes[self.episode_idx]
        self.bbox_sizes = []
        xy_avg = np.mean(
            [np.float32(actor_cfg["pose"])[:2] for actor_cfg in episode["actors"]], axis=0
        )
        for actor_cfg in episode["actors"]:
            model_id = actor_cfg["model_id"]
            model_scale = actor_cfg["scale"]
            pose = np.float32(actor_cfg["pose"])
            obj = self._load_model(model_id, model_scale=model_scale)
            pos_offset = [0.45 - xy_avg[0], 0, 1e-3]
            obj.set_pose(sapien.Pose(pose[:3] + pos_offset, pose[3:]))
            self.objs.append(obj)

            bbox = self.model_db[model_id]["bbox"]
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.bbox_sizes.append(bbox_size * model_scale)
        return


ENV_DICT = {
    "giga_packed": PickGigaPackedEnv,
    "giga_pile": PickGigaPileEnv,
    "ycb_packed": PickYCBPackedEnv,
    "ycb_pile": PickYCBPileEnv,
}

if __name__ == "__main__":
    env = TableEnv()
    env.step_physics(10)
