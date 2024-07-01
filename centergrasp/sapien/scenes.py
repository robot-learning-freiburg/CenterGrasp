import numpy as np
from typing import List, Tuple
from typing import Optional
import sapien.core as sapien
from dataclasses import dataclass
from multiprocessing.synchronize import Lock as LockBase
from centergrasp.configs import Directories, ZED2HALF_PARAMS
from centergrasp.cameras import sample_cam_poses_shell
from centergrasp.rgb.heatmaps import heatmap_from_segmentation
from centergrasp.mesh_utils import SceneObject, AmbientCGTexture
import centergrasp.sapien.sapien_utils as sapien_utils


@dataclass
class GTData:
    rgb: np.ndarray
    depth: np.ndarray
    camTposes: np.ndarray
    binary_masks: np.ndarray
    heatmap: np.ndarray


class SceneImgRenderer:
    def __init__(
        self, headless: bool, raytracing: bool, imgs_per_scene: int, lock: Optional[LockBase] = None
    ) -> None:
        # Initialize SAPIEN
        if raytracing:
            sapien_utils.enable_raytracing()
        self.engine, self.renderer, self.scene = sapien_utils.init_sapien(
            headless=headless, physics_dt=1 / 240
        )
        if not headless:
            self.viewer = sapien_utils.init_viewer(self.scene, self.renderer, show_axes=False)
        sapien_utils.init_default_material(self.scene)
        self.sensor = sapien_utils.add_sensor(self.scene, ZED2HALF_PARAMS, name="sensor")
        self.camera_obs_config = sapien_utils.CameraObsConfig(
            rgb=True, depth_real=True, depth_gt=False, segmentation=True, normal=False
        )
        self.textures = [AmbientCGTexture(path) for path in Directories.TEXTURES.iterdir()]
        self.lights = sapien_utils.init_lights(self.scene)
        self.sapien_objs: List[sapien.Actor] = []
        self.table: sapien.Actor = None
        self.ground: sapien.ActorStatic = None
        self.imgs_per_scene = imgs_per_scene
        self.lock = lock
        return

    def _randomize_light(self) -> None:
        for light in self.lights:
            light_position = np.random.uniform(-2, 2, size=3)
            light_position[2] += 3
            light.set_position(light_position)
        return

    def _load_random_floor(self) -> None:
        if self.ground is not None:
            self.scene.remove_actor(self.ground)
        material = sapien_utils.render_material_from_ambient_cg_texture(
            self.renderer, np.random.choice(self.textures)
        )
        self.ground = self.scene.add_ground(altitude=-0.75, render_material=material)
        return

    def _load_random_table(self) -> None:
        if self.table is not None:
            self.scene.remove_actor(self.table)
        material = sapien_utils.render_material_from_ambient_cg_texture(
            self.renderer, np.random.choice(self.textures)
        )
        table_half_size = [
            np.random.uniform(0.2, 0.6),
            np.random.uniform(0.2, 0.6),
            np.random.uniform(0.01, 0.1),
        ]
        table_position = [0.15, 0.15, 0.05 - table_half_size[2]]
        self.table = sapien_utils.add_table(
            self.scene,
            half_size=table_half_size,
            position=table_position,
            material=material,
        )
        return

    def _load_objs_random_material(self, objs: List[SceneObject]) -> List[sapien.Actor]:
        for obj in self.sapien_objs:
            self.scene.remove_actor(obj)
        materials = [sapien_utils.random_render_material(self.renderer) for _ in objs]
        sapien_objs = [
            sapien_utils.add_object_kinematic(self.scene, obj, material)
            for obj, material in zip(objs, materials)
        ]
        return sapien_objs

    def _sample_random_camera_poses(self, objs_center: np.ndarray) -> np.ndarray:
        camera_poses = sample_cam_poses_shell(
            center=objs_center, coi_half_size=0.05, num_poses=self.imgs_per_scene
        )
        return camera_poses

    def _render_obs(self, cam_pose: np.ndarray) -> sapien_utils.CameraObs:
        self.sensor.set_pose(sapien.Pose.from_transformation_matrix(cam_pose))
        self.scene.update_render()
        if self.lock is not None:
            self.lock.acquire()
        camera_obs = sapien_utils.get_sensor_obs(self.sensor, self.camera_obs_config)
        if self.lock is not None:
            self.lock.release()
        return camera_obs

    def _objs_pose_in_cam_frame(self, objs: List[SceneObject], cam_pose: np.ndarray) -> np.ndarray:
        camTobjs = np.array([np.linalg.inv(cam_pose) @ obj.pose4x4 for obj in objs])
        return camTobjs

    def _setup_scene(self, objs: List[SceneObject]) -> None:
        self.renderer.clear_cached_resources()
        self._load_random_floor()
        self._load_random_table()
        self._randomize_light()
        self.sapien_objs = self._load_objs_random_material(objs)
        self.scene.step()
        return

    def _make_gt(self, objs: List[SceneObject], camera_pose: np.ndarray) -> GTData:
        camera_obs = self._render_obs(camera_pose)
        heatmap, simple_bmasks = heatmap_from_segmentation(
            segmentation=camera_obs.segmentation, indices=[actor.id for actor in self.sapien_objs]
        )
        camTposes = self._objs_pose_in_cam_frame(objs, camera_pose)
        return GTData(camera_obs.rgb, camera_obs.depth_real, camTposes, simple_bmasks, heatmap)

    def make_data(self, objs: List[SceneObject]) -> Tuple[List[GTData], List[dict]]:
        self._setup_scene(objs)
        objs_center = np.mean([obj.pose4x4[:3, 3] for obj in objs], axis=0)
        objs_info = [{"mesh_path": str(obj.visual_fpath), "scale": list(obj.scale)} for obj in objs]
        camera_poses = self._sample_random_camera_poses(objs_center)
        gt_data_list = [self._make_gt(objs, camera_pose) for camera_pose in camera_poses]

        # For debugging
        # while not self.viewer.closed:  # Press key q to quit
        #     self.scene.step()
        #     self.scene.update_render()
        #     self.viewer.render()
        return gt_data_list, objs_info
