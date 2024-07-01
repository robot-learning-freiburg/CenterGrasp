from typing import List, Optional, Tuple
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh
import copy

if not os.uname()[1].startswith("rlgpu"):
    import rerun as rr

import centergrasp.rgb.heatmaps as heatmaps
import centergrasp.rgb.pose_utils as pose_utils
from centergrasp.rgb.data_structures import RgbdDataNp
from centergrasp.sgdf.data_structures import SGDFPrediction
from centergrasp.rgb.data_structures import FullObjPred, PostprObjPred
from centergrasp.configs import ZED2HALF_PARAMS
from simnet.lib.transform import Pose


COLOR_PALETTE = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
]


def color_pcd(pcd_np: np.ndarray, color: Optional[Tuple] = None):
    """
    Args:
        pcd: numpy array of shape (N, 3)
    """
    if not color:
        min_z = pcd_np[:, 2].min()
        max_z = pcd_np[:, 2].max()
        cmap_norm = mpl.colors.Normalize(vmin=min_z, vmax=max_z)
        # 'hsv' is changeable to any name as stated here:
        # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        point_colors = plt.get_cmap("jet")(cmap_norm(pcd_np[:, 2]))[:, :3]  # We don't need alpha
    else:
        assert len(color) == 3
        N, _ = pcd_np.shape
        point_colors = np.tile(color, (N, 1))
    return point_colors


def visualize_predictions(predictions, full_pcd):
    vis = Open3DViewer()
    vis.add_geometries("full_pcd", full_pcd)
    for i, pred in enumerate(predictions):
        vis.add_pointcloud_o3d("pred", pred.pointcloud_o3d, i)
        vis.add_grasps(f"grasps_{i}_", pred.best_grasps)
    vis.run_vis()
    return


class RerunViewer:
    def __init__(self, addr: str = None):
        rr.init("centergrasp")
        if addr is None:
            rr.spawn()
        else:
            rr.connect(addr)
        RerunViewer.clear()
        return

    @staticmethod
    def visualize_prediction(preds_raw: FullObjPred, preds_pp: PostprObjPred, idx: int):
        RerunViewer.add_o3d_pointcloud(f"vis/shapes_raw/{idx}", preds_raw.pc_o3d, radii=0.003)
        RerunViewer.add_o3d_pointcloud(f"vis/shapes_icp/{idx}", preds_pp.pc_o3d, radii=0.003)
        RerunViewer.add_grasps(f"vis/grasps/{idx}", preds_pp.grasp_poses)
        return

    @staticmethod
    def vis_sgdf_prediction(name: str, data: SGDFPrediction, num_grasps: int = 10):
        grasps_idx = np.random.choice(data.grasp_poses.shape[0], size=num_grasps, replace=False)
        RerunViewer.add_o3d_pointcloud(f"vis/{name}/pc", data.pc_o3d, radii=0.002)
        RerunViewer.add_grasps(f"vis/{name}/grasps", data.grasp_poses[grasps_idx])

    @staticmethod
    def vis_rgbd_data(data: RgbdDataNp, K: np.ndarray = None):
        RerunViewer.add_rgb("rgb", data.rgb)
        RerunViewer.add_depth("depth", data.depth)
        RerunViewer.add_heatmap("heatmap", data.rgb, data.heatmap)
        if data.poses is not None:
            RerunViewer.add_poses("poses", data.rgb, data.poses, K)
        return

    @staticmethod
    def add_o3d_pointcloud(name: str, pointcloud: o3d.geometry.PointCloud, radii: float = None):
        points = np.asanyarray(pointcloud.points)
        colors = np.asanyarray(pointcloud.colors) if pointcloud.has_colors() else None
        colors_uint8 = (colors * 255).astype(np.uint8) if pointcloud.has_colors() else None
        RerunViewer.add_np_pointcloud(name, points, colors_uint8, radii)
        return

    @staticmethod
    def add_np_pointcloud(
        name: str, points: np.ndarray, colors_uint8: np.ndarray = None, radii: float = None
    ):
        rr_points = rr.Points3D(positions=points, colors=colors_uint8, radii=radii)
        rr.log(name, rr_points)
        return

    @staticmethod
    def add_grasps(name: str, grasp_poses: np.ndarray, color=[0.0, 1.0, 0.0]):
        grasps_trimesh = create_markers_multiple(
            grasp_poses, color, axis_frame=True, highlight_first=True
        )
        RerunViewer.add_mesh_list_trimesh(name, grasps_trimesh)
        return

    @staticmethod
    def add_axis(name: str, pose: np.ndarray, size: float = 0.004):
        mesh = trimesh.creation.axis(origin_size=size, transform=pose)
        RerunViewer.add_mesh_trimesh(name, mesh)
        return

    @staticmethod
    def add_aabb(name: str, centers: np.ndarray, extents: np.ndarray):
        rr.log(name, rr.Boxes3D(centers=centers, sizes=extents))
        return

    @staticmethod
    def add_mesh_trimesh(name: str, mesh: trimesh.Trimesh):
        rr_mesh = rr.Mesh3D(
            vertex_positions=mesh.vertices,
            vertex_colors=mesh.visual.vertex_colors,
            vertex_normals=mesh.vertex_normals,
            indices=mesh.faces,
        )
        rr.log(name, rr_mesh)
        return

    @staticmethod
    def add_mesh_list_trimesh(name: str, meshes: List[trimesh.Trimesh]):
        for i, mesh in enumerate(meshes):
            RerunViewer.add_mesh_trimesh(name + f"/{i}", mesh)
        return

    @staticmethod
    def add_rgb(name: str, rgb_uint8: np.ndarray):
        rr.log(name, rr.Image(rgb_uint8))

    @staticmethod
    def add_depth(name: str, detph: np.ndarray):
        rr.log(name, rr.DepthImage(detph))

    @staticmethod
    def add_heatmap(name: str, rgb: np.ndarray, heatmap: np.ndarray):
        net_heatmap_vis = heatmaps.visualize_heatmap(rgb, heatmap, with_peaks=True)
        rr.log(name, rr.Image(net_heatmap_vis))

    @staticmethod
    def add_poses(name: str, rgb: np.ndarray, poses: np.ndarray, K: np.ndarray):
        _poses = [Pose(camera_T_object=pose) for pose in poses]
        poses_vis = pose_utils.visualize_poses(rgb, _poses, K)
        rr.log(name, rr.Image(poses_vis))

    @staticmethod
    def clear():
        rr.log("vis", rr.Clear(recursive=True))
        return


class Open3DViewer:
    def __init__(self) -> None:
        self.app = o3d.visualization.gui.Application.instance  # type: ignore
        self.app.initialize()
        self.vis = o3d.visualization.O3DVisualizer()  # type: ignore
        self.vis.show_skybox(False)
        self.app.add_window(self.vis)
        frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.add_geometries("camera_frame", frame_mesh)
        self.o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=ZED2HALF_PARAMS.width,
            height=ZED2HALF_PARAMS.height,
            intrinsic_matrix=ZED2HALF_PARAMS.K,
        )
        return

    def add_geometries(self, name, geoms):
        if isinstance(geoms, list):
            for i, geom in enumerate(geoms):
                self.vis.add_geometry(name + str(i), geom)
        elif isinstance(geoms, dict):
            for i, geom in enumerate(geoms.values()):
                self.vis.add_geometry(name + str(i), geom)
        else:
            self.vis.add_geometry(name, geoms)
        return

    def add_trimesh_obj(self, name, trimesh_obj, transform):
        mesh = trimesh_obj.as_open3d
        mesh.transform(transform)
        self.add_geometries(name, mesh)
        return

    def add_grasps(self, name, grasps, grasp_color=[0, 255, 0]):
        grasps_trimesh = create_markers_multiple(grasps, grasp_color, axis_frame=False)
        color = np.array(grasp_color) / 255
        grasps_markers = [obj.as_open3d.paint_uniform_color(color) for obj in grasps_trimesh]
        self.add_geometries(name, grasps_markers)
        return

    def add_pointcloud_np(self, name, pointcloud, idx=0):
        pointcloud_marker = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointcloud))
        self.add_pointcloud_o3d(name, pointcloud_marker, idx)

    def add_pointcloud_o3d(self, name, pointcloud, idx=0):
        color = np.array([*COLOR_PALETTE[idx % len(COLOR_PALETTE)], 1.0])
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.base_color = color
        material.point_size = 5
        self.vis.add_geometry(name + f"_{idx}", pointcloud, material)
        return

    def add_pcd_from_rgbd(self, rgb, depth):
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.o3d_camera_intrinsic)
        self.vis.add_geometry("raw_pcd", pcd)
        return

    def run_vis(self):
        # self.vis.reset_camera_to_default()
        extrinsic = np.eye(4)
        self.vis.setup_camera(self.o3d_camera_intrinsic, extrinsic)
        self.app.run()
        return


class Open3DOfflineRenderer:
    def __init__(
        self, width: int = ZED2HALF_PARAMS.width, height: int = ZED2HALF_PARAMS.height
    ) -> None:
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        self.reset()

    def add_grasps(self, grasps: List[trimesh.Trimesh], base_name="grasp", color=[0.0, 1.0, 0.0]):
        for idx, grasp_trimesh in enumerate(grasps):
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            name = f"{base_name}_{idx}"
            self.renderer.scene.remove_geometry(name)
            grasp_o3d = copy.deepcopy(grasp_trimesh.as_open3d)
            grasp_o3d.paint_uniform_color(color)
            self.renderer.scene.add_geometry(name, grasp_o3d, mat)

    def add_pointcloud(self, pcd_np, name: str = "object_pc", color: Optional[Tuple] = None):
        """
        Places a point cloud in the scene, if there is already one with the same name, replaces it
        """
        if len(pcd_np) == 0:  # Don't add empty pcs
            return
        self.renderer.scene.remove_geometry(name)
        point_colors = color_pcd(pcd_np, color=color)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_np))
        # Set colors
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        # Default material
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        # Add to scene
        self.renderer.scene.add_geometry(name, pcd, mat)

    def reset(self):
        self.renderer.scene.camera.look_at([0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.0, 0.0, 1.0])
        # self.renderer.scene.set_background()

    def render(self):
        """
        Renders the scene to a numpy array
        """
        img_o3d = self.renderer.render_to_image()
        return np.asarray(img_o3d)


def create_gripper_marker(color=[0, 0, 255], gripper_width=0.08, tube_radius=0.002, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.
    It is represented in "hand_link" frame (i.e. origin is at the wrist)
    palm frame: += 0.0624 in z direction
    ttip frame: += 0.1034 in z direction

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    left_y = -0.5 * (gripper_width + tube_radius)
    right_y = 0.5 * (gripper_width + tube_radius)
    mid_z = 0.0624 - (0.5 * tube_radius)
    top_z = 0.1124
    cfr = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [0.0, right_y, mid_z],
            [0.0, right_y, top_z],
        ],
    )
    cfl = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [0.0, left_y, mid_z],
            [0.0, left_y, top_z],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections, segment=[[0, 0, 0], [0, 0, mid_z]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[[0, left_y, mid_z], [0, right_y, mid_z]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp


def create_markers(hand_pose, color, tube_radius=0.002, axis_frame: bool = True):
    """A single gripper marker"""
    palm_pose = hand_pose.copy()
    palm_pose[:3, 3] += 0.0624 * hand_pose[:3, 2]
    position_marker = trimesh.creation.axis(transform=palm_pose, origin_size=0.002)  # type: ignore
    stick_marker = create_gripper_marker(color, tube_radius=tube_radius)  # type: ignore
    stick_marker.apply_transform(hand_pose)  # type: ignore
    return np.array([stick_marker] + ([position_marker] if axis_frame else [])).flatten().tolist()


def create_markers_multiple(
    hand_poses: np.ndarray, color: list, axis_frame: bool = True, highlight_first: bool = False
):
    """Multiple gripper markers"""
    res = (
        np.array([create_markers(t, color, axis_frame=axis_frame) for t in hand_poses])
        .flatten()
        .tolist()
    )
    if highlight_first and len(hand_poses) > 1:
        first_marker = create_markers(
            hand_poses[0], color, tube_radius=0.006, axis_frame=axis_frame
        )
        res[0] = first_marker[0]
    return res


def create_markers_multiple_fat(hand_poses: np.ndarray, color: list, axis_frame: bool = True):
    res = (
        np.array(
            [create_markers(t, color, tube_radius=0.005, axis_frame=axis_frame) for t in hand_poses]
        )
        .flatten()
        .tolist()
    )
    return res
