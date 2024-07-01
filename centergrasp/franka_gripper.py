import numpy as np
import open3d as o3d
import spatialmath as sm
from centergrasp.configs import Directories
import centergrasp.se3_utils as utils_se3
from centergrasp.o3d_live_vis import O3dLiveVisualizer


class FrankaGripperO3d:
    # See specs at
    # https://download.franka.de/documents/220010_Product%20Manual_Franka%20Hand_1.2_EN.pdf
    def __init__(self, vis: O3dLiveVisualizer) -> None:
        # Visualizer
        self.vis = vis
        mesh_type = "visual" if vis.vis_flag else "collision"
        hand_meshes_path = Directories.FRANKA / "meshes" / mesh_type

        # Get meshes
        self.frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

        hand_path = hand_meshes_path / "hand.stl"
        hand_mesh = o3d.io.read_triangle_mesh(str(hand_path))
        hand_mesh.compute_vertex_normals()

        left_finger_path = hand_meshes_path / "finger.stl"
        left_finger_mesh = o3d.io.read_triangle_mesh(str(left_finger_path))
        left_finger_mesh.compute_vertex_normals()

        right_finger_path = hand_meshes_path / "finger.stl"
        right_finger_mesh = o3d.io.read_triangle_mesh(str(right_finger_path))
        right_finger_mesh.compute_vertex_normals()

        grasp_pose_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        self.meshes = {
            "hand": hand_mesh,
            "left_finger": left_finger_mesh,
            "right_finger": right_finger_mesh,
            "grasp_pose": grasp_pose_mesh,
        }
        self.pointclouds = None

        self.gripper_width = 0.0
        self.wTgrasp = sm.SE3.Tz(0.103)
        self.init_gripper()

        # Move to origin
        wTgoal = sm.SE3()
        self.set_grasp_pose(wTgoal)

        self.vis.add_and_render(self.meshes)
        return

    def init_gripper(self):
        # Process left finger mesh
        self.meshes["left_finger"].translate((0.0, 0.0, 0.0584))

        # Process right finger mesh
        R = self.meshes["right_finger"].get_rotation_matrix_from_xyz((0, 0, np.pi))
        self.meshes["right_finger"].rotate(R, center=(0, 0, 0))
        self.meshes["right_finger"].translate((0.0, 0.0, 0.0584))

        # Process grasp frame
        self.meshes["grasp_pose"].translate((0.0, 0.0, 0.103))
        return

    def init_pointcloud(self):
        pc_hand = self.meshes["hand"].sample_points_poisson_disk(
            number_of_points=1000, init_factor=5
        )
        pc_left_finger = self.meshes["left_finger"].sample_points_poisson_disk(
            number_of_points=200, init_factor=5
        )
        pc_right_finger = self.meshes["right_finger"].sample_points_poisson_disk(
            number_of_points=200, init_factor=5
        )
        self.pointclouds = {
            "hand": pc_hand,
            "left_finger": pc_left_finger,
            "right_finger": pc_right_finger,
        }
        self.pc_centers = {
            "hand": pc_hand.get_center(),
            "left_finger": pc_left_finger.get_center(),
            "right_finger": pc_right_finger.get_center(),
        }
        # self.vis.add_and_render(self.pointclouds)
        return

    def set_grasp_pose(self, wTgoal):
        transform_op = wTgoal * self.wTgrasp.inv()
        for _, mesh in self.meshes.items():
            mesh.transform(transform_op.A)
        self.wTgrasp = wTgoal
        self.vis.update_and_render(self.meshes)
        if self.pointclouds is not None:
            for _, pc in self.pointclouds.items():
                pc.transform(transform_op.A)
                # self.vis.update_and_render(self.pointclouds)
            for k in self.pc_centers:
                self.pc_centers[k] = transform_op * self.pc_centers[k]
                self.pc_centers[k] = self.pc_centers[k].squeeze()
        return

    def align_to_surface(self, point, normal):
        pose_centered = utils_se3.align_y_to_normal(point, normal)
        pose_aligned = utils_se3.translate_pose_y(pose_centered, -0.035)
        self.set_grasp_pose(pose_aligned)
        return

    def step_rotation(self, rot_deg):
        new_pose = utils_se3.rotate_pose_y(self.wTgrasp, rot_deg)
        self.set_grasp_pose(new_pose)
        return

    def set_gripper_width(self, width):
        delta = width - self.gripper_width
        if delta == 0:
            return

        # Fingers
        rel_t = sm.SE3.Ty(delta / 2)
        transform_op = self.wTgrasp * rel_t * self.wTgrasp.inv()
        self.meshes["left_finger"].translate(transform_op.t)
        self.meshes["right_finger"].translate(-transform_op.t)
        self.vis.update_and_render([self.meshes["left_finger"], self.meshes["right_finger"]])
        self.gripper_width = width
        if self.pointclouds is not None:
            self.pc_centers["left_finger"] += transform_op.t
            self.pc_centers["right_finger"] += -transform_op.t
            self.pointclouds["left_finger"].translate(transform_op.t)
            self.pointclouds["right_finger"].translate(-transform_op.t)
            # self.vis.update_and_render(
            #     [self.pointclouds["left_finger"], self.pointclouds["right_finger"]]
            # )

        # TODO: summarize what you did here, maybe make order
        # Cylinder
        # if "cylinder_mesh" in self.meshes:
        #     self.vis.remove_and_render(self.meshes["cylinder_mesh"])
        self.meshes["cylinder_mesh"] = o3d.geometry.TriangleMesh.create_cylinder(
            radius=0.012, height=width
        )
        self.meshes["cylinder_mesh"].transform(self.wTgrasp.A)
        rel_R = sm.SE3.Rx(90, "deg")
        transform_op = self.wTgrasp * rel_R * self.wTgrasp.inv()
        self.meshes["cylinder_mesh"].rotate(transform_op.R)
        # self.vis.add_and_render(self.meshes["cylinder_mesh"])
        return

    def vector_grasp_to_w(self, v):
        w_v = self.wTgrasp.A @ np.append(v, 1)
        return w_v[:3]

    def vector_w_to_grasp(self, v):
        graspTw = self.wTgrasp.inv()
        gr_v = graspTw * v
        return gr_v

    def center_gripper(self, pc):
        # Get pc in gripper frame
        gr_pc = np.array([self.vector_w_to_grasp(p) for p in pc])

        # Find mean in y direction and move gripper there
        if gr_pc.ndim < 2:
            return
        center = np.mean(gr_pc[:, 1])
        if center < 0:
            return
        new_pose = utils_se3.translate_pose_y(self.wTgrasp, center)
        self.set_grasp_pose(new_pose)
        return
