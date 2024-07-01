import numpy as np
import open3d as o3d
import spatialmath as sm
import sapien.core as sapien
import mplib.pymp.fcl as fcl
from centergrasp.configs import Directories, GripperKinTree


class FrankaGripperFCL:
    def __init__(self) -> None:
        self.handTcyl = sm.SE3.Rt(R=sm.SO3.Rx(90, "deg"), t=[0.0, 0.0, 0.1034])
        self.kin_tree = GripperKinTree()

        # Get meshes
        hand_meshes_path = Directories.FRANKA / "meshes/collision/"
        hand_path = hand_meshes_path / "hand.stl.convex.stl"
        left_finger_path = hand_meshes_path / "finger.stl.convex.stl"
        right_finger_path = hand_meshes_path / "finger.stl.convex.stl"
        hand_fcl = fcl.load_mesh_as_Convex(str(hand_path), scale=[1, 1, 1])
        left_finger_fcl = fcl.load_mesh_as_Convex(str(left_finger_path), scale=[1, 1, 1])
        right_finger_fcl = fcl.load_mesh_as_Convex(str(right_finger_path), scale=[1, 1, 1])
        cylinder_fcl = fcl.Cylinder(radius=0.012, lz=0.08)

        # Set collision objects
        self.collision_objects = [
            fcl.CollisionObject(hand_fcl, np.zeros(3), np.array([1, 0, 0, 0])),
            fcl.CollisionObject(left_finger_fcl, np.zeros(3), np.array([1, 0, 0, 0])),
            fcl.CollisionObject(right_finger_fcl, np.zeros(3), np.array([1, 0, 0, 0])),
        ]
        self.collision_cylinder = fcl.CollisionObject(
            cylinder_fcl, np.zeros(3), np.array([1, 0, 0, 0])
        )
        self.move_to_grasp_pose(sm.SE3())
        self.collision_request = fcl.CollisionRequest()
        return

    def move_to_grasp_pose(self, wThand: sm.SE3):
        # Move hand
        hand_pose = sapien.Pose.from_transformation_matrix(wThand.A)
        hand_pose7d = np.concatenate([hand_pose.p, hand_pose.q])
        self.collision_objects[0].set_transformation(hand_pose7d)
        # Move left finger
        wTlf = wThand.A @ self.kin_tree.leftfinger
        leftfinger_pose = sapien.Pose.from_transformation_matrix(wTlf)
        leftfinger_pose7d = np.concatenate([leftfinger_pose.p, leftfinger_pose.q])
        self.collision_objects[1].set_transformation(leftfinger_pose7d)
        # Move right finger
        wTrf = wThand.A @ self.kin_tree.rightfinger
        rightfinger_pose = sapien.Pose.from_transformation_matrix(wTrf)
        rightfinger_pose7d = np.concatenate([rightfinger_pose.p, rightfinger_pose.q])
        self.collision_objects[2].set_transformation(rightfinger_pose7d)
        # Move cylinder
        wTcyl = wThand * self.handTcyl
        cyl_pose = sapien.Pose.from_transformation_matrix(wTcyl.A)
        cyl_pose7d = np.concatenate([cyl_pose.p, cyl_pose.q])
        self.collision_cylinder.set_transformation(cyl_pose7d)
        return

    def is_colliding(self, collision_obj: fcl.CollisionObject):
        # Check collision
        results = [
            fcl.collide(obj, collision_obj, self.collision_request)
            for obj in self.collision_objects
        ]
        collisions = [r.is_collision() for r in results]
        return any(collisions)

    def is_empty(self, collision_obj: fcl.CollisionObject):
        result = fcl.collide(self.collision_cylinder, collision_obj, self.collision_request)
        return not result.is_collision()

    def check_valid_grasp(self, wTgrasp: sm.SE3, collision_obj: fcl.CollisionObject):
        # Move the gripper to the grasp pose
        self.move_to_grasp_pose(wTgrasp)

        # Check grasp not empty
        if self.is_empty(collision_obj):
            return False

        # Check gripper not colliding with scene
        if self.is_colliding(collision_obj):
            return False

        # Else, gripper is valid
        return True

    def visulalize_o3d(self, pointcloud: np.ndarray = None):
        """
        Note: to visualize the sapien gripper, use the script: 'check_gripper.py'
        """
        frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        o3d_vis_list = [frame_mesh]
        # Gripper pcd
        for i, c_obj in enumerate(self.collision_objects):
            vertices = c_obj.get_collision_geometry().get_vertices()
            rotation = c_obj.get_rotation()
            translation = c_obj.get_translation()
            transform = sm.SE3.Rt(R=rotation, t=translation, check=False)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
            pcd.transform(transform.A)
            o3d_vis_list.append(pcd)

        if pointcloud is not None:
            obj_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointcloud))
            o3d_vis_list.append(obj_pcd)

        # Visulize
        o3d.visualization.draw_geometries(o3d_vis_list)
        return


class FrankaKinematics:
    def __init__(self) -> None:
        self.pandahandTee = sm.SE3.Trans([0, 0, 0.1034])
        pass

    def goal_reachable(
        self,
        grasp_pose: np.ndarray,
        joint_state: np.ndarray,
        wTbase: sm.SE3,
        wTcam: sm.SE3,
    ):
        """
        grasp_pose: goal hand pose, in camera frame
        """
        assert len(joint_state) == 7
        success = True  # TODO
        return success
