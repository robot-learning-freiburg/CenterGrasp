import numpy as np
import spatialmath as sm
import sapien.core as sapien
import mplib.pymp.fcl as fcl
from dataclasses import dataclass


@dataclass
class StickGripperKinTree:
    wThand: sm.SE3

    @property
    def palm_center(self) -> sm.SE3:
        # the center of the palm obb
        return self.wThand * sm.SE3.Trans(0.0, 0.0, 0.0574)

    @property
    def leftfinger(self) -> sm.SE3:
        # the center of the left finger obb
        return self.palm_center * sm.SE3.Trans(0.0, -0.045, 0.03)

    @property
    def rightfinger(self) -> sm.SE3:
        # the center of the right finger obb
        return self.palm_center * sm.SE3.Trans(0.0, 0.045, 0.03)

    @property
    def inner_center(self) -> sm.SE3:
        # the center of the inner workspace obb
        return self.palm_center * sm.SE3.Trans(0.0, 0.0, 0.03)


class StickGripperFCL:
    def __init__(self) -> None:
        # Collision Geometries
        palm_fcl = fcl.Box(x=0.02, y=0.10, z=0.01)
        left_finger_fcl = fcl.Box(x=0.02, y=0.01, z=0.05)
        right_finger_fcl = fcl.Box(x=0.02, y=0.01, z=0.05)
        inner_fcl = fcl.Box(x=0.02, y=0.08, z=0.05)

        # Set collision objects
        self.collision_objects = [
            fcl.CollisionObject(palm_fcl, np.zeros(3), np.array([1, 0, 0, 0])),
            fcl.CollisionObject(left_finger_fcl, np.zeros(3), np.array([1, 0, 0, 0])),
            fcl.CollisionObject(right_finger_fcl, np.zeros(3), np.array([1, 0, 0, 0])),
        ]
        self.inner_ws = fcl.CollisionObject(inner_fcl, np.zeros(3), np.array([1, 0, 0, 0]))
        self.move_to_grasp_pose(sm.SE3())
        self.collision_request = fcl.CollisionRequest()
        return

    def move_to_grasp_pose(self, wThand: sm.SE3):
        kin_tree = StickGripperKinTree(wThand)
        # Move palm
        palm_pose = sapien.Pose.from_transformation_matrix(kin_tree.palm_center.A)
        palm_pose7d = np.concatenate([palm_pose.p, palm_pose.q])
        self.collision_objects[0].set_transformation(palm_pose7d)
        # Move left finger
        leftfinger_pose = sapien.Pose.from_transformation_matrix(kin_tree.leftfinger.A)
        leftfinger_pose7d = np.concatenate([leftfinger_pose.p, leftfinger_pose.q])
        self.collision_objects[1].set_transformation(leftfinger_pose7d)
        # Move right finger
        rightfinger_pose = sapien.Pose.from_transformation_matrix(kin_tree.rightfinger.A)
        rightfinger_pose7d = np.concatenate([rightfinger_pose.p, rightfinger_pose.q])
        self.collision_objects[2].set_transformation(rightfinger_pose7d)
        # Move inner workspace
        cyl_pose = sapien.Pose.from_transformation_matrix(kin_tree.inner_center.A)
        cyl_pose7d = np.concatenate([cyl_pose.p, cyl_pose.q])
        self.inner_ws.set_transformation(cyl_pose7d)
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
        result = fcl.collide(self.inner_ws, collision_obj, self.collision_request)
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


if __name__ == "__main__":
    from centergrasp.visualization import RerunViewer as RV

    RV()
    GRIPPER_FCL = StickGripperFCL()

    # make a 3d uniform grid point cloud
    x_ = np.linspace(-0.2, 0.2, 100)
    y_ = np.linspace(-0.2, 0.2, 100)
    z_ = np.linspace(-0.2, 0.2, 100)
    x, y, z = np.meshgrid(x_, y_, z_, indexing="ij")
    pcd_np = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    collision_mask = []
    inner_mask = []
    for p in pcd_np:
        octree = fcl.OcTree(vertices=p, resolution=0.0001)
        octree_fcl = fcl.CollisionObject(octree, np.zeros(3), np.array([1, 0, 0, 0]))
        collision = GRIPPER_FCL.is_colliding(octree_fcl)
        inner = not GRIPPER_FCL.is_empty(octree_fcl)
        collision_mask.append(collision)
        inner_mask.append(inner)

    collision_mask = np.array(collision_mask)
    inner_mask = np.array(inner_mask)
    collision_pcd = pcd_np[collision_mask]
    inner_pcd = pcd_np[inner_mask]
    out_pcd = pcd_np[~collision_mask & ~inner_mask]
    colors_collision = np.array([255, 0, 0])[None, :].repeat(collision_pcd.shape[0], axis=0)
    colors_inner = np.array([0, 255, 0])[None, :].repeat(inner_pcd.shape[0], axis=0)

    RV.add_axis("vis/origin", pose=np.eye(4))
    RV.add_np_pointcloud(
        "vis/collision_pcd", collision_pcd, colors_uint8=colors_collision, radii=0.002
    )
    RV.add_np_pointcloud("vis/inner_pcd", inner_pcd, colors_uint8=colors_inner, radii=0.002)
    RV.add_np_pointcloud("vis/out_pcd", out_pcd)
    print("Done")
