import mplib
import numpy as np
import spatialmath as sm
from typing import List, Tuple
from centergrasp.configs import Directories
from centergrasp.sapien.sapien_utils import Trajectory

PANDA_LINK_NAMES = [
    "panda_link0",
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_link8",
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
]

PANDA_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]

PANDA_JOINT_VEL_LIMITS = np.array(
    [
        2.1750,
        2.1750,
        2.1750,
        2.1750,
        2.6100,
        2.6100,
        2.6100,
    ]
)

# Time constant, i.e. how much time to reach velocity limits [seconds]
PANDA_ACC_TAU = 0.25


class MPLibPlanner:
    def __init__(self) -> None:
        franka_urdf_path = str(Directories.FRANKA / "panda.urdf")
        qd_limits = PANDA_JOINT_VEL_LIMITS / 2
        qdd_limits = qd_limits / PANDA_ACC_TAU
        self._planner = mplib.Planner(
            urdf=franka_urdf_path,
            srdf=franka_urdf_path.replace(".urdf", ".srdf"),
            user_link_names=PANDA_LINK_NAMES,
            user_joint_names=PANDA_JOINT_NAMES,
            move_group="panda_hand",
            joint_vel_limits=qd_limits,
            joint_acc_limits=qdd_limits,
        )

    def get_ee_pose(self, joint_state: np.ndarray) -> sm.SE3:
        ee_index = self._planner.link_name_2_idx[self._planner.move_group]
        self._planner.robot.set_qpos(joint_state, True)
        self._planner.pinocchio_model.compute_forward_kinematics(joint_state)
        pose7d = self._planner.pinocchio_model.get_link_pose(ee_index)
        pose = sm.SE3.Rt(t=pose7d[:3], R=sm.base.q2r(pose7d[3:]))
        return pose

    def update_obstacle_pc(self, pc: np.ndarray):
        self._planner.update_point_cloud(pc)
        return

    def ik(self, target_pose: sm.SE3, joint_state: np.ndarray) -> Tuple[str, List[np.ndarray]]:
        target_pose_7d = np.concatenate([target_pose.t, sm.base.r2q(target_pose.R)])
        result, output = self._planner.IK(target_pose_7d, joint_state)
        return result, output

    def plan_traj(
        self, target_pose: sm.SE3, joint_state: np.ndarray, dt: float, use_point_cloud: bool = False
    ) -> Trajectory:
        target_pose_7d = np.concatenate([target_pose.t, sm.base.r2q(target_pose.R)])
        result = self._planner.plan_screw(
            target_pose_7d, joint_state, time_step=dt, use_point_cloud=use_point_cloud
        )
        if result["status"] != "Success":
            result = self._planner.plan(
                target_pose_7d, joint_state, time_step=dt, use_point_cloud=use_point_cloud
            )
            if result["status"] != "Success":
                print(result["status"])
                return None
        traj = Trajectory(
            time=result["time"],
            position=result["position"],
            velocity=result["velocity"],
            acceleration=result["acceleration"],
        )
        return traj
