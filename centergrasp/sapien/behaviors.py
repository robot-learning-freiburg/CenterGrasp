import numpy as np
import spatialmath as sm
from centergrasp.sapien.sapien_envs import PickEnv
from centergrasp.pipelines.planners import MPLibPlanner
from centergrasp.pipelines.centergrasp_pipeline import CenterGraspPipeline

VERTICAL_EE_R = np.array(
    [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]
)


class PickBehavior:
    def __init__(
        self,
        environment: PickEnv,
        pipeline: CenterGraspPipeline,
        planner: MPLibPlanner,
        dt: float = 1 / 240,
    ):
        self.environment = environment
        self.pipeline = pipeline
        self.planner = planner
        self.dt = dt
        self.lift_position = np.array([0.4, -0.2, 0.6])
        self.place_position = np.array([0.0, -0.4, 0.6])
        self.static_obstacle_pc = self.environment.static_scene_pc
        return

    def plan_traj(self, bTtarget: sm.SE3, use_point_cloud: bool = False):
        joint_states = self.environment.sapien_robot.get_joint_state()
        trajectory = self.planner.plan_traj(
            bTtarget, joint_states, dt=self.dt, use_point_cloud=use_point_cloud
        )
        return trajectory

    def move_to_pose(self, bTtarget: sm.SE3, use_point_cloud: bool = False) -> bool:
        trajectory = self.plan_traj(bTtarget, use_point_cloud=use_point_cloud)
        if trajectory is None:
            return False
        self.environment.execute_traj(trajectory)
        return True

    def run(self) -> dict:
        run_complete = self.run_behavior()
        self.environment.reset_robot()
        grasp_success = self.environment.evaluate_success()
        info = {"run_complete": run_complete, "grasp_success": grasp_success}
        return info

    def run_behavior(self) -> bool:
        # Get observations
        obs = self.environment.get_obs()

        # Add pointcloud to planner
        obs_pc = self.environment.get_scene_pc(obs)
        obstacle_pc = np.concatenate([obs_pc, self.static_obstacle_pc], axis=0)
        self.planner.update_obstacle_pc(obstacle_pc)

        # Run pipeline
        bTgrasp_list = self.pipeline.predict_grasp(obs)
        if len(bTgrasp_list) == 0:
            return False

        # Move to pregrasp
        for bTgrasp in bTgrasp_list:
            bTpregrasp = bTgrasp * sm.SE3.Trans([0, 0, -0.08])
            success = self.move_to_pose(bTpregrasp, use_point_cloud=True)
            if success:
                break
        if not success:
            return False

        # Move to grasp
        self.planner.update_obstacle_pc(self.static_obstacle_pc)
        success = self.move_to_pose(bTgrasp, use_point_cloud=False)
        if not success:
            return False

        # Close gripper
        self.environment.close_gripper()

        # Move up
        bTpostgrasp = sm.SE3.Trans([0.0, 0.0, 0.2]) * bTgrasp
        success = self.move_to_pose(bTpostgrasp, use_point_cloud=True)
        if not success:
            return False

        # Rotate to vertical orientation
        bTpostgrasp_vertical = sm.SE3.Rt(R=VERTICAL_EE_R, t=bTpostgrasp.t)
        success = self.move_to_pose(bTpostgrasp_vertical, use_point_cloud=True)
        if not success:
            return False

        # Move to lift position
        self.planner.update_obstacle_pc(obstacle_pc)
        bTlift = sm.SE3.Rt(R=VERTICAL_EE_R, t=self.lift_position)
        success = self.move_to_pose(bTlift, use_point_cloud=True)
        if not success:
            return False

        # Place
        bTplace = sm.SE3.Rt(R=VERTICAL_EE_R, t=self.place_position)
        success = self.move_to_pose(bTplace, use_point_cloud=True)
        if not success:
            return False

        # Open gripper
        self.environment.open_gripper()
        return True


class PickBehaviorGripper(PickBehavior):
    def set_pose(self, bTtarget: sm.SE3):
        self.environment.sapien_robot.set_root_pose(bTtarget.A)
        return

    def move_to_pose(self, bTtarget: sm.SE3):
        speed = 0.1
        delta_space = bTtarget.t - self.environment.sapien_robot.get_root_pose()[:3, 3]
        distance = np.linalg.norm(delta_space)
        direction = delta_space / distance
        velocity = direction * speed
        time = distance / speed
        self.environment.sapien_robot.set_root_velocity(velocity, np.zeros(3))
        self.environment.step_physics(time)
        self.environment.sapien_robot.set_root_velocity(np.zeros(3), np.zeros(3))
        return

    def run_behavior(self) -> bool:
        # Get observations
        obs = self.environment.get_obs()

        # Run pipeline
        bTgrasp_list = self.pipeline.predict_grasp(obs)
        if len(bTgrasp_list) == 0:
            return False

        # Move to pregrasp
        bTgrasp = bTgrasp_list[0]
        bTpregrasp = bTgrasp * sm.SE3.Trans([0, 0, -0.08])
        self.set_pose(bTpregrasp)

        # Move to grasp
        self.move_to_pose(bTgrasp)

        # Close gripper
        self.environment.close_gripper()

        # Move up
        bTpostgrasp = sm.SE3.Trans([0.0, 0.0, 0.4]) * bTgrasp
        self.move_to_pose(bTpostgrasp)

        # Place
        bTplace = sm.SE3.Rt(R=bTpostgrasp.R, t=self.place_position, check=False)
        self.move_to_pose(bTplace)

        # Open gripper
        self.environment.open_gripper()
        return True
