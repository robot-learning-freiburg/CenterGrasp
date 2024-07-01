import numpy as np
import spatialmath as sm
from centergrasp.configs import Directories
from centergrasp.pipelines.planners import MPLibPlanner
import centergrasp.sapien.sapien_utils as sapien_utils

Q_READY = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0, 0])
EE_START = np.array(
    [
        [8.4088784e-01, -4.8657018e-04, 5.4120922e-01, 2.8628591e-02],
        [5.2415123e-03, -9.9994540e-01, -9.0428386e-03, -1.4148998e-01],
        [5.4118407e-01, 1.0440767e-02, -8.4083939e-01, 6.7434818e-01],
        [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
    ]
)

# Init Sapien
physics_dt = 1 / 240
engine, renderer, scene = sapien_utils.init_sapien(False, physics_dt)
viewer = sapien_utils.init_viewer(scene, renderer, show_axes=True)
_ = sapien_utils.init_lights(scene)
loader = scene.create_urdf_loader()
loader.fix_root_link = True
franka_urdf_path = Directories.FRANKA / "panda.urdf"
robot = sapien_utils.add_franka(scene, franka_urdf_path)
scene.step()

target_ee_pose = sm.SE3(EE_START, check=False)
planner = MPLibPlanner()
_, joint_states_goal = planner.ik(target_ee_pose, Q_READY)

for q_goal in joint_states_goal:
    print(q_goal)

    # Go there
    robot.set_qpos(q_goal)
    robot.set_drive_target(q_goal)
    for _ in range(100):
        scene.step()
        scene.update_render()
        viewer.render()

    input("Press Enter to continue...")

    # while not viewer.closed:  # Press key q to quit
    #     scene.step()
