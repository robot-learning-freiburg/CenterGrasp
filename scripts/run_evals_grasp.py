import os
import itertools

robot_types = ["franka", "gripper"]
envs = ["giga_packed", "giga_pile", "ycb_packed", "ycb_pile"]
methods = ["centergrasp", "centergrasp_noicp", "giga"]
seeds = [123, 456, 789]

for robot_type, env, method, seed in itertools.product(robot_types, envs, methods, seeds):
    cmd = f"python scripts/evaluate.py --robot-type {robot_type} --env {env} --method {method} --seed {seed} --headless --log-wandb"  # noqa E501
    os.system(cmd)
