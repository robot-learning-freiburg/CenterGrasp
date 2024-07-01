import os
import itertools

envs = ["giga_packed", "giga_pile", "ycb_packed", "ycb_pile"]
seeds = [123, 456, 789]

for env, seed in itertools.product(envs, seeds):
    cmd = f"python scripts/evaluate_shape.py --env {env} --seed {seed} --headless --log-wandb"  # noqa E501
    os.system(cmd)
