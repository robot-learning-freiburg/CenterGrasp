#!/usr/bin/env bash

python scripts/evaluate_shape.py --seed 123 --headless --log-wandb
python scripts/evaluate_shape.py --seed 456 --headless --log-wandb
python scripts/evaluate_shape.py --seed 789 --headless --log-wandb