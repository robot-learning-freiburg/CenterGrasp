#!/usr/bin/env bash

python scripts/evaluate.py --env giga_packed --method centergrasp --seed 123 --headless --log-wandb
python scripts/evaluate.py --env giga_pile --method centergrasp --seed 123 --headless --log-wandb

python scripts/evaluate.py --env giga_packed --method centergrasp_noicp --seed 123 --headless --log-wandb
python scripts/evaluate.py --env giga_pile --method centergrasp_noicp --seed 123 --headless --log-wandb

python scripts/evaluate.py --env giga_packed --method giga --seed 123 --headless --log-wandb
python scripts/evaluate.py --env giga_pile --method giga --seed 123 --headless --log-wandb


python scripts/evaluate.py --env giga_packed --method centergrasp --seed 456 --headless --log-wandb
python scripts/evaluate.py --env giga_pile --method centergrasp --seed 456 --headless --log-wandb

python scripts/evaluate.py --env giga_packed --method centergrasp_noicp --seed 456 --headless --log-wandb
python scripts/evaluate.py --env giga_pile --method centergrasp_noicp --seed 456 --headless --log-wandb

python scripts/evaluate.py --env giga_packed --method giga --seed 456 --headless --log-wandb
python scripts/evaluate.py --env giga_pile --method giga --seed 456 --headless --log-wandb

python scripts/evaluate.py --env giga_packed --method centergrasp --seed 789 --headless --log-wandb
python scripts/evaluate.py --env giga_pile --method centergrasp --seed 789 --headless --log-wandb

python scripts/evaluate.py --env giga_packed --method centergrasp_noicp --seed 789 --headless --log-wandb
python scripts/evaluate.py --env giga_pile --method centergrasp_noicp --seed 789 --headless --log-wandb

python scripts/evaluate.py --env giga_packed --method giga --seed 789 --headless --log-wandb
python scripts/evaluate.py --env giga_pile --method giga --seed 789 --headless --log-wandb