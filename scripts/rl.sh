#!/bin/bash

#SBATCH --output=slurm_logs/wandb_%j.out # Standard output log
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1

# Load Python module

module load python
# export PYTHONPATH=$(pwd):$PYTHONPATH
# source .venv/bin/activate

#python examples/continual_cifar100.py --num-tasks 100
uv run examples/drone.py --episodes 500