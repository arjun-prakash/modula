#!/bin/bash

#SBATCH --output=slurm_logs/wandb_%j.out # Standard output log
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH -p gpu --gres=gpu:1 
#SBATCH --constraint=geforce3090


# Load Python module

module load python
#export PYTHONPATH=$(pwd):$PYTHONPATH

# CIFAR-100 examples
# uv run benchmark/cifar100.py --steps 5000 --learning-rates 1e-3 --methods adam --use-wandb --wandb-project cifar100-benchmark
# uv run benchmark/cifar100.py --steps 5000 --learning-rates 1e-3 --methods manifold manifold_online manifold_admm --use-wandb --wandb-project cifar100-benchmark

# CIFAR-10 default runs
# uv run benchmark/cifar10.py --steps 5000 --learning-rates 1e-3 --methods adam --use-wandb --wandb-project cifar10-benchmark
# uv run benchmark/cifar10.py --steps 5000 --learning-rates 1e-3 --methods manifold manifold_online manifold_admm --use-wandb --wandb-project cifar10-benchmark
