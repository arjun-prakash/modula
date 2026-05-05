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

# CIFAR-10 MLP runs
# uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 --hidden-sizes 256 512 1024 2048 --methods adam --use-wandb --wandb-project cifar10-width-sweep --wandb-group "Adam baseline"
# uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 --hidden-sizes 256 512 1024 2048 --methods manifold_online --manifold-scaling fan_ratio --use-wandb --wandb-project cifar10-width-sweep --wandb-group "Manifold Online scale factor"

uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 --hidden-sizes 256 512 1024 2048 --methods manifold_online --manifold-scaling fan_max --use-wandb --wandb-project cifar10-width-sweep --wandb-group "Manifold Online max factor"
uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 --hidden-sizes 256 512 1024 2048 --methods manifold_online --manifold-scaling none --use-wandb --wandb-project cifar10-width-sweep --wandb-group "Manifold Online no factor"