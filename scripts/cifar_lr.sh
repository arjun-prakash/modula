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
uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 --methods adam --use-wandb --wandb-project cifar10-lr-sweep --wandb-group "Adam baseline"
uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 --methods muon --muon-scaling fan_ratio --use-wandb --wandb-project cifar10-lr-sweep --wandb-group "Muon Moonlight"
uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 --methods muon --muon-scaling fan_max --use-wandb --wandb-project cifar10-lr-sweep --wandb-group "Muon K. Jordan"

uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 --methods manifold --use-wandb --manifold-scaling fan_max --wandb-project cifar10-lr-sweep --wandb-group "Manifold max factor"
uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 --methods manifold --use-wandb --manifold-scaling fan_ratio --wandb-project cifar10-lr-sweep --wandb-group "Manifold ratio factor"

uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 --methods manifold_online --use-wandb --manifold-scaling fan_max --wandb-project cifar10-lr-sweep --wandb-group "Manifold Online max factor"
uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 --methods manifold_online --use-wandb --manifold-scaling fan_ratio --wandb-project cifar10-lr-sweep --wandb-group "Manifold Online ratio factor"

uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 --methods manifold_admm --use-wandb --manifold-scaling fan_max --wandb-project cifar10-lr-sweep --wandb-group "Manifold ADMM max factor"
uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 --methods manifold_admm --use-wandb --manifold-scaling fan_ratio --wandb-project cifar10-lr-sweep --wandb-group "Manifold ADMM ratio factor"

uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 --methods manifold_online --use-wandb --manifold-scaling none --wandb-project cifar10-lr-sweep --wandb-group "Manifold Online no scale"

# CIFAR-10 runs

#uv run benchmark/cifar10.py --steps 5000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 5 10 --methods adam --use-wandb --wandb-project cifar10-lr-sweep-temp
#uv run benchmark/cifar10.py --steps 5000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 5 10 --methods manifold manifold_online manifold_admm --use-wandb --wandb-project cifar10-lr-sweep-temp
