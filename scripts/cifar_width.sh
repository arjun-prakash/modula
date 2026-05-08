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

# uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 --hidden-sizes 256 512 1024 2048 --methods manifold_online --manifold-scaling fan_max --use-wandb --wandb-project cifar10-width-sweep --wandb-group "Manifold Online max factor"
# uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 --hidden-sizes 256 512 1024 2048 --methods manifold_online --manifold-scaling none --use-wandb --wandb-project cifar10-width-sweep --wandb-group "Manifold Online no factor"

# uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 --hidden-sizes 256 512 1024 --methods manifold_online --manifold-scaling fan_ratio --trunk wide3 --linear-normalizations rms_radius --use-wandb --wandb-project cifar10-width-sweep --wandb-group "Manifold Online scale factor non-square"
# uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 --hidden-sizes 256 512 1024 --methods manifold_online --manifold-scaling fan_max --trunk wide3 --use-wandb --wandb-project cifar10-width-sweep --wandb-group "Manifold Online max factor non-square"
# uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 --hidden-sizes 256 512 1024 --methods manifold_online --manifold-scaling none --trunk wide3 --use-wandb --wandb-project cifar10-width-sweep --wandb-group "Manifold Online max factor non-square"

# uv run benchmark/cifar10_mlp.py --steps 10000 --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 --hidden-sizes 256 512 1024 --methods adam --trunk wide3 --use-wandb --wandb-project cifar10-width-sweep --wandb-group "Adam baseline non-square"

uv run benchmark/cifar10_mlp.py --steps 10000 \
  --learning-rates 6.1e-5 1.22e-4 2.44e-4 4.88e-4 9.76e-4 1.95e-3 3.9e-3 7.81e-3 1.56e-2 3.12e-2 6.25e-2\
  --hidden-sizes 256 512 1024 2048 \
  --methods sgd \
  --linear-normalizations sp \
  --use-wandb \
  --wandb-project cifar10-width-sweep \
  --wandb-group "SGD baseline new lr"
