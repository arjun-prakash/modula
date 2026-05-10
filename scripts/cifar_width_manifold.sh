#!/bin/bash

#SBATCH --output=slurm_logs/wandb_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH -p gpu --gres=gpu:1
#SBATCH --constraint=geforce3090

module load python

LRS="6.1e-5 1.22e-4 2.44e-4 4.88e-4 9.76e-4 1.95e-3 3.9e-3 7.81e-3 1.56e-2 3.12e-2 6.25e-2 1.25e-1 2.5e-1 5e-1 1"

LRS_EDIT="4.88e-4 9.76e-4 1.95e-3 3.9e-3 7.81e-3 1.56e-2 3.12e-2 6.25e-2 1.25e-1 2.5e-1 5e-1"
WIDTHS="256 512 1024 2048"

# this is -7.75, -7.5, -7.25, -7.0, -6.75, -6.5, -6.25 on log2 scale
LRS_PRECISE="0.00464534 0.00552427 0.00656864 0.0078125 0.00929068 0.01104854 0.01313901"


uv run benchmark/cifar10_mlp_mup.py \
  --epochs 20 \
  --learning-rates $LRS_EDIT \
  --hidden-sizes 2048 \
  --methods manifold_online \
  --manifold-scaling fan_ratio \
  --linear-normalizations rms_radius \
  --use-wandb \
  --wandb-project cifar10-width-sweep-final \
  --wandb-group "muP width test2"

  uv run benchmark/cifar10_mlp_sp.py \
  --epochs 20 \
  --learning-rates $LRS_EDIT \
  --hidden-sizes 256 512 1024 2048 \
  --methods manifold_online \
  --use-wandb \
  --wandb-project cifar10-width-sweep-final \
  --wandb-group "SP Manifold"