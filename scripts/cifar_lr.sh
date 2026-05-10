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

LRS="1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1"

LRS_EDIT="4.88e-4 9.76e-4 1.95e-3 3.9e-3 7.81e-3 1.56e-2 3.12e-2 6.25e-2 1.25e-1 2.5e-1 5e-1"
HIDDEN_SIZE=256

uv run benchmark/cifar10_mlp_sp.py \
  --steps 10000 \
  --learning-rates $LRS_EDIT \
  --hidden-sizes $HIDDEN_SIZE \
  --methods adam \
  --use-wandb \
  --wandb-project cifar10-lr-sweep-new \
  --wandb-group "Adam baseline"

# uv run benchmark/cifar10_mlp_mup.py \
#   --steps 10000 \
#   --learning-rates $LRS_EDIT \
#   --hidden-sizes $HIDDEN_SIZE \
#   --methods manifold \
#   --manifold-scaling none \
#   --manifold-momentum 0.0 \
#   --manifold-weight-decay 0.0 \
#   --linear-normalizations unit_stiefel \
#   --use-wandb \
#   --wandb-project cifar10-lr-sweep-new \
#   --wandb-group "Manifold no momentum no wd"

# uv run benchmark/cifar10_mlp_mup.py \
#   --steps 10000 \
#   --learning-rates $LRS_EDIT \
#   --hidden-sizes $HIDDEN_SIZE \
#   --methods manifold  \
#   --manifold-scaling fan_ratio \
#   --linear-normalizations rms_radius \
#   --use-wandb \
#   --wandb-project cifar10-lr-sweep-new \
#   --wandb-group "Manifold"

# uv run benchmark/cifar10_mlp_mup.py \
#   --steps 10000 \
#   --learning-rates $LRS_EDIT \
#   --hidden-sizes $HIDDEN_SIZE \
#   --methods manifold_admm \
#   --manifold-scaling fan_ratio \
#   --linear-normalizations rms_radius \
#   --use-wandb \
#   --wandb-project cifar10-lr-sweep-new \
#   --wandb-group "Manifold ADMM"
