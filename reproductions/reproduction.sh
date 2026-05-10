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

# this is -7.75, -7.5, -7.25, -7.0, -6.75, -6.5, -6.25 on log2 scale
LRS_PRECISE="0.00464534 0.00552427 0.00656864 0.0078125 0.00929068 0.01104854 0.01313901"

# ths is -11 to -1
LRS_EDIT="4.88e-4 9.76e-4 1.95e-3 3.9e-3 7.81e-3 1.56e-2 3.12e-2 6.25e-2 1.25e-1 2.5e-1 5e-1"

LRS_SMALL="0.00006 0.00012 0.00024"

# uv run --with mup python reproductions/mup_mlp_sp_sgd/train.py \
#   --data_dir /tmp \
#   --log_dir reproductions/mup_mlp_sp_sgd/results \
#   --widths 8192 \
#   --output-_mult 0 \
#   --input_mult 0 \
#   --learning_rates 1.56e-2 3.12e-2 6.25e-2 1.25e-1 2.5e-1 5e-1 \
#   --log_file "extended-8192-larger"

uv run --with mup python reproductions/mup_mlp_sp_sgd/train.py \
  --data_dir /tmp \
  --log_dir reproductions/mup_mlp_sp_sgd/results \
  --widths 256 2048 8192 \
  --learning_rates $LRS_SMALL\
  --log_file "full-train-with-inout-mult-extended"