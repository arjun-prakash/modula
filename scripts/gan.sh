#!/bin/bash

#SBATCH --output=slurm_logs/wandb_%j.out # Standard output log
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090

# Load Python module

module load python
#export PYTHONPATH=$(pwd):$PYTHONPATH

#uv run examples/gan.py --steps 5000 --learning-rate 5e-3 --method dualize
uv run examples/gan.py --steps 5000 --learning-rate 5e-2 --method manifold_online --target-norm 5.0