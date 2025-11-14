#!/bin/bash

#SBATCH --output=slurm_logs/wandb_%j.out # Standard output log
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH -p gpu --gres=gpu:2     # number of gpus per node
#SBATCH --constraint=geforce3090


# Load Python module

module load python
#export PYTHONPATH=$(pwd):$PYTHONPATH

#uv run examples/gan.py --steps 5000 --learning-rate 5e-3 --method dualize
#uv run examples/cifar_gan.py --steps 50000 --learning-rate 5e-3 --discriminator-method manifold_online --generator-method dualize 
#uv run examples/cifar_gan.py --steps 50000 --learning-rate 5e-3 --discriminator-method dualize --generator-method dualize 
uv run examples/cifar_lipschtz_gan.py --steps 50000 --learning-rate 5e-3 --discriminator-method manifold_online --generator-method dualize 