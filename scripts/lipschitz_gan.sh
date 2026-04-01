#!/bin/bash

#SBATCH --output=slurm_logs/wandb_%j.out # Standard output log
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH -p gpu --gres=gpu:1     # number of gpus per node
#SBATCH --constraint=geforce3090


# Load Python module

module load python
#export PYTHONPATH=$(pwd):$PYTHONPATH

#uv run examples/cifar_lipschtz_gan.py --steps 5000 --generator-lrs 5e-2 --discriminator-lrs 5e-2 --discriminator-method manifold_online --generator-method dualize --use-wandb --wandb-project lipschitz-gan-gradients \
#uv run examples/cifar_lipschtz_gan.py --steps 5000 --generator-lrs 5e-4 --discriminator-lrs 5e-2 --discriminator-method manifold_online --generator-method descent --use-wandb --wandb-project lipschitz-gan-gradients \
uv run examples/cifar_lipschtz_gan.py --steps 100000 --generator-lrs 1e-4 --discriminator-lrs 0.5 --discriminator-method manifold_online --generator-method adam --use-wandb --wandb-project lipschitz-gan-gradients --save-model --no-gradient-penalty \
#uv run examples/cifar_lipschtz_gan.py --steps 100000 --generator-lrs 1e-4 --discriminator-lrs 1e-4 --discriminator-method adam --generator-method adam --use-wandb --wandb-project lipschitz-gan-gradients --save-model --no-gradient-penalty \