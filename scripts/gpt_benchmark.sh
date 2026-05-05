#!/bin/bash

#SBATCH --output=slurm_logs/wandb_%j.out # Standard output log
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH -p gpu --gres=gpu:1
#SBATCH --constraint=geforce3090

module load python

MANIFOLD_METHODS="manifold_admm manifold_online"
MANIFOLD_LAYER_POLICIES="mlp_only attention_value_out_mlp all_blocks"

# uv run benchmark/gpt.py \
#   --steps 5000 \
#   --learning-rates 1e-3 \
#   --methods adam \
#   --layer-policies none \
#   --context-length 64 \
#   --batch-size 12 \
#   --eval-every 100 \
#   --eval-iters 20 \
#   --d-embed 128 \
#   --num-heads 4 \
#   --d-query 32 \
#   --d-value 32 \
#   --num-blocks 4 \
#   --use-wandb \
#   --wandb-project gpt-manifold-muon-benchmark

# uv run benchmark/gpt.py \
#   --steps 5000 \
#   --learning-rates 1e-3 \
#   --methods adamw \
#   --adam-weight-decay 0.01 \
#   --layer-policies none \
#   --context-length 64 \
#   --batch-size 12 \
#   --eval-every 100 \
#   --eval-iters 20 \
#   --d-embed 128 \
#   --num-heads 4 \
#   --d-query 32 \
#   --d-value 32 \
#   --num-blocks 4 \
#   --use-wandb \
#   --wandb-project gpt-manifold-muon-benchmark

uv run benchmark/gpt.py \
  --steps 5000 \
  --learning-rates 1e-3 5e-3 \
  --methods $MANIFOLD_METHODS \
  --layer-policies $MANIFOLD_LAYER_POLICIES \
  --context-length 64 \
  --batch-size 12 \
  --eval-every 100 \
  --eval-iters 20 \
  --d-embed 128 \
  --num-heads 4 \
  --d-query 32 \
  --d-value 32 \
  --num-blocks 4 \
  --manifold-momentum 0.9 \
  --manifold-weight-decay 0.01 \
  --manifold-scaling fan_ratio \
  --use-wandb \
  --wandb-project gpt-manifold-muon-benchmark
