# muP MLP SP+SGD Reproduction

This directory is a narrow reproduction of the standard-parameterization SGD
path from `microsoft/mup/examples/MLP/main.py`. It keeps the original CIFAR-10
MLP shape, initialization, `MuReadout`, `MuSGD`, momentum, batch size, epoch
count, ReLU/xent multiplier defaults, and `set_base_shapes(mynet, None)` SP
setup, while removing the original script's μP, coordinate-check, tanh, and MSE
branches.

Source matched: https://github.com/microsoft/mup/blob/main/examples/MLP/main.py

## Run One Setting

```bash
interact -q gpu -g 1
uv run --with mup python reproductions/mup_mlp_sp_sgd/train.py \
  --data_dir /tmp \
  --log_dir reproductions/mup_mlp_sp_sgd/results \
  --log_file h1024_lr_2m7.tsv \
  --chart_file h1024_lr_2m7_final_train_loss.tsv \
  --widths 1024 \
  --lr 0.0078125
```

## Run A Width/LR Sweep

```bash
interact -q gpu -g 1
uv run --with mup python reproductions/mup_mlp_sp_sgd/train.py \
  --data_dir /tmp \
  --log_dir reproductions/mup_mlp_sp_sgd/results \
  --log_file sp_sgd_width_lr_sweep.tsv \
  --chart_file final_train_loss.tsv \
  --widths 256 512 1024 2048 \
  --learning_rates \
    0.00048828125 0.0009765625 0.001953125 0.00390625 \
    0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5
```

The output file defaults to `logs.tsv`, and can be renamed with `--log_file`.
It contains full-epoch train loss and test loss after every epoch. Defaults
intentionally mirror the original script where relevant: `--batch_size 64`,
`--epochs 20`, `--momentum 0.9`, `--lr 0.1`, `--input_mult 0.00390625`, and
`--output_mult 32.0`.

The compact chart file defaults to `final_train_loss.tsv`, and can be renamed
with `--chart_file`. It is a width-by-`log2(lr)` matrix whose cells contain the
final epoch's full train loss for that width/LR run.

## Scope

This is not wired into the Modula benchmark entrypoints. It is meant to answer a
single question: what happens when the original muP project's CIFAR-10 MLP code
runs its SP setup with SGD and momentum?
