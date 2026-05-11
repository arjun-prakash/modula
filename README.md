# Modula CIFAR-10 MLP Benchmarks

This repository contains two CIFAR-10 MLP benchmark entrypoints built on top of
the Modula framework:

- `benchmark/cifar10_mlp_sp.py`: standard-parameterization MLP trained with SGD.
- `benchmark/cifar10_mlp_mup.py`: muP-style Modula MLP trained with the manifold update.

The first non-synthetic run downloads CIFAR-10 into `benchmark/cifar10_files/`.
Benchmark summaries are written as JSON files under `results/` by default.

## Setup

Create an environment and install the repository in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If your machine needs a specific JAX build for CUDA, install the appropriate JAX
wheel for your system before running the benchmarks.

## Run SP

Run the standard-parameterization baseline:

```bash
python benchmark/cifar10_mlp_sp.py \
  --steps 4000 \
  --learning-rate 0.1 \
  --hidden-size 64 \
  --output results/cifar10_mlp_sp.json
```

Useful flags:

- `--steps`: number of training steps.
- `--learning-rate`: SGD learning rate.
- `--hidden-size`: width of both hidden layers.
- `--batch-size`: mini-batch size.
- `--eval-every`: progress update interval.
- `--seed`: PRNG seed.
- `--output`: JSON output path.

## Run muP

Run the muP-style Modula benchmark:

```bash
python benchmark/cifar10_mlp_mup.py \
  --steps 4000 \
  --learning-rate 0.01 \
  --hidden-size 64 \
  --mup-base-width 256 \
  --output results/cifar10_mlp_mup.json
```

Useful flags:

- `--steps`: number of training steps.
- `--learning-rate`: manifold-update learning rate.
- `--hidden-size`: width of both hidden layers.
- `--mup-base-width`: base width used for readout initialization and update scaling.
- `--batch-size`: mini-batch size.
- `--eval-every`: progress update interval.
- `--seed`: PRNG seed.
- `--output`: JSON output path.

## Outputs

Each script prints a one-line run summary and writes a JSON payload with:

- run configuration,
- train and test accuracy,
- final mini-batch loss,
- full-train loss,
- runtime metrics,
- trunk geometry diagnostics.

## Credit

The benchmark code uses the Modula framework and its JAX implementation of
modules, atoms, bonds, duality, and projection/retraction operations. Modula is
based on:

- Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola, and Jeremy
  Bernstein, "Scalable Optimization in the Modular Norm", NeurIPS 2024.
- Jeremy Bernstein and Laker Newhouse, "Modular Duality in Deep Learning",
  arXiv:2410.21265.

The original Modula project and documentation are available at
https://github.com/modula-systems/modula and https://docs.modula.systems.

## License

This repository is released under the MIT license in `LICENSE`.
