# TODO

- Benchmark manifold update scaling: cache the per-atom outer scale coefficients in `benchmark/common.py` once per run instead of recomputing module-tree target-norm routing every training step. The current implementation is correct, but it adds avoidable Python overhead to `manifold`, `manifold_online`, and `manifold_admm` updates.
