#!/usr/bin/env python3
"""
deeponet.py - Modula-based Deep Operator Network implementation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from modula.atom import Bias, Linear
from modula.bond import HadamardProduct, ReLU, Select
from modula.abstract import Module


ArrayLike = Union[np.ndarray, jnp.ndarray]
PathLike = Union[str, Path]


__all__ = ['DeepONet', 'DeepONetConfig', 'build_deeponet_module']


def sequential(modules: Sequence[Module]) -> Module:
    if not modules:
        raise ValueError('sequential requires at least one module')
    composed = modules[-1]
    for block in reversed(modules[:-1]):
        composed = composed @ block
    return composed


@dataclass(frozen=True)
class DeepONetConfig:
    branch_input_dim: int = 10
    trunk_input_dim: int = 10
    hidden_dim: int = 64
    output_dim: int = 3
    branch_mass: float = 1.0
    trunk_mass: float = 1.0


def _build_branch(config: DeepONetConfig) -> Module:
    branch = sequential([
        Select(0),
        Linear(config.hidden_dim, config.branch_input_dim),
        Bias(config.hidden_dim),
        ReLU(),
        Linear(config.hidden_dim, config.hidden_dim),
        Bias(config.hidden_dim),
        ReLU(),
        Linear(config.hidden_dim, config.hidden_dim),
        Bias(config.hidden_dim),
    ])
    branch.tare(absolute=config.branch_mass)
    return branch


def _build_trunk(config: DeepONetConfig) -> Module:
    trunk = sequential([
        Select(1),
        Linear(config.hidden_dim, config.trunk_input_dim),
        Bias(config.hidden_dim),
        ReLU(),
        Linear(config.hidden_dim, config.hidden_dim),
        Bias(config.hidden_dim),
        ReLU(),
        Linear(config.hidden_dim, config.hidden_dim),
        Bias(config.hidden_dim),
    ])
    trunk.tare(absolute=config.trunk_mass)
    return trunk


def build_deeponet_module(config: DeepONetConfig) -> Module:
    branch = _build_branch(config)
    trunk = _build_trunk(config)
    features = HadamardProduct() @ (branch, trunk)

    model = sequential([
        features,
        Linear(config.output_dim, config.hidden_dim),
        Bias(config.output_dim),
    ])
    return model


def _as_numpy(weights: Sequence[ArrayLike]) -> List[np.ndarray]:
    return [np.asarray(w) for w in weights]


def _as_jax(weights: Sequence[ArrayLike]) -> List[jnp.ndarray]:
    return [jnp.asarray(w) for w in weights]


def _normalise_inputs(gust_features: ArrayLike, trunk_vec: ArrayLike) -> Tuple[jnp.ndarray, jnp.ndarray]:
    gf = jnp.asarray(gust_features, dtype=jnp.float32)
    tv = jnp.asarray(trunk_vec, dtype=jnp.float32)
    if gf.ndim == 1:
        gf = gf[None, :]
    if tv.ndim == 1:
        tv = tv[None, :]
    return gf, tv


class DeepONet:
    """Convenience wrapper around the Modula DeepONet module."""

    def __init__(self, config: Optional[DeepONetConfig] = None, *, jit: bool = True):
        self.config = config or DeepONetConfig()
        self.model = build_deeponet_module(self.config)
        if jit:
            self.model.jit()
        self._weights: Optional[List[jnp.ndarray]] = None

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------
    def initialize(self, key: [jax.Array]) -> List[jnp.ndarray]:
        self._weights = self.model.initialize(key)
        return self._weights

    def set_weights(self, weights: Sequence[ArrayLike]) -> None:
        self._weights = _as_jax(weights)

    @property
    def weights(self) -> List[jnp.ndarray]:
        if self._weights is None:
            raise ValueError("DeepONet weights are not initialised.")
        return self._weights

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_weights(self, filepath: PathLike, weights: Optional[Sequence[ArrayLike]] = None) -> None:
        weights_to_save = _as_numpy(weights or self.weights)
        payload = {f"arr_{idx}": array for idx, array in enumerate(weights_to_save)}
        payload["num_arrays"] = np.array(len(weights_to_save), dtype=np.int32)
        payload["config_json"] = np.array(json.dumps(asdict(self.config)))
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        np.savez(Path(filepath), **payload)

    def load_weights(self, filepath: PathLike) -> List[jnp.ndarray]:
        with np.load(Path(filepath), allow_pickle=False) as data:
            try:
                config_payload = json.loads(str(data["config_json"].item()))
                config = DeepONetConfig(**config_payload)
            except KeyError:
                config = self.config
            if config != self.config:
                self.config = config
                self.model = build_deeponet_module(self.config)
                self.model.jit()

            num_arrays = int(data["num_arrays"]) if "num_arrays" in data else None
            if num_arrays is None:
                keys = sorted(k for k in data.files if k.startswith("arr_"))
            else:
                keys = [f"arr_{idx}" for idx in range(num_arrays)]
            weights = [jnp.asarray(data[key]) for key in keys]
            self._weights = weights
        return self.weights

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def apply(self, gust_features: ArrayLike, trunk_vec: ArrayLike, *, weights: Optional[Sequence[ArrayLike]] = None) -> jnp.ndarray:
        model_weights = _as_jax(weights or self.weights)
        gf, tv = _normalise_inputs(gust_features, trunk_vec)
        return self.model((gf, tv), model_weights)

    def forward(self, gust_features: ArrayLike, trunk_vec: ArrayLike, *, weights: Optional[Sequence[ArrayLike]] = None) -> np.ndarray:
        output = self.apply(gust_features, trunk_vec, weights=weights)
        return np.asarray(output)

    __call__ = forward
