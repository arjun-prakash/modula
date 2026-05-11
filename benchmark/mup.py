"""CIFAR MLP builders for muP-style Modula runs."""

import math

from benchmark.scaling import iter_weighted_atoms
from modula import mup as mup_atoms
from modula.bond import Flatten, ReLU

MLP_FEATURE_DIM = 64
PARAMETERIZATION = "rms_radius"


def build_mlp_trunk(hidden_size: int = MLP_FEATURE_DIM):
    trunk = ReLU() @ mup_atoms.RMSRadiusLinear(hidden_size, hidden_size)
    trunk @= ReLU() @ mup_atoms.RMSRadiusLinear(hidden_size, 32 * 32 * 3)
    trunk @= Flatten()
    trunk.jit()
    return trunk


def build_mlp_classifier_head(num_classes: int, feature_dim: int = MLP_FEATURE_DIM):
    head = mup_atoms.Linear(num_classes, feature_dim)
    head.jit()
    return head


def _head_fanin(head) -> int:
    weighted_atoms = iter_weighted_atoms(head)
    if weighted_atoms:
        fanin = getattr(weighted_atoms[0], "fanin", None)
    else:
        fanin = getattr(head, "fanin", None)
    if fanin is None:
        raise ValueError(f"Cannot infer head fan-in for {type(head).__name__}")
    return int(fanin)


def head_adam_update_scale(head, *, base_width: int = 1) -> float:
    return float(base_width) / float(_head_fanin(head))


def head_init_scale(head, *, base_width: int = 1) -> float:
    return math.sqrt(float(base_width) / float(_head_fanin(head)))


def build_cifar_mlp_models(num_classes: int, hidden_size: int = MLP_FEATURE_DIM):
    return build_mlp_trunk(hidden_size), build_mlp_classifier_head(num_classes, hidden_size)
