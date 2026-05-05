if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from benchmark.common import stiefel_deviation, tree_l2_norm
from benchmark.run_logging import create_run_logger
from modula.atom import Embed, Linear


BLOCK_LAYER_NAMES = ("q", "k", "v", "attn_out", "mlp_up", "mlp_down")
LAYER_POLICY_CHOICES = (
    "none",
    "mlp_only",
    "attention_only",
    "attention_value_out_mlp",
    "all_blocks",
)
_POLICY_LAYERS = {
    "none": (),
    "mlp_only": ("mlp_up", "mlp_down"),
    "attention_only": ("q", "k", "v", "attn_out"),
    "attention_value_out_mlp": ("v", "attn_out", "mlp_up", "mlp_down"),
    "all_blocks": BLOCK_LAYER_NAMES,
}
MANIFOLD_METHODS = ("manifold", "manifold_online", "manifold_admm")
GPT_METHOD_CHOICES = ("adam", "adamw", *MANIFOLD_METHODS)
MANIFOLD_SCALING_CHOICES = ("fan_ratio", "fan_max", "none")


@dataclass
class TokenDatasetBundle:
    name: str
    vocab_size: int
    train_tokens: jnp.ndarray
    val_tokens: jnp.ndarray


@dataclass
class AdamState:
    count: int
    m: Mapping
    v: Mapping


def canonicalize_layer_policies(layer_policies: Sequence[str]) -> list[str]:
    canonical: list[str] = []
    for policy in layer_policies:
        normalized = policy.lower()
        if normalized not in LAYER_POLICY_CHOICES:
            raise ValueError(f"Unknown layer policy: {policy}")
        if normalized not in canonical:
            canonical.append(normalized)
    return canonical


def canonicalize_gpt_methods(methods: Sequence[str]) -> list[str]:
    canonical: list[str] = []
    for method in methods:
        normalized = method.lower()
        if normalized not in GPT_METHOD_CHOICES:
            raise ValueError(f"Unknown GPT method: {method}")
        if normalized not in canonical:
            canonical.append(normalized)
    return canonical


def selected_layer_names(policy: str) -> Tuple[str, ...]:
    if policy not in _POLICY_LAYERS:
        raise ValueError(f"Unknown layer policy: {policy}")
    return tuple(_POLICY_LAYERS[policy])


def selected_atom_keys(policy: str, num_blocks: int) -> Tuple[Tuple[int, str], ...]:
    names = selected_layer_names(policy)
    return tuple((block_idx, name) for block_idx in range(num_blocks) for name in names)


def _make_synthetic_token_data(
    *,
    vocab_size: int,
    context_length: int,
    smoke_test: bool,
    seed: int,
) -> TokenDatasetBundle:
    train_length = 512 if smoke_test else 8192
    val_length = 256 if smoke_test else 2048
    rng = np.random.default_rng(seed)

    def make_stream(length: int) -> np.ndarray:
        base = np.arange(length, dtype=np.int32)
        periodic = (base * 17 + base // 7) % vocab_size
        noise = rng.integers(0, vocab_size, size=length, dtype=np.int32)
        return ((periodic + noise) % vocab_size).astype(np.int32)

    min_length = context_length + 2
    train_tokens = make_stream(max(train_length, min_length))
    val_tokens = make_stream(max(val_length, min_length))
    return TokenDatasetBundle(
        name="synthetic_gpt",
        vocab_size=vocab_size,
        train_tokens=jnp.asarray(train_tokens, dtype=jnp.int32),
        val_tokens=jnp.asarray(val_tokens, dtype=jnp.int32),
    )


def prepare_token_dataset(
    *,
    synthetic_data: bool,
    vocab_size: int,
    context_length: int,
    smoke_test: bool,
    seed: int,
) -> TokenDatasetBundle:
    if synthetic_data:
        return _make_synthetic_token_data(
            vocab_size=vocab_size,
            context_length=context_length,
            smoke_test=smoke_test,
            seed=seed,
        )

    from examples.data.shakespeare import load_shakespeare

    data = load_shakespeare(context_length, batch_size=1, shuffle=False)
    train_tokens = jnp.asarray(np.asarray(data["train_loader"].dataset.data), dtype=jnp.int32)
    val_tokens = jnp.asarray(np.asarray(data["val_loader"].dataset.data), dtype=jnp.int32)
    return TokenDatasetBundle(
        name="tiny_shakespeare",
        vocab_size=int(data["vocab_size"]),
        train_tokens=train_tokens,
        val_tokens=val_tokens,
    )


def get_lm_batch(key, tokens, *, context_length: int, batch_size: int):
    max_start = tokens.shape[0] - context_length - 1
    if max_start <= 0:
        raise ValueError("Token stream is too short for the requested context length")
    starts = jax.random.randint(key, (batch_size,), 0, max_start)
    offsets = jnp.arange(context_length, dtype=jnp.int32)
    positions = starts[:, None] + offsets[None, :]
    return tokens[positions], tokens[positions + 1]


class GPTBenchmarkModel:
    def __init__(
        self,
        *,
        vocab_size: int,
        num_heads: int,
        d_embed: int,
        d_query: int,
        d_value: int,
        num_blocks: int,
        attention_scale: float,
        final_scale: float,
        rope_base: int = 10000,
    ) -> None:
        if d_query <= 0 or d_query % 2 != 0:
            raise ValueError("d_query must be a positive even integer for RoPE")
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.d_embed = d_embed
        self.d_query = d_query
        self.d_value = d_value
        self.num_blocks = num_blocks
        self.attention_scale = float(attention_scale)
        self.final_scale = float(final_scale)
        self.residual_keep = 1.0 - 1.0 / (2.0 * num_blocks)
        self.residual_add = 1.0 / (2.0 * num_blocks)

        self.embed = Embed(d_embed, vocab_size)
        self.head = Linear(vocab_size, d_embed)
        self.blocks = []
        for _ in range(num_blocks):
            self.blocks.append(
                {
                    "q": Linear(num_heads * d_query, d_embed),
                    "k": Linear(num_heads * d_query, d_embed),
                    "v": Linear(num_heads * d_value, d_embed),
                    "attn_out": Linear(d_embed, num_heads * d_value),
                    "mlp_up": Linear(4 * d_embed, d_embed),
                    "mlp_down": Linear(d_embed, 4 * d_embed),
                }
            )

        rope_dim = d_query // 2
        self.rope_inverse_frequencies = 1 / rope_base ** (jnp.arange(rope_dim) / rope_dim)

    def initialize(self, key):
        key, embed_key, head_key = jax.random.split(key, 3)
        params = {
            "embed": self.embed.initialize(embed_key)[0],
            "blocks": [],
            "head": self.head.initialize(head_key)[0],
        }
        for block_atoms in self.blocks:
            block_params = {}
            for name in BLOCK_LAYER_NAMES:
                key, subkey = jax.random.split(key)
                block_params[name] = block_atoms[name].initialize(subkey)[0]
            params["blocks"].append(block_params)
        return params

    def _linear(self, x, weight):
        return jnp.einsum("...ij,...j->...i", weight, x)

    def _split_heads(self, x, head_dim):
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        batch, num_heads, seq_len, head_dim = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, seq_len, num_heads * head_dim)

    def _rope(self, x):
        seq_len = x.shape[-2]
        distance = jnp.arange(seq_len)
        freqs = jnp.outer(distance, self.rope_inverse_frequencies)
        cos = jnp.cos(freqs)[None, None, :, :]
        sin = jnp.sin(freqs)[None, None, :, :]
        rope_dim = self.d_query // 2
        x1 = x[..., rope_dim:]
        x2 = x[..., :rope_dim]
        y1 = cos * x1 + sin * x2
        y2 = -sin * x1 + cos * x2
        return jnp.concatenate([y1, y2], axis=-1)

    def _attention(self, x, block_params):
        q = self._split_heads(self._linear(x, block_params["q"]), self.d_query)
        k = self._split_heads(self._linear(x, block_params["k"]), self.d_query)
        v = self._split_heads(self._linear(x, block_params["v"]), self.d_value)
        q = self._rope(q)
        k = self._rope(k)

        scores = q @ k.transpose(0, 1, 3, 2)
        scores = scores * (1.0 / self.d_query)
        mask = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
        scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)
        scores = jax.nn.softmax(self.attention_scale * scores, axis=-1)

        values = self._merge_heads(scores @ v)
        return self._linear(values / 3.0, block_params["attn_out"])

    def _mlp(self, x, block_params):
        hidden = self._linear(x, block_params["mlp_up"])
        hidden = jax.nn.gelu(hidden) / 1.1289
        return self._linear(hidden, block_params["mlp_down"])

    def forward(self, params, tokens):
        x = params["embed"][tokens]
        for block_params in params["blocks"]:
            x = self.residual_keep * x + self.residual_add * self._attention(x, block_params)
            x = self.residual_keep * x + self.residual_add * self._mlp(x, block_params)
        return self.final_scale * self._linear(x, params["head"])

    def loss(self, params, inputs, targets):
        logits = self.forward(params, inputs)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        token_losses = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
        return jnp.mean(token_losses)


def make_adam_mask(params, active_selected: set[Tuple[int, str]]):
    return {
        "embed": True,
        "head": True,
        "blocks": [
            {name: (block_idx, name) not in active_selected for name in block_params}
            for block_idx, block_params in enumerate(params["blocks"])
        ],
    }


def init_adam_state(params) -> AdamState:
    return AdamState(
        count=0,
        m=jax.tree_util.tree_map(jnp.zeros_like, params),
        v=jax.tree_util.tree_map(jnp.zeros_like, params),
    )


def adam_step(
    params,
    grads,
    state: AdamState,
    mask,
    *,
    learning_rate: float,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
):
    count = state.count + 1

    def update_m(m, g, enabled):
        return beta1 * m + (1.0 - beta1) * g if enabled else m

    def update_v(v, g, enabled):
        return beta2 * v + (1.0 - beta2) * (g * g) if enabled else v

    m_next = jax.tree_util.tree_map(update_m, state.m, grads, mask)
    v_next = jax.tree_util.tree_map(update_v, state.v, grads, mask)

    def update_param(p, g, m, v, enabled):
        if not enabled:
            return p
        m_hat = m / (1.0 - beta1**count)
        v_hat = v / (1.0 - beta2**count)
        direction = m_hat / (jnp.sqrt(v_hat) + eps)
        if weight_decay:
            direction = direction + weight_decay * p
        return p - learning_rate * direction

    new_params = jax.tree_util.tree_map(update_param, params, grads, m_next, v_next, mask)
    deltas = jax.tree_util.tree_map(lambda new, old: new - old, new_params, params)
    return new_params, AdamState(count=count, m=m_next, v=v_next), deltas


def _zeros_like_params(params):
    return {
        "embed": jnp.zeros_like(params["embed"]),
        "head": jnp.zeros_like(params["head"]),
        "blocks": [
            {name: jnp.zeros_like(weight) for name, weight in block_params.items()}
            for block_params in params["blocks"]
        ],
    }


def _copy_params(params):
    return {
        "embed": params["embed"],
        "head": params["head"],
        "blocks": [dict(block_params) for block_params in params["blocks"]],
    }


def init_manifold_state(model: GPTBenchmarkModel, params, active_selected: set[Tuple[int, str]], method: str):
    dual_state = {}
    if method == "manifold_online":
        for block_idx, name in active_selected:
            atom = model.blocks[block_idx][name]
            dual_state[(block_idx, name)] = atom.init_dual_state([params["blocks"][block_idx][name]])

    momentum_state = None
    if method in MANIFOLD_METHODS:
        momentum_state = {
            (block_idx, name): jnp.zeros_like(params["blocks"][block_idx][name])
            for block_idx, name in active_selected
        }
    return dual_state, momentum_state


def manifold_update_scale(atom: Linear, *, scaling: str) -> float:
    if scaling == "none":
        return 1.0
    if scaling == "fan_ratio":
        return float(np.sqrt(atom.fanout / atom.fanin))
    if scaling == "fan_max":
        return float(np.sqrt(max(atom.fanin, atom.fanout)))
    raise ValueError(f"Unknown manifold scaling: {scaling}")


def apply_manifold_updates(
    model: GPTBenchmarkModel,
    params,
    grads,
    *,
    active_selected: set[Tuple[int, str]],
    method: str,
    learning_rate: float,
    dual_alpha: float,
    dual_beta: float,
    admm_steps: int,
    admm_rho: float,
    manifold_momentum: float,
    manifold_weight_decay: float,
    manifold_scaling: str,
    dual_state,
    momentum_state,
):
    if not active_selected:
        return params, dual_state, momentum_state, _zeros_like_params(params)

    new_params = _copy_params(params)
    deltas = _zeros_like_params(params)

    for block_idx, name in sorted(active_selected):
        atom = model.blocks[block_idx][name]
        weight = params["blocks"][block_idx][name]
        grad = grads["blocks"][block_idx][name]
        solver_grad = grad

        key = (block_idx, name)
        momentum = momentum_state[key]
        momentum = manifold_momentum * momentum + grad
        momentum_state[key] = momentum
        solver_grad = momentum

        if method == "manifold":
            tangent = atom.dual_ascent([weight], [solver_grad])[0]
        elif method == "manifold_online":
            tangent_list, next_state = atom.online_dual_ascent(
                dual_state[key],
                [weight],
                [solver_grad],
                alpha=dual_alpha,
                beta=dual_beta,
            )
            dual_state[key] = next_state
            tangent = tangent_list[0]
        elif method == "manifold_admm":
            tangent = atom.admm_dual_ascent(
                [weight],
                [solver_grad],
                steps=admm_steps,
                rho=admm_rho,
            )[0]
        else:
            raise ValueError(f"Unknown manifold method: {method}")

        scale = jnp.asarray(manifold_update_scale(atom, scaling=manifold_scaling), dtype=weight.dtype)
        direction = scale * tangent + manifold_weight_decay * weight
        updated = atom.retract([weight - learning_rate * direction])[0]

        new_params["blocks"][block_idx][name] = updated
        deltas["blocks"][block_idx][name] = updated - weight

    return new_params, dual_state, momentum_state, deltas


def compute_stiefel_metrics(
    model: GPTBenchmarkModel,
    params,
    policy_selected: Iterable[Tuple[int, str]],
) -> Dict[str, float]:
    deviations = []
    metrics: Dict[str, float] = {}
    for block_idx, name in policy_selected:
        atom = model.blocks[block_idx][name]
        weight = params["blocks"][block_idx][name]
        deviation = stiefel_deviation(atom, weight)
        if deviation is None:
            continue
        deviations.append(deviation)
        metrics[f"stiefel_deviation_block_{block_idx}_{name}"] = float(deviation)

    if deviations:
        metrics["stiefel_deviation_mean"] = float(np.mean(deviations))
        metrics["stiefel_deviation_max"] = float(np.max(deviations))
    else:
        metrics["stiefel_deviation_mean"] = 0.0
        metrics["stiefel_deviation_max"] = 0.0
    return metrics


def compute_eval_loss(
    loss_fn,
    params,
    tokens,
    *,
    context_length: int,
    batch_size: int,
    eval_iters: int,
    key,
) -> float:
    losses = []
    for idx in range(eval_iters):
        batch_key = jax.random.fold_in(key, idx)
        inputs, targets = get_lm_batch(batch_key, tokens, context_length=context_length, batch_size=batch_size)
        losses.append(loss_fn(params, inputs, targets))
    return float(jnp.mean(jnp.asarray(losses)))


def train_single_run(
    model: GPTBenchmarkModel,
    dataset: TokenDatasetBundle,
    *,
    context_length: int,
    batch_size: int,
    eval_iters: int,
    steps: int,
    learning_rate: float,
    eval_every: int,
    seed: int,
    method: str,
    layer_policy: str,
    dual_alpha: float,
    dual_beta: float,
    admm_steps: int,
    admm_rho: float,
    manifold_momentum: float,
    manifold_weight_decay: float,
    manifold_scaling: str,
    adam_weight_decay: float,
    logger=None,
    show_progress: bool = True,
):
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = model.initialize(init_key)

    policy_selected = set(selected_atom_keys(layer_policy, model.num_blocks))
    active_selected = set() if method in ("adam", "adamw") else policy_selected
    adam_mask = make_adam_mask(params, active_selected)
    adam_state = init_adam_state(params)
    dual_state, momentum_state = init_manifold_state(model, params, active_selected, method)

    loss_fn = jax.jit(model.loss)
    loss_and_grad = jax.jit(jax.value_and_grad(model.loss))

    progress = tqdm(
        range(steps),
        desc=f"{method}/{layer_policy} lr={learning_rate:.3g}",
        leave=False,
        disable=not show_progress,
    )
    start_time = time.perf_counter()
    last_loss = 0.0

    for step in progress:
        key, batch_key = jax.random.split(key)
        inputs, targets = get_lm_batch(
            batch_key,
            dataset.train_tokens,
            context_length=context_length,
            batch_size=batch_size,
        )
        loss_value, grads = loss_and_grad(params, inputs, targets)
        last_loss = float(loss_value)
        before_params = params

        params, adam_state, adam_deltas = adam_step(
            params,
            grads,
            adam_state,
            adam_mask,
            learning_rate=learning_rate,
            weight_decay=0.0 if method == "adam" else adam_weight_decay,
        )
        params, dual_state, momentum_state, manifold_deltas = apply_manifold_updates(
            model,
            params,
            grads,
            active_selected=active_selected,
            method=method,
            learning_rate=learning_rate,
            dual_alpha=dual_alpha,
            dual_beta=dual_beta,
            admm_steps=admm_steps,
            admm_rho=admm_rho,
            manifold_momentum=manifold_momentum,
            manifold_weight_decay=manifold_weight_decay,
            manifold_scaling=manifold_scaling,
            dual_state=dual_state,
            momentum_state=momentum_state,
        )
        deltas = jax.tree_util.tree_map(lambda new, old: new - old, params, before_params)

        if step % eval_every == 0 or step == steps - 1:
            eval_key = jax.random.fold_in(jax.random.PRNGKey(seed), step)
            train_loss = compute_eval_loss(
                loss_fn,
                params,
                dataset.train_tokens,
                context_length=context_length,
                batch_size=batch_size,
                eval_iters=eval_iters,
                key=eval_key,
            )
            val_loss = compute_eval_loss(
                loss_fn,
                params,
                dataset.val_tokens,
                context_length=context_length,
                batch_size=batch_size,
                eval_iters=eval_iters,
                key=jax.random.fold_in(eval_key, 1),
            )
            elapsed = time.perf_counter() - start_time
            tokens_seen = (step + 1) * batch_size * context_length
            epoch = tokens_seen / max(int(dataset.train_tokens.shape[0]), 1)
            metrics = {
                "epoch": float(epoch),
                "loss": float(loss_value),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "embed_grad_norm": tree_l2_norm(grads["embed"]),
                "block_grad_norm": tree_l2_norm(grads["blocks"]),
                "head_grad_norm": tree_l2_norm(grads["head"]),
                "embed_update_norm": tree_l2_norm(deltas["embed"]),
                "block_update_norm": tree_l2_norm(deltas["blocks"]),
                "head_update_norm": tree_l2_norm(deltas["head"]),
                "adam_update_norm": tree_l2_norm(adam_deltas),
                "manifold_update_norm": tree_l2_norm(manifold_deltas),
                "elapsed_time_seconds": float(elapsed),
                "seconds_per_step_so_far": float(elapsed / max(step + 1, 1)),
                "tokens_per_second_so_far": float(tokens_seen / max(elapsed, 1e-12)),
                **compute_stiefel_metrics(model, params, policy_selected),
            }
            if logger is not None:
                logger.log(metrics)
            progress.set_description(
                f"{method}/{layer_policy} lr={learning_rate:.3g} | epoch={epoch:.2f} | train={train_loss:.3f} | val={val_loss:.3f}"
            )

    elapsed = time.perf_counter() - start_time
    final_epoch = (steps * batch_size * context_length) / max(int(dataset.train_tokens.shape[0]), 1)
    final_key = jax.random.PRNGKey(seed + 1)
    final_train_loss = compute_eval_loss(
        loss_fn,
        params,
        dataset.train_tokens,
        context_length=context_length,
        batch_size=batch_size,
        eval_iters=eval_iters,
        key=final_key,
    )
    final_val_loss = compute_eval_loss(
        loss_fn,
        params,
        dataset.val_tokens,
        context_length=context_length,
        batch_size=batch_size,
        eval_iters=eval_iters,
        key=jax.random.fold_in(final_key, 1),
    )
    result = {
        "train_loss": float(final_train_loss),
        "val_loss": float(final_val_loss),
        "final_batch_loss": float(last_loss),
        "final_epoch": float(final_epoch),
        "training_time_seconds": float(elapsed),
        "seconds_per_step": float(elapsed / max(steps, 1)),
        "tokens_per_second": float((steps * batch_size * context_length) / max(elapsed, 1e-12)),
        **compute_stiefel_metrics(model, params, policy_selected),
    }
    if logger is not None:
        logger.log(result)
    return result


def apply_smoke_overrides(args) -> None:
    if not args.smoke_test:
        return
    args.learning_rates = [float(args.learning_rates[0] if args.learning_rates else 1e-2)]
    args.steps = min(int(args.steps), 2)
    args.batch_size = min(int(args.batch_size), 8)
    args.eval_every = 1
    args.eval_iters = min(int(args.eval_iters), 2)
    args.context_length = min(int(args.context_length), 32)
    args.d_embed = min(int(args.d_embed), 64)
    args.num_blocks = min(int(args.num_blocks), 2)
    args.d_query = min(int(args.d_query), max(2, args.d_embed // max(args.num_heads, 1)))
    if args.d_query % 2:
        args.d_query += 1
    args.d_value = min(int(args.d_value), max(1, args.d_embed // max(args.num_heads, 1)))


def make_run_config(args, *, dataset_name: str, method: str, layer_policy: str, learning_rate: float):
    return {
        "dataset": dataset_name,
        "method": method,
        "layer_policy": layer_policy,
        "learning_rate": float(learning_rate),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "eval_every": int(args.eval_every),
        "eval_iters": int(args.eval_iters),
        "seed": int(args.seed),
        "dual_alpha": float(args.dual_alpha),
        "dual_beta": float(args.dual_beta),
        "admm_steps": int(args.admm_steps),
        "admm_rho": float(args.admm_rho),
        "manifold_momentum": float(args.manifold_momentum),
        "manifold_weight_decay": float(args.manifold_weight_decay),
        "manifold_scaling": str(args.manifold_scaling),
        "adam_weight_decay": float(args.adam_weight_decay),
        "context_length": int(args.context_length),
        "vocab_size": int(args.vocab_size),
        "num_heads": int(args.num_heads),
        "d_embed": int(args.d_embed),
        "d_query": int(args.d_query),
        "d_value": int(args.d_value),
        "num_blocks": int(args.num_blocks),
        "attention_scale": float(args.attention_scale),
        "final_scale": float(args.final_scale),
        "synthetic_data": bool(args.synthetic_data),
        "smoke_test": bool(args.smoke_test),
    }


def save_results(dataset_name, config, results, best_runs, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": dataset_name, "config": dict(config), "methods": {}}
    for method, policies in results.items():
        payload["methods"][method] = {"policies": {}}
        for policy, runs in policies.items():
            payload["methods"][method]["policies"][policy] = {
                "runs": [dict(run) for run in runs],
            }
            key = (method, policy)
            if key in best_runs:
                payload["methods"][method]["policies"][policy]["best"] = dict(best_runs[key])

    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GPT benchmark with policy-based Manifold Muon layers")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dual-alpha", type=float, default=2e-5)
    parser.add_argument("--dual-beta", type=float, default=0.9)
    parser.add_argument("--admm-steps", type=int, default=10)
    parser.add_argument("--admm-rho", type=float, default=4.0)
    parser.add_argument("--manifold-momentum", type=float, default=0.9)
    parser.add_argument("--manifold-weight-decay", type=float, default=0.01)
    parser.add_argument("--manifold-scaling", type=str, default="fan_ratio", choices=MANIFOLD_SCALING_CHOICES)
    parser.add_argument("--adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--methods", type=str, nargs="+", default=["adam", "adamw", "manifold_online", "manifold_admm"], choices=GPT_METHOD_CHOICES)
    parser.add_argument("--layer-policies", type=str, nargs="+", default=["none", "mlp_only"], choices=LAYER_POLICY_CHOICES)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=65)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-embed", type=int, default=128)
    parser.add_argument("--d-query", type=int, default=32)
    parser.add_argument("--d-value", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--attention-scale", type=float, default=1.0)
    parser.add_argument("--final-scale", type=float, default=1.0)
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--results-path", type=Path, default=Path("results/benchmark_gpt_results.json"))
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gpt-manifold-muon-benchmark")
    parser.add_argument("--wandb-entity", type=str, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.methods = canonicalize_gpt_methods(args.methods)
    args.layer_policies = canonicalize_layer_policies(args.layer_policies)
    apply_smoke_overrides(args)

    dataset = prepare_token_dataset(
        synthetic_data=args.synthetic_data,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )
    args.vocab_size = dataset.vocab_size
    model = GPTBenchmarkModel(
        vocab_size=dataset.vocab_size,
        num_heads=args.num_heads,
        d_embed=args.d_embed,
        d_query=args.d_query,
        d_value=args.d_value,
        num_blocks=args.num_blocks,
        attention_scale=args.attention_scale,
        final_scale=args.final_scale,
    )

    results = {method: {policy: [] for policy in args.layer_policies} for method in args.methods}
    best_runs = {}
    base_key = jax.random.PRNGKey(args.seed)

    for method_idx, method in enumerate(args.methods):
        method_key = jax.random.fold_in(base_key, method_idx)
        for policy_idx, layer_policy in enumerate(args.layer_policies):
            if method in ("adam", "adamw") and layer_policy != "none":
                continue
            if method not in ("adam", "adamw") and layer_policy == "none":
                continue
            policy_key = jax.random.fold_in(method_key, policy_idx)
            for lr_idx, learning_rate in enumerate(args.learning_rates):
                run_key = jax.random.fold_in(policy_key, lr_idx)
                run_seed = int(jax.random.randint(run_key, (), 0, np.iinfo(np.int32).max))
                run_name = f"gpt-{method}-{layer_policy}-lr{learning_rate:.3g}"
                logger = create_run_logger(
                    use_wandb=args.use_wandb,
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=run_name,
                    config=make_run_config(
                        args,
                        dataset_name=dataset.name,
                        method=method,
                        layer_policy=layer_policy,
                        learning_rate=learning_rate,
                    ),
                    group="gpt-manifold-muon-benchmark",
                    tags=[dataset.name, method, layer_policy],
                )
                try:
                    run_result = train_single_run(
                        model,
                        dataset,
                        context_length=args.context_length,
                        batch_size=args.batch_size,
                        eval_iters=args.eval_iters,
                        steps=args.steps,
                        learning_rate=learning_rate,
                        eval_every=args.eval_every,
                        seed=run_seed,
                        method=method,
                        layer_policy=layer_policy,
                        dual_alpha=args.dual_alpha,
                        dual_beta=args.dual_beta,
                        admm_steps=args.admm_steps,
                        admm_rho=args.admm_rho,
                        manifold_momentum=args.manifold_momentum,
                        manifold_weight_decay=args.manifold_weight_decay,
                        manifold_scaling=args.manifold_scaling,
                        adam_weight_decay=args.adam_weight_decay,
                        logger=logger,
                        show_progress=not args.smoke_test,
                    )
                finally:
                    logger.finish()

                run_result["learning_rate"] = float(learning_rate)
                run_result["method"] = method
                run_result["layer_policy"] = layer_policy
                results[method][layer_policy].append(run_result)

                best_key = (method, layer_policy)
                best = best_runs.get(best_key)
                if best is None or run_result["val_loss"] < best["val_loss"]:
                    best_runs[best_key] = dict(run_result)

                print(
                    f"[{method}/{layer_policy}] lr={learning_rate:.3g}: "
                    f"train loss={run_result['train_loss']:.4f} | "
                    f"val loss={run_result['val_loss']:.4f} | "
                    f"epoch={run_result['final_epoch']:.2f} | "
                    f"time={run_result['training_time_seconds']:.2f}s"
                )

    config = make_run_config(
        args,
        dataset_name=dataset.name,
        method=",".join(args.methods),
        layer_policy=",".join(args.layer_policies),
        learning_rate=float(args.learning_rates[0]),
    )
    config["learning_rates"] = [float(rate) for rate in args.learning_rates]
    save_results(dataset.name, config, results, best_runs, args.results_path)


if __name__ == "__main__":
    main()
