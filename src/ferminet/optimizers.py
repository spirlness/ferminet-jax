# Copyright 2024 FermiNet Authors.
# Licensed under the Apache License, Version 2.0.

"""Optimizer factories for FermiNet training."""

from collections.abc import Callable, Mapping
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
import kfac_jax
import ml_collections
import optax

from ferminet import constants, types

Array = jnp.ndarray
ParamTree = types.ParamTree


def make_schedule(cfg: ml_collections.ConfigDict) -> optax.Schedule:
    """Builds warmup + inverse decay schedule."""
    cfg_any = cast(Any, cfg)
    lr_cfg = cfg_any.optim.lr
    return optax.warmup_exponential_decay_schedule(
        init_value=0.0,
        peak_value=lr_cfg.rate,
        warmup_steps=int(lr_cfg.get("warmup_steps", 1000)),
        transition_steps=1,
        decay_rate=lr_cfg.decay,
        transition_begin=int(lr_cfg.delay),
    )


def find_dense_params(
    params: Any,
    in_dim: int,
) -> list[tuple[Array, Optional[Array]]]:
    """Search for (w, b) pairs with matching input dimension."""
    dense_params: list[tuple[Array, Optional[Array]]] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            w = None
            b = None
            if "w" in node:
                w = node["w"]
            elif "weight" in node:
                w = node["weight"]
            if "b" in node:
                b = node["b"]
            elif "bias" in node:
                b = node["bias"]
            if w is not None and isinstance(w, jnp.ndarray) and w.ndim == 2:
                if w.shape[0] == in_dim:
                    dense_params.append((cast(Array, w), cast(Optional[Array], b)))
            for value in node.values():
                visit(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                visit(value)

    visit(params)
    return dense_params


def register_kfac_dense(params: ParamTree, inputs: Array) -> bool:
    """Register a dense layer for KFAC using matching parameters."""
    dense_params = find_dense_params(params, inputs.shape[-1])
    if not dense_params:
        return False
    w, b = dense_params[0]
    outputs = jnp.dot(inputs, w)
    if b is not None:
        outputs = outputs + b
    kfac_jax.register_dense(inputs, outputs, w, b)
    return True


def make_kfac_optimizer(
    cfg: ml_collections.ConfigDict,
    loss_fn: Callable[[ParamTree, jax.Array, types.FermiNetData], tuple[Array, Any]],
) -> tuple[kfac_jax.Optimizer, Callable[..., Any], Callable[..., Any]]:
    """Create KFAC optimizer init and update functions."""
    cfg_any = cast(Any, cfg)

    val_and_grad = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

    kfac_cfg = cfg_any.optim.kfac
    lr_cfg = cfg_any.optim.lr

    def learning_rate_schedule(t: jnp.ndarray) -> jnp.ndarray:
        return lr_cfg.rate * jnp.power((1.0 / (1.0 + (t / lr_cfg.delay))), lr_cfg.decay)

    optimizer = kfac_jax.Optimizer(
        value_and_grad_func=val_and_grad,
        l2_reg=kfac_cfg.l2_reg,
        norm_constraint=kfac_cfg.norm_constraint,
        value_func_has_aux=True,
        value_func_has_rng=True,
        learning_rate_schedule=learning_rate_schedule,
        curvature_ema=kfac_cfg.get("cov_ema_decay", 0.95),
        inverse_update_period=kfac_cfg.invert_every,
        min_damping=kfac_cfg.get("min_damping", 1e-4),
        num_burnin_steps=0,
        register_only_generic=kfac_cfg.register_only_generic,
        estimation_mode="fisher_exact",
        multi_device=True,
        pmap_axis_name=constants.PMAP_AXIS_NAME,
    )

    def init_fn(params: ParamTree, key: jax.Array, data: types.FermiNetData) -> Any:
        return optimizer.init(params, key, data)

    def update_fn(
        params: ParamTree,
        opt_state: Any,
        key: jax.Array,
        data: types.FermiNetData,
        step: jnp.ndarray,
    ) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        del step
        shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
        shared_damping = kfac_jax.utils.replicate_all_local_devices(
            jnp.asarray(kfac_cfg.damping)
        )
        new_params, new_state, stats = optimizer.step(
            params=params,
            state=opt_state,
            rng=key,
            batch=data,
            momentum=shared_mom,
            damping=shared_damping,
        )
        loss_value = stats.get("loss", jnp.asarray(0.0))
        aux = stats.get("aux", None)
        return new_params, new_state, loss_value, aux, stats

    return optimizer, init_fn, update_fn


def make_adam_optimizer(
    cfg: ml_collections.ConfigDict,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Create Adam optimizer init and update functions."""
    cfg_any = cast(Any, cfg)
    schedule = make_schedule(cfg)
    optimizer = optax.adam(
        learning_rate=schedule,
        b1=cfg_any.optim.adam.b1,
        b2=cfg_any.optim.adam.b2,
        eps=cfg_any.optim.adam.eps,
    )

    def init_fn(params: ParamTree) -> Any:
        return optimizer.init(params)

    def update_fn(
        params: ParamTree,
        opt_state: Any,
        key: jax.Array,
        data: types.FermiNetData,
        step: jnp.ndarray,
        loss_fn: Callable[
            [ParamTree, jax.Array, types.FermiNetData], tuple[Array, Any]
        ],
    ) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        (loss_value, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, key, data
        )
        grads = constants.pmean(grads)
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        stats = {"grad_norm": optax.global_norm(grads)}
        return new_params, new_state, loss_value, aux, stats

    return init_fn, update_fn


def make_optimizer(
    cfg: ml_collections.ConfigDict,
    loss_fn: Callable[[ParamTree, jax.Array, types.FermiNetData], tuple[Array, Any]],
    params: ParamTree,
) -> tuple[Optional[kfac_jax.Optimizer], Callable[..., Any], Callable[..., Any]]:
    """Create optimizer factory based on config."""
    del params
    cfg_any = cast(Any, cfg)
    if cfg_any.optim.optimizer == "kfac":
        return make_kfac_optimizer(cfg, loss_fn)
    init_fn, update_fn = make_adam_optimizer(cfg)
    return None, init_fn, update_fn
