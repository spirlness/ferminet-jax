# Copyright 2024 FermiNet Authors.
# Licensed under the Apache License, Version 2.0.

"""Training loop for FermiNet VMC with KFAC/Adam optimizers."""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import lax
import kfac_jax
import ml_collections

from ferminet import (
    base_config,
    checkpoint,
    constants,
    loss,
    mcmc,
    optimizers,
    train_utils,
    types,
)
from ferminet.utils import devices as device_utils

Array = jnp.ndarray
ParamTree = types.ParamTree

make_schedule = optimizers.make_schedule
_prepare_system = train_utils.prepare_system
_build_network = train_utils.build_network
_make_local_energy_fn = train_utils.make_local_energy_fn
_init_mcmc_data = train_utils.init_mcmc_data
_restore_checkpoint = train_utils.restore_checkpoint
_shard_array = device_utils.shard_array
_shard_data = device_utils.shard_data
_p_split = device_utils.p_split


def _filter_kwargs(fn: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Filter kwargs to those accepted by fn."""
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _to_host(tree: Any) -> Any:
    """Convert a PyTree of device arrays to a PyTree of host scalars."""
    host_tree = jax.device_get(tree)

    def _to_scalar(x: Any) -> float:
        x = jnp.asarray(x)
        if x.ndim > 0:
            x = jnp.reshape(x, (-1,))[0]
        return float(x)

    return jax.tree_util.tree_map(_to_scalar, host_tree)


def _convert_to_float(value: Any) -> float:
    """Convert a numpy array or scalar to a Python float."""
    if hasattr(value, "ndim") and value.ndim > 0:
        return float(value.ravel()[0])
    return float(value)


def update_width_step(
    energy: jnp.ndarray,
    pmove: jnp.ndarray,
    width: jnp.ndarray,
    step: jnp.ndarray,
    adapt_frequency: int,
    pmove_max: float,
    pmove_min: float,
    width_min: float,
    width_max: float,
    default_width: float,
) -> jnp.ndarray:
    """Updates MCMC width and handles non-finite energy on device."""
    # Check for NaNs/Infs
    is_finite = jnp.isfinite(energy)
    # If any device has non-finite energy, we should probably reset?
    # Or just local reset? Since params are synced, if one is NaN, params likely NaN everywhere.
    # We sync the valid flag across devices.
    all_finite = lax.pmin(is_finite, axis_name=constants.PMAP_AXIS_NAME)

    # Calculate adapted width
    pmove_mean = lax.pmean(pmove, axis_name=constants.PMAP_AXIS_NAME)

    updated_width = mcmc.update_mcmc_width_jax(
        step,
        width,
        adapt_frequency,
        pmove_mean,
        pmove_max=pmove_max,
        pmove_min=pmove_min,
        width_min=width_min,
        width_max=width_max,
    )

    should_adapt = (step + 1) % adapt_frequency == 0
    width_adapted = jnp.where(should_adapt, updated_width, width)

    # If not finite, reset to default width
    final_width = jnp.where(all_finite, width_adapted, default_width)
    return final_width


def make_update_width_fn(
    cfg: ml_collections.ConfigDict,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Creates a pmapped function to update MCMC width."""
    cfg_any = cast(Any, cfg)
    adapt_frequency = int(cfg_any.mcmc.adapt_frequency)
    pmove_max = float(cfg_any.mcmc.get("pmove_max", 0.55))
    pmove_min = float(cfg_any.mcmc.get("pmove_min", 0.5))
    width_min = 0.001  # Hardcoded in original python function default?
    # mcmc.update_mcmc_width has defaults width_min=0.001, width_max=10.0
    # We should probably respect those defaults or read from config if available.
    width_max = 10.0
    default_width = float(cfg_any.mcmc.move_width)

    @functools.partial(jax.pmap, axis_name=constants.PMAP_AXIS_NAME)
    def update_width_fn(
        energy: jnp.ndarray,
        pmove: jnp.ndarray,
        width: jnp.ndarray,
        step: jnp.ndarray,
    ) -> jnp.ndarray:
        return update_width_step(
            energy,
            pmove,
            width,
            step,
            adapt_frequency=adapt_frequency,
            pmove_max=pmove_max,
            pmove_min=pmove_min,
            width_min=width_min,
            width_max=width_max,
            default_width=default_width,
        )

    return update_width_fn


def train(cfg: ml_collections.ConfigDict) -> Mapping[str, Any]:
    """Run VMC training with KFAC or Adam optimizer."""
    cfg = base_config.resolve(cfg)
    cfg_any = cast(Any, cfg)
    key = jax.random.PRNGKey(cfg_any.debug.get("seed", 0))

    atoms, charges, spins, ndim = _prepare_system(cfg)
    batch_size = int(cfg_any.batch_size)

    init_fn, apply_log, _ = _build_network(cfg, atoms, charges, spins)

    local_energy_fn = _make_local_energy_fn(apply_log, charges, spins, cfg)
    loss_fn = loss.make_loss(
        apply_log,
        local_energy_fn,
        clip_local_energy=cfg_any.optim.clip_local_energy,
    )

    key, init_key, data_key = jax.random.split(key, 3)
    params = init_fn(init_key)
    data = _init_mcmc_data(data_key, atoms, charges, spins, batch_size, ndim)

    devices = jax.devices()
    device_count = len(devices)
    if batch_size % device_count != 0:
        raise ValueError(
            f"Batch size {batch_size} must be divisible by device count {device_count}."
        )
    batch_per_device = batch_size // device_count

    data = _shard_data(data, device_count)
    params = device_utils.replicate_tree(params, devices)
    key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

    _, init_opt_state, update_fn = optimizers.make_optimizer(cfg, loss_fn, params)
    if cfg_any.optim.optimizer == "kfac":
        key, init_keys = _p_split(key)
        opt_state = init_opt_state(params, init_keys, data)
    else:
        opt_state = jax.pmap(init_opt_state)(params)

    params, opt_state, data, step = _restore_checkpoint(
        cfg, params, opt_state, data, step=0
    )

    mcmc_step = mcmc.make_mcmc_step(
        apply_log,
        batch_per_device=batch_per_device,
        steps=int(cfg_any.mcmc.steps),
        atoms=atoms if cfg_any.mcmc.use_langevin else None,
        ndim=ndim,
    )

    schedule = make_schedule(cfg)
    width = jnp.full((device_count,), float(cfg_any.mcmc.move_width))

    pmapped_update_width_fn = make_update_width_fn(cfg)

    # Arguments for update_width_step (for Adam)
    adapt_frequency = int(cfg_any.mcmc.adapt_frequency)
    pmove_max = float(cfg_any.mcmc.get("pmove_max", 0.55))
    pmove_min = float(cfg_any.mcmc.get("pmove_min", 0.5))
    default_width = float(cfg_any.mcmc.move_width)

    if cfg_any.optim.optimizer == "kfac":
        pmapped_mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)

        def kfac_step_fn(
            params: ParamTree,
            opt_state: Any,
            data: types.FermiNetData,
            key: jax.Array,
            step: jnp.ndarray,
            mcmc_width: Any,
        ) -> tuple[Any, Any, Any, Any, train_utils.StepStats, Any]:
            mcmc_keys, loss_keys = _p_split(key)
            new_data, pmove = pmapped_mcmc_step(params, data, mcmc_keys, mcmc_width)

            # Copy params and opt_state because update_fn (KFAC) donates them.
            params_old = jax.tree_util.tree_map(lambda x: x.copy(), params)
            opt_state_old = jax.tree_util.tree_map(lambda x: x.copy(), opt_state)

            new_params, new_opt_state, loss_value, aux, _ = update_fn(
                params,
                opt_state,
                loss_keys,
                new_data,
                step,
            )

            # Update width on device
            new_width = pmapped_update_width_fn(loss_value, pmove, mcmc_width, step)

            energy = loss_value[0] if hasattr(loss_value, "__getitem__") else loss_value
            variance = (
                aux.variance[0]
                if hasattr(aux.variance, "__getitem__")
                else aux.variance
            )
            pmove_val = pmove[0] if hasattr(pmove, "__getitem__") else pmove
            step_val = step[0] if hasattr(step, "__getitem__") else step
            lr = jnp.asarray(schedule(step_val))
            step_stats = train_utils.StepStats(
                energy=energy, variance=variance, pmove=pmove_val, learning_rate=lr
            )

            is_finite = jnp.isfinite(energy)
            new_params = jax.tree_util.tree_map(
                lambda p, np: jnp.where(is_finite, np, p), params_old, new_params
            )
            new_opt_state = jax.tree_util.tree_map(
                lambda p, np: jnp.where(is_finite, np, p), opt_state_old, new_opt_state
            )

            return new_params, new_opt_state, new_data, loss_keys, step_stats, new_width

        step_fn = kfac_step_fn
    else:

        @jax.jit
        @constants.pmap
        def adam_step_fn(
            params: ParamTree,
            opt_state: Any,
            data: types.FermiNetData,
            key: jax.Array,
            step: jnp.ndarray,
            mcmc_width: Any,
        ) -> tuple[Any, Any, Any, Any, train_utils.StepStats, Any]:
            key, mcmc_key, loss_key = jax.random.split(key, 3)
            new_data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)

            new_params, new_opt_state, loss_value, aux, _ = update_fn(
                params, opt_state, loss_key, new_data, step
            )

            new_width = update_width_step(
                loss_value,
                pmove,
                mcmc_width,
                step,
                adapt_frequency=adapt_frequency,
                pmove_max=pmove_max,
                pmove_min=pmove_min,
                width_min=0.001,
                width_max=10.0,
                default_width=default_width,
            )

            energy = constants.pmean(loss_value)
            variance = constants.pmean(aux.variance)
            pmove = constants.pmean(pmove)
            lr = jnp.asarray(schedule(step))
            stats = train_utils.StepStats(
                energy=energy, variance=variance, pmove=pmove, learning_rate=lr
            )

            is_finite = jnp.isfinite(energy)
            new_params = jax.tree_util.tree_map(
                lambda p, np: jnp.where(is_finite, np, p), params, new_params
            )
            new_opt_state = jax.tree_util.tree_map(
                lambda p, np: jnp.where(is_finite, np, p), opt_state, new_opt_state
            )

            return new_params, new_opt_state, new_data, key, stats, new_width

        step_fn = adam_step_fn

    iterations = int(cfg_any.optim.iterations)
    print_every = int(cfg_any.log.print_every)
    checkpoint_every = int(cfg_any.log.checkpoint_every)
    adapt_frequency = int(cfg_any.mcmc.adapt_frequency)
    save_path = cfg_any.log.save_path

    start = time.time()
    for i in range(step, iterations):
        step_array = jnp.full((device_count,), i, dtype=jnp.int32)

        step_result = step_fn(params, opt_state, data, key, step_array, width)
        step_result = cast(tuple[Any, Any, Any, Any, Any, Any], step_result)
        new_params, new_opt_state, data, key, stats, width = step_result

        params, opt_state = new_params, new_opt_state

        if (i + 1) % print_every == 0:
            stats_host = jax.device_get(stats)

            def _to_float(arr):
                if arr.ndim > 0:
                    return float(arr.ravel()[0])
                return float(arr)

            energy_val = _to_float(stats_host.energy)

            log_stats = train_utils.StepStats(
                energy=energy_val,
                variance=_to_float(stats_host.variance),
                pmove=_to_float(stats_host.pmove),
                learning_rate=_to_float(stats_host.learning_rate),
            )

            # Width is now sharded on device, just grab first element for logging
            width_val = _to_float(jax.device_get(width[0]))

            wall = time.time() - start
            train_utils.log_stats(i + 1, log_stats, wall, width_val)
            start = time.time()

        if (i + 1) % checkpoint_every == 0:
            host_params = jax.tree_util.tree_map(lambda x: jax.device_get(x)[0], params)
            host_opt_state = jax.tree_util.tree_map(
                lambda x: jax.device_get(x)[0], opt_state
            )
            host_data = jax.tree_util.tree_map(lambda x: jax.device_get(x)[0], data)
            checkpoint.save_checkpoint(
                save_path, i + 1, host_params, host_opt_state, host_data
            )

    host_params = jax.tree_util.tree_map(lambda x: jax.device_get(x)[0], params)
    host_opt_state = jax.tree_util.tree_map(lambda x: jax.device_get(x)[0], opt_state)
    host_data = jax.tree_util.tree_map(lambda x: jax.device_get(x)[0], data)
    return {
        "params": host_params,
        "opt_state": host_opt_state,
        "data": host_data,
        "step": iterations,
    }
