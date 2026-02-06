# Copyright 2024 FermiNet Authors.
# Licensed under the Apache License, Version 2.0.

"""JAX device and sharding utilities."""

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import kfac_jax

from ferminet import types


def replicate_tree(tree: Any, devices: Sequence[jax.Device]) -> Any:
    """Replicate a PyTree across all devices."""
    del devices
    return kfac_jax.utils.replicate_all_local_devices(tree)


def shard_array(arr: jnp.ndarray, device_count: int) -> jnp.ndarray:
    """Shard an array across devices along the first dimension."""
    if arr.shape[0] % device_count != 0:
        raise ValueError(
            "Batch dimension must be divisible by device count."
            f" batch={arr.shape[0]} devices={device_count}"
        )
    per_device = arr.shape[0] // device_count
    new_shape = (device_count, per_device) + arr.shape[1:]
    reshaped = jnp.reshape(arr, new_shape)
    return kfac_jax.utils.broadcast_all_local_devices(reshaped)


def p_split(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split a key across devices using kfac_jax utility."""
    return kfac_jax.utils.p_split(key)


def shard_data(
    data: types.FermiNetData,
    device_count: int,
) -> types.FermiNetData:
    """Shard FermiNetData across multiple devices."""
    return types.FermiNetData(
        positions=shard_array(jnp.asarray(data.positions), device_count),
        spins=replicate_tree(jnp.asarray(data.spins), jax.devices()),
        atoms=replicate_tree(jnp.asarray(data.atoms), jax.devices()),
        charges=replicate_tree(jnp.asarray(data.charges), jax.devices()),
    )
