from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp


def init_linear_layer(
    key: jax.Array,
    in_dim: int,
    out_dim: int,
    include_bias: bool = True,
) -> Mapping[str, jnp.ndarray]:
    """Initialize a linear layer with Xavier initialization.

    Returns:
        Dict with 'w' (weights) and optionally 'b' (bias).
    """
    stddev = jnp.sqrt(2.0 / (in_dim + out_dim))
    split_keys = jax.random.split(key)
    key = split_keys[0]
    subkey = split_keys[1]
    w = jax.random.normal(subkey, (in_dim, out_dim)) * stddev
    params = {"w": w}
    if include_bias:
        params["b"] = jnp.zeros(out_dim)
    return params


def linear_layer(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Apply a linear layer: y = x @ w + b."""
    y = x @ w
    if b is not None:
        y = y + b
    return y


def array_partitions(sizes: Sequence[int]) -> tuple[int, ...]:
    """Returns partition indices for splitting an array.

    For sizes [a, b, c], returns (a, a + b) for use with jnp.split.
    """
    partitions: list[int] = []
    total = 0
    for size in sizes[:-1]:
        total += size
        partitions.append(total)
    return tuple(partitions)


def split_into_blocks(
    arr: jnp.ndarray,
    sizes: Sequence[int],
) -> tuple[jnp.ndarray, ...]:
    """Split array into blocks along first two axes.

    For 2 spin channels (up, down):
    Returns (up_up, up_down, down_up, down_down) blocks.
    """
    partitions = array_partitions(sizes)
    rows = jnp.split(arr, partitions, axis=0)
    blocks: list[jnp.ndarray] = []
    for row in rows:
        cols = jnp.split(row, partitions, axis=1)
        blocks.extend(cols)
    return tuple(blocks)
