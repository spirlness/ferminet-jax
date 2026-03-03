import time

import jax
import jax.numpy as jnp

from ferminet.networks import _masked_mean


def optimized_masked_mean(values: jnp.ndarray) -> jnp.ndarray:
    n = values.shape[0]
    summed = jnp.sum(values, axis=1) - jnp.diagonal(values, axis1=0, axis2=1).T
    denom = jnp.maximum(n - 1.0, 1.0)
    return summed / denom


def main():
    n = 1000
    feat = 100

    key = jax.random.PRNGKey(0)
    values = jax.random.normal(key, (n, n, feat))

    eye = jnp.eye(n)
    mask = 1.0 - eye

    masked_mean_jit = jax.jit(_masked_mean)
    optimized_masked_mean_jit = jax.jit(optimized_masked_mean)

    # Warmup
    _ = masked_mean_jit(values, mask)
    jax.block_until_ready(_)

    _ = optimized_masked_mean_jit(values)
    jax.block_until_ready(_)

    start = time.time()
    for _ in range(100):
        res = masked_mean_jit(values, mask)
        jax.block_until_ready(res)
    end = time.time()
    print(f"Time for original: {end - start:.4f} seconds")

    start = time.time()
    for _ in range(100):
        res = optimized_masked_mean_jit(values)
        jax.block_until_ready(res)
    end = time.time()
    print(f"Time for optimized: {end - start:.4f} seconds")

    # check correctness
    diff = jnp.max(
        jnp.abs(masked_mean_jit(values, mask) - optimized_masked_mean_jit(values))
    )
    print(f"Max difference: {diff}")


if __name__ == "__main__":
    main()
