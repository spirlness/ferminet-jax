import time

import jax
import jax.numpy as jnp


def _masked_mean_old(values: jax.Array, mask: jax.Array) -> jax.Array:
    mask_expanded = mask[..., None]
    summed = jnp.sum(values * mask_expanded, axis=1)
    denom = jnp.sum(mask_expanded, axis=1)
    return summed / jnp.maximum(denom, 1.0)


def _masked_mean_new(values: jax.Array) -> jax.Array:
    summed = jnp.sum(values, axis=1) - jnp.diagonal(values, axis1=0, axis2=1).T
    n = values.shape[0]
    denom = jnp.maximum(n - 1.0, 1.0)
    return summed / denom


@jax.jit
def bench_old(values, mask):
    return _masked_mean_old(values, mask)


@jax.jit
def bench_new(values):
    return _masked_mean_new(values)


key = jax.random.PRNGKey(0)
N = 128
F = 64
values = jax.random.normal(key, (N, N, F))
mask = 1.0 - jnp.eye(N)

# warmup
bench_old(values, mask).block_until_ready()
bench_new(values).block_until_ready()

n_iters = 1000

t0 = time.time()
for _ in range(n_iters):
    bench_old(values, mask).block_until_ready()
t1 = time.time()

t2 = time.time()
for _ in range(n_iters):
    bench_new(values).block_until_ready()
t3 = time.time()

print(f"Old: {(t1 - t0) / n_iters * 1000:.3f} ms")
print(f"New: {(t3 - t2) / n_iters * 1000:.3f} ms")
