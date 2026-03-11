import jax
import jax.numpy as jnp
import time
import numpy as np

def _electron_electron_mask(n_electrons: int):
    eye = jnp.eye(n_electrons)
    return 1.0 - eye

@jax.jit
def _masked_mean_old(values, mask):
    mask_expanded = mask[..., None]
    summed = jnp.sum(values * mask_expanded, axis=1)
    denom = jnp.sum(mask_expanded, axis=1)
    return summed / jnp.maximum(denom, 1.0)

@jax.jit
def _masked_mean_new(values):
    n = values.shape[0]
    summed = jnp.sum(values, axis=1) - jnp.diagonal(values, axis1=0, axis2=1).T
    denom = jnp.maximum(n - 1.0, 1.0)
    return summed / denom

values = jnp.array(np.random.rand(128, 128, 64))
mask = _electron_electron_mask(128)

# warm up
_masked_mean_old(values, mask).block_until_ready()
_masked_mean_new(values).block_until_ready()

start = time.time()
for _ in range(1000):
    _masked_mean_old(values, mask).block_until_ready()
end = time.time()
print(f"Old: {end - start:.5f}s")

start = time.time()
for _ in range(1000):
    _masked_mean_new(values).block_until_ready()
end = time.time()
print(f"New: {end - start:.5f}s")
