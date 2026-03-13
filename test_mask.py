import jax
import jax.numpy as jnp


def _masked_mean_old(values: jax.Array, mask: jax.Array) -> jax.Array:
    mask_expanded = mask[..., None]
    summed = jnp.sum(values * mask_expanded, axis=1)
    denom = jnp.sum(mask_expanded, axis=1)
    return summed / jnp.maximum(denom, 1.0)


def _masked_mean_new(values: jax.Array) -> jax.Array:
    # Use jnp.diagonal(..., axis1=0, axis2=1) which gives shape (feat, n)
    # Then transpose it to (n, feat)
    summed = jnp.sum(values, axis=1) - jnp.diagonal(values, axis1=0, axis2=1).T
    n = values.shape[0]
    denom = jnp.maximum(n - 1.0, 1.0)
    return summed / denom


key = jax.random.PRNGKey(0)
values = jax.random.normal(key, (4, 4, 3))
mask = 1.0 - jnp.eye(4)

old = _masked_mean_old(values, mask)
new = _masked_mean_new(values)

print("Old:", old)
print("New:", new)
print("Difference:", jnp.max(jnp.abs(old - new)))
