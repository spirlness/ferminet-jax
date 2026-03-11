import jax.numpy as jnp
import numpy as np

def _electron_electron_mask(n_electrons: int):
    eye = jnp.eye(n_electrons)
    return 1.0 - eye

def _masked_mean_old(values, mask):
    mask_expanded = mask[..., None]
    summed = jnp.sum(values * mask_expanded, axis=1)
    denom = jnp.sum(mask_expanded, axis=1)
    return summed / jnp.maximum(denom, 1.0)

def _masked_mean_new(values):
    n = values.shape[0]
    summed = jnp.sum(values, axis=1) - jnp.diagonal(values, axis1=0, axis2=1).T
    denom = jnp.maximum(n - 1.0, 1.0)
    return summed / denom

values = jnp.array(np.random.rand(4, 4, 3))
mask = _electron_electron_mask(4)
res1 = _masked_mean_old(values, mask)
res2 = _masked_mean_new(values)

print(jnp.allclose(res1, res2))
