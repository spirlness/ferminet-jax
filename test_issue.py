import jax
import jax.numpy as jnp
from ferminet import train

@jax.jit
def test_fn():
    energy = jnp.array(1.0)
    is_finite = jnp.isfinite(energy)
    return is_finite

print(test_fn())
