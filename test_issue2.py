import jax
import jax.numpy as jnp
from ferminet import train

energy = jnp.array(1.0)
print(jnp.isfinite(energy))
