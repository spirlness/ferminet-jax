import jax
import jax.numpy as jnp

@jax.jit
def foo(val):
    if val > 0.3:
        return 1
    return 0

foo(jnp.array(0.4))
