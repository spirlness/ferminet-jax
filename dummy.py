import jax
import jax.numpy as jnp

def test():
    a = jnp.array(1.0)
    print(type(a))
    print(isinstance(a, jax.Array))

test()
