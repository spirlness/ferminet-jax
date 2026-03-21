import jax
import jax.numpy as jnp
import math

val = 3.14

def jnp_check():
    return jnp.isfinite(val)

def math_check():
    return math.isfinite(val)

print(jnp_check())
print(math_check())
