import math
import time

import jax
import jax.numpy as jnp

val = jnp.array([1.0])

t0 = time.perf_counter()
for _ in range(100):
    # This is effectively what happens when energy_val = float(stats_host[ENERGY])
    # and then we do jnp.isfinite(energy_val).
    # Since energy_val is a float, jnp.isfinite(energy_val) calls JAX machinery
    # on a standard python float, likely creating a temporary JAX array, executing the
    # operation, and doing another host-device synchronization to extract the boolean scalar.
    v = float(jax.device_get(val)[0])
    if not jnp.isfinite(v):
        pass
t1 = time.perf_counter()

print(f"jnp.isfinite on Python float: {t1 - t0:.5f} s")

t0 = time.perf_counter()
for _ in range(100):
    v = float(jax.device_get(val)[0])
    if not math.isfinite(v):
        pass
t1 = time.perf_counter()

print(f"math.isfinite on Python float: {t1 - t0:.5f} s")
