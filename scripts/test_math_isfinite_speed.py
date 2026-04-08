import math
import time

import jax.numpy as jnp

x = 5.0
t0 = time.time()
for _ in range(1000):
    math.isfinite(x)
t1 = time.time()
print(f"math.isfinite: {t1 - t0:.5f}s")

t0 = time.time()
for _ in range(1000):
    jnp.isfinite(x)
t1 = time.time()
print(f"jnp.isfinite: {t1 - t0:.5f}s")
