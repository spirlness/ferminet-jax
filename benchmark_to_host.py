import jax
import jax.numpy as jnp
import time

def benchmark():
    stats = jnp.array([[1.0, 2.0, 3.0, 4.0]])

    start = time.perf_counter()
    for _ in range(1000):
        val = jax.device_get(stats[0, 2])
        float(val)
    end = time.perf_counter()
    print(f"Slice + get: {end - start:.4f}s")

    start = time.perf_counter()
    for _ in range(1000):
        val = jax.device_get(stats)
        float(val[0, 2])
    end = time.perf_counter()
    print(f"Get whole + slice: {end - start:.4f}s")

benchmark()
