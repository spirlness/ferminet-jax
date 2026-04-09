import jax
import jax.numpy as jnp
import time
from functools import partial

@partial(jax.jit, static_argnums=0)
def compute_stats_tuple(iters=1000):
    return jax.lax.fori_loop(0, iters, lambda i, val: (val[0]+1.0, val[1]+1.0, val[2]+0.5, val[3]+0.1), (0.0, 0.0, 0.0, 0.0))

# Warmup
t = compute_stats_tuple(10)

def benchmark_tuple_single_transfer(n_trials=100):
    start = time.time()
    for _ in range(n_trials):
        stats = compute_stats_tuple(100)
        stats_host = jax.device_get(stats)
        _ = float(stats_host[0])
        _ = float(stats_host[1])
        _ = float(stats_host[2])
        _ = float(stats_host[3])
    return time.time() - start

print(f"Tuple (single transfer): {benchmark_tuple_single_transfer()}")
