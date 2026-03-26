import jax
import jax.numpy as jnp
import time
from jax.tree_util import tree_map
from ferminet import mcmc

def run_bench():
    # simulate stats array
    stats = jnp.array([[1.0, 2.0, 0.95, 0.001], [1.0, 2.0, 0.95, 0.001]])

    # simulate pmove extraction
    t0 = time.time()
    for i in range(100):
        if stats.ndim == 2:
            pmove_ref = stats[0, 2]
        else:
            pmove_ref = stats[2]
        # _to_host
        host_tree = jax.device_get(pmove_ref)
        pmove_val = float(host_tree)
    t1 = time.time()

    print(f"Current latency: {(t1 - t0) / 100 * 1000:.2f} ms")

    # simulate no sync
    t0 = time.time()
    for i in range(100):
        if stats.ndim == 2:
            pmove_ref = stats[0, 2]
        else:
            pmove_ref = stats[2]

    t1 = time.time()

    print(f"No sync latency: {(t1 - t0) / 100 * 1000:.2f} ms")

run_bench()
