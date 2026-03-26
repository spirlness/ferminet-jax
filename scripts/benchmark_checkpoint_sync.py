import jax
import jax.numpy as jnp
import time
from jax.tree_util import tree_map

def run_bench():
    # create a dummy tree with 100 leaves
    tree = {f"k{i}": jnp.ones((4, 10, 10)) for i in range(100)}
    opt_state = {f"k{i}": jnp.ones((4, 10, 10)) for i in range(100)}
    data = {f"k{i}": jnp.ones((4, 10, 10)) for i in range(10)}

    # warm up
    jax.device_get(tree)

    # test sequential
    t0 = time.time()
    for _ in range(10):
        h1 = tree_map(lambda x: jax.device_get(x)[0], tree)
        h2 = tree_map(lambda x: jax.device_get(x)[0], opt_state)
        h3 = tree_map(lambda x: jax.device_get(x)[0], data)
    t1 = time.time()

    print(f"Sequential latency: {(t1 - t0) / 10 * 1000:.2f} ms")

    # test grouped
    t0 = time.time()
    for _ in range(10):
        host_trees = jax.device_get((tree, opt_state, data))
        h1, h2, h3 = tree_map(lambda x: x[0] if getattr(x, 'ndim', 0) > 0 else x, host_trees)
    t1 = time.time()

    print(f"Grouped latency: {(t1 - t0) / 10 * 1000:.2f} ms")

run_bench()
