import time
import jax
import jax.numpy as jnp
from ferminet import mcmc
from ferminet.types import FermiNetData

def setup_benchmark_data(batch_size=128, num_electrons=16, ndim=3):
    key = jax.random.PRNGKey(42)
    positions = jax.random.normal(key, (batch_size, num_electrons * ndim))
    lp_1 = jax.random.normal(key, (batch_size,))
    lp_2 = jax.random.normal(key, (batch_size,))
    ratio = lp_2 - lp_1
    num_accepts = jnp.array(0.0)

    return positions, positions + 0.1, lp_1, lp_2, ratio, key, num_accepts

def bench():
    x1, x2, lp_1, lp_2, ratio, key, num_accepts = setup_benchmark_data(1024, 64, 3)

    # JIT compile the target function
    jitted_accept = jax.jit(mcmc.mh_accept)

    # Warmup
    jitted_accept(x1, x2, lp_1, lp_2, ratio, key, num_accepts)

    start = time.time()
    for _ in range(10000):
        res = jitted_accept(x1, x2, lp_1, lp_2, ratio, key, num_accepts)
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, res)
    end = time.time()

    print(f"Time for 10000 iterations: {end - start:.4f}s")

if __name__ == "__main__":
    bench()
