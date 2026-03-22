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

def mh_accept_old(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    lp_1: jnp.ndarray,
    lp_2: jnp.ndarray,
    ratio: jnp.ndarray,
    key: jax.Array,
    num_accepts: jnp.ndarray,
    hmean1: jnp.ndarray | None = None,
    hmean2: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jax.Array, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Metropolis-Hastings accept/reject step with non-finite guards."""
    key, subkey = mcmc._split_key(key)
    rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
    finite_proposal = jnp.isfinite(lp_2) & jnp.isfinite(ratio)
    cond = (ratio > rnd) & finite_proposal
    x_new = jnp.where(cond[..., None], x2, x1)
    lp_new = jnp.where(cond, lp_2, lp_1)
    num_accepts += jnp.sum(cond)

    if hmean1 is not None and hmean2 is not None:
        hmean_new = jnp.where(cond[..., None, None, None], hmean2, hmean1)
    else:
        hmean_new = None

    return x_new, key, lp_new, num_accepts, hmean_new

def mh_accept_new(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    lp_1: jnp.ndarray,
    lp_2: jnp.ndarray,
    ratio: jnp.ndarray,
    subkey: jax.Array,
    num_accepts: jnp.ndarray,
    hmean1: jnp.ndarray | None = None,
    hmean2: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Metropolis-Hastings accept/reject step with non-finite guards."""
    rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
    finite_proposal = jnp.isfinite(lp_2) & jnp.isfinite(ratio)
    cond = (ratio > rnd) & finite_proposal
    x_new = jnp.where(cond[..., None], x2, x1)
    lp_new = jnp.where(cond, lp_2, lp_1)
    num_accepts += jnp.sum(cond)

    if hmean1 is not None and hmean2 is not None:
        hmean_new = jnp.where(cond[..., None, None, None], hmean2, hmean1)
    else:
        hmean_new = None

    return x_new, lp_new, num_accepts, hmean_new

def bench():
    x1, x2, lp_1, lp_2, ratio, key, num_accepts = setup_benchmark_data(1024, 64, 3)

    jitted_old = jax.jit(mh_accept_old)
    jitted_new = jax.jit(mh_accept_new)

    jitted_old(x1, x2, lp_1, lp_2, ratio, key, num_accepts)
    jitted_new(x1, x2, lp_1, lp_2, ratio, key, num_accepts)

    start1 = time.time()
    for _ in range(10000):
        res = jitted_old(x1, x2, lp_1, lp_2, ratio, key, num_accepts)
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, res)
    end1 = time.time()

    start2 = time.time()
    for _ in range(10000):
        res = jitted_new(x1, x2, lp_1, lp_2, ratio, key, num_accepts)
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, res)
    end2 = time.time()

    print(f"Old time for 10000 iterations: {end1 - start1:.4f}s")
    print(f"New time for 10000 iterations: {end2 - start2:.4f}s")

if __name__ == "__main__":
    bench()
