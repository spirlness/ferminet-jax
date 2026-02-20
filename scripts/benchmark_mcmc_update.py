import time
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from ferminet import mcmc
from ferminet.types import FermiNetData, ParamTree


def _split_key(key: jax.Array) -> Tuple[jax.Array, jax.Array]:
    keys = jax.random.split(key)
    return keys[0], keys[1]


def mh_accept_unoptimized(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    lp_1: jnp.ndarray,
    lp_2: jnp.ndarray,
    ratio: jnp.ndarray,
    key: jax.Array,
    num_accepts: jnp.ndarray,
) -> Tuple[jnp.ndarray, jax.Array, jnp.ndarray, jnp.ndarray]:
    """Metropolis-Hastings accept/reject step with non-finite guards (Unoptimized)."""
    key, subkey = _split_key(key)
    rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
    finite_proposal = jnp.isfinite(lp_2) & jnp.isfinite(ratio)
    cond = (ratio > rnd) & finite_proposal
    x_new = jnp.where(cond[..., None], x2, x1)
    lp_new = jnp.where(cond, lp_2, lp_1)
    num_accepts += jnp.sum(cond)
    return x_new, key, lp_new, num_accepts


def mh_update_unoptimized(
    params: ParamTree,
    f: Callable,
    data: FermiNetData,
    key: jax.Array,
    lp_1: jnp.ndarray,
    num_accepts: jnp.ndarray,
    stddev: float = 0.02,
    atoms: jnp.ndarray | None = None,
    ndim: int = 3,
) -> Tuple[FermiNetData, jax.Array, jnp.ndarray, jnp.ndarray]:
    """Unoptimized mh_update."""
    key, subkey = _split_key(key)
    positions, spins, atoms_data, charges = mcmc._asarray_data(data)
    x1: jnp.ndarray = positions

    if atoms is None:
        noise = jax.random.normal(subkey, shape=x1.shape)
        x2 = x1 + stddev * noise
        lp_2 = 2.0 * f(params, x2, spins, atoms_data, charges)
        ratio = lp_2 - lp_1
    else:
        # Dummy implementation for asymmetric proposal if needed, but benchmark without atoms for simplicity
        # Or copy logic if needed. For now, assume symmetric proposal as it dominates computation if network is small
        # But wait, we want to measure RNG overhead. So let's keep it simple.
        noise = jax.random.normal(subkey, shape=x1.shape)
        x2 = x1 + stddev * noise
        lp_2 = 2.0 * f(params, x2, spins, atoms_data, charges)
        ratio = lp_2 - lp_1

    x_new, key, lp_new, num_accepts = mh_accept_unoptimized(
        x1, x2, lp_1, lp_2, ratio, key, num_accepts
    )

    new_data = data._replace(positions=x_new)
    return new_data, key, lp_new, num_accepts


def dummy_network(params, positions, spins, atoms, charges):
    # Simulate network reshaping
    if positions.ndim == 2:
        positions = positions.reshape(positions.shape[0], -1, 3)
    return jnp.sum(positions**2, axis=[-1, -2])


def benchmark():
    print("Setting up benchmark...")
    batch_size = 4096  # Large batch to make RNG significant? Or small to make overhead significant?
    # Large batch makes compute dominate. Small batch makes overhead dominate.
    # The optimization is about overhead of splitting keys.
    # So small-ish batch might be better? Or maybe moderate.
    batch_size = 128
    nelec = 32
    ndim = 3
    key = jax.random.PRNGKey(42)

    positions = jax.random.normal(key, (batch_size, nelec * ndim))
    spins = jnp.zeros((batch_size, nelec), dtype=jnp.int32)  # Dummy
    atoms = jnp.zeros((1, ndim))
    charges = jnp.zeros((1,))

    data = FermiNetData(positions=positions, spins=spins, atoms=atoms, charges=charges)
    lp_1 = dummy_network({}, positions, spins, atoms, charges) * 2.0
    num_accepts = jnp.zeros(())

    # Compile functions
    update_opt = jax.jit(mcmc.mh_update, static_argnames=("f", "ndim"))
    update_unopt = jax.jit(mh_update_unoptimized, static_argnames=("f", "ndim"))

    # Warmup
    print("Warming up...")
    key, subkey = jax.random.split(key)
    _ = update_opt(
        {},
        dummy_network,
        data,
        subkey,
        lp_1,
        num_accepts,
        stddev=0.1,
        atoms=None,
        ndim=ndim,
    )
    key, subkey = jax.random.split(key)
    _ = update_unopt(
        {},
        dummy_network,
        data,
        subkey,
        lp_1,
        num_accepts,
        stddev=0.1,
        atoms=None,
        ndim=ndim,
    )

    steps = 10000

    print(f"Running {steps} steps for Optimized version...")
    key_loop = key

    def loop_body(i, carry):
        d, k, lp, n = carry
        return update_opt(
            {}, dummy_network, d, k, lp, n, stddev=0.1, atoms=None, ndim=ndim
        )

    # Use python loop for timing Python overhead too?
    # Or use lax.scan to measure pure device execution time?
    # The optimization is in JAX graph construction (fewer split ops), so compiled code should be faster.
    # We should measure the compiled execution time.

    # Let's use lax.fori_loop to keep it on device and measure pure execution time.

    @jax.jit
    def run_opt(data, key, lp_1, num_accepts):
        def body(i, val):
            d, k, lp, n = val
            return update_opt(
                {}, dummy_network, d, k, lp, n, stddev=0.1, atoms=None, ndim=ndim
            )

        return jax.lax.fori_loop(0, steps, body, (data, key, lp_1, num_accepts))

    @jax.jit
    def run_unopt(data, key, lp_1, num_accepts):
        def body(i, val):
            d, k, lp, n = val
            return update_unopt(
                {}, dummy_network, d, k, lp, n, stddev=0.1, atoms=None, ndim=ndim
            )

        return jax.lax.fori_loop(0, steps, body, (data, key, lp_1, num_accepts))

    # Run Optimized
    res = run_opt(data, key_loop, lp_1, num_accepts)
    jax.block_until_ready(res)  # Warmup compiled function

    t0 = time.perf_counter()
    res = run_opt(data, key_loop, lp_1, num_accepts)
    jax.block_until_ready(res)
    t1 = time.perf_counter()
    opt_time = t1 - t0
    print(f"Optimized time: {opt_time:.6f} s")

    # Run Unoptimized
    res = run_unopt(data, key_loop, lp_1, num_accepts)
    jax.block_until_ready(res)  # Warmup compiled function

    t0 = time.perf_counter()
    res = run_unopt(data, key_loop, lp_1, num_accepts)
    jax.block_until_ready(res)
    t1 = time.perf_counter()
    unopt_time = t1 - t0
    print(f"Unoptimized time: {unopt_time:.6f} s")

    diff = unopt_time - opt_time
    percent = (diff / unopt_time) * 100
    print(f"Improvement: {diff:.6f} s ({percent:.2f}%)")


if __name__ == "__main__":
    benchmark()
