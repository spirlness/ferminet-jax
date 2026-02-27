import time

import jax
import jax.numpy as jnp

from ferminet import mcmc, types


def _split_key(key):
    keys = jax.random.split(key)
    return keys[0], keys[1]


def mh_accept_old(
    x1, x2, lp_1, lp_2, ratio, key, num_accepts, hmean1=None, hmean2=None
):
    key, subkey = _split_key(key)
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

    return x_new, lp_new, num_accepts, hmean_new, key


def mh_update_old(
    params, f, data, key, lp_1, num_accepts, hmean_1, stddev=0.02, atoms=None, ndim=3
):
    # Old logic: split key once for proposal, pass key to mh_accept_old
    key, subkey = jax.random.split(key)

    positions, spins, atoms_data, charges = mcmc._asarray_data(data)
    x1 = positions

    if atoms is None:
        noise = jax.random.normal(subkey, shape=x1.shape)
        stddev_arr = jnp.asarray(stddev)
        if stddev_arr.ndim >= 1 and x1.ndim == 2:
            stddev_broad = jnp.repeat(stddev_arr, ndim)[None, :]
        else:
            stddev_broad = stddev_arr
        x2 = x1 + stddev_broad * noise
        lp_2 = 2.0 * f(params, x2, spins, atoms_data, charges)
        ratio = lp_2 - lp_1
        hmean_2 = None
        x2_flat = x2
    else:
        # Asymmetric proposal (simplified for benchmark, assuming symmetric logic mostly used or similar overhead)
        # But to be fair, let's keep it close to original
        n = x1.shape[0]
        x1 = jnp.reshape(x1, [n, -1, 1, ndim])
        if hmean_1 is None:
            hmean_1 = mcmc._harmonic_mean(x1, atoms)

        noise = jax.random.normal(subkey, shape=x1.shape)
        x2 = x1 + stddev * hmean_1 * noise
        x2_flat = jnp.reshape(x2, [n, -1])
        lp_2 = 2.0 * f(params, x2_flat, spins, atoms_data, charges)
        hmean_2 = mcmc._harmonic_mean(x2, atoms)

        lq_1 = mcmc._log_prob_gaussian(x1, x2, stddev * hmean_1)
        lq_2 = mcmc._log_prob_gaussian(x2, x1, stddev * hmean_2)
        ratio = lp_2 + lq_2 - lp_1 - lq_1
        x1 = jnp.reshape(x1, [n, -1])

    # Pass key to mh_accept_old
    x_new, lp_new, num_accepts, hmean_new, key = mh_accept_old(
        x1, x2_flat, lp_1, lp_2, ratio, key, num_accepts, hmean_1, hmean_2
    )

    new_data = data._replace(positions=x_new)
    return new_data, key, lp_new, num_accepts, hmean_new


def benchmark():
    print(f"Devices: {jax.devices()}")

    # Setup dummy data
    # Increase iterations, decrease batch size slightly to emphasize overhead?
    # Or keep batch large to simulate real workload.
    batch_size = 2048
    nelec = 32
    ndim = 3
    positions = jnp.zeros((batch_size, nelec * ndim))
    spins = jnp.zeros((batch_size,), dtype=jnp.int32)  # dummy
    atoms = jnp.zeros((1, ndim))
    charges = jnp.array([1.0])
    data = types.FermiNetData(
        positions=positions, spins=spins, atoms=atoms, charges=charges
    )

    key = jax.random.PRNGKey(0)
    lp_1 = jnp.zeros((batch_size,))
    num_accepts = jnp.array(0.0)
    hmean_1 = None

    # Dummy network function
    def dummy_network(params, positions, spins, atoms, charges):
        # minimal work
        return jnp.sum(positions, axis=-1)

    # JIT compile both functions
    mh_update_new_jit = jax.jit(mcmc.mh_update, static_argnames=("f", "ndim"))
    mh_update_old_jit = jax.jit(mh_update_old, static_argnames=("f", "ndim"))

    # Warmup
    print("Warming up...")
    _ = mh_update_new_jit({}, dummy_network, data, key, lp_1, num_accepts, hmean_1)
    _ = mh_update_old_jit({}, dummy_network, data, key, lp_1, num_accepts, hmean_1)

    steps = 5000

    # Benchmark New
    print(f"Benchmarking New (Optimized) for {steps} steps...")

    def step_new(carry, _):
        data, key, lp, acc, hmean = carry
        out = mh_update_new_jit({}, dummy_network, data, key, lp, acc, hmean)
        return out, None

    carry_new = (data, key, lp_1, num_accepts, hmean_1)
    # Compile scan
    scan_new = jax.jit(lambda c: jax.lax.scan(step_new, c, None, length=steps))
    _ = scan_new(carry_new)  # warmup scan

    jax.block_until_ready(carry_new[0].positions)  # ensure ready
    start_time = time.perf_counter()
    final_carry_new, _ = scan_new(carry_new)
    jax.block_until_ready(final_carry_new[0].positions)
    end_time = time.perf_counter()
    time_new = end_time - start_time
    print(f"New Time: {time_new:.6f} s")

    # Benchmark Old
    print(f"Benchmarking Old (Unoptimized) for {steps} steps...")

    def step_old(carry, _):
        data, key, lp, acc, hmean = carry
        out = mh_update_old_jit({}, dummy_network, data, key, lp, acc, hmean)
        return out, None

    carry_old = (data, key, lp_1, num_accepts, hmean_1)
    scan_old = jax.jit(lambda c: jax.lax.scan(step_old, c, None, length=steps))
    _ = scan_old(carry_old)  # warmup scan

    jax.block_until_ready(carry_old[0].positions)
    start_time = time.perf_counter()
    final_carry_old, _ = scan_old(carry_old)
    jax.block_until_ready(final_carry_old[0].positions)
    end_time = time.perf_counter()
    time_old = end_time - start_time
    print(f"Old Time: {time_old:.6f} s")

    print(f"Speedup: {time_old / time_new:.4f}x")


if __name__ == "__main__":
    benchmark()
