import time
from typing import Any, cast

import jax
import jax.numpy as jnp

from ferminet.configs import helium
from ferminet.networks import make_fermi_net


def benchmark():
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([2.0])
    nspins = (1, 1)
    spins_arr = jnp.array([0, 1])

    cfg = helium.get_config()
    cfg_any = cast(Any, cfg)
    cfg_any.network.determinants = 2
    cfg_any.network.ferminet.hidden_dims = ((256, 32), (256, 32), (256, 32), (256, 32))

    init_fn, apply_fn, _ = make_fermi_net(atoms, charges, nspins, cfg)
    key = jax.random.PRNGKey(0)
    params = init_fn(key)

    # Batch of 4096 configurations
    batch_size = 4096
    key, subkey = jax.random.split(key)
    positions = jax.random.normal(subkey, (batch_size, sum(nspins) * 3))

    jitted_apply = jax.jit(apply_fn)

    # Warmup
    _ = jitted_apply(params, positions, spins_arr, atoms, charges)
    jax.block_until_ready(_)

    num_iters = 100
    start_time = time.time()
    for _ in range(num_iters):
        _ = jitted_apply(params, positions, spins_arr, atoms, charges)
        jax.block_until_ready(_)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iters
    print(f"Average time per apply: {avg_time * 1000:.3f} ms")


if __name__ == "__main__":
    benchmark()
