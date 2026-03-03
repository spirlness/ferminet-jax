import time

import jax
import jax.numpy as jnp

from ferminet import base_config
from ferminet.networks import make_fermi_net


def main():
    cfg = base_config.default()
    cfg.system.electrons = (16, 16)  # 32 electrons total
    cfg.network.ferminet.hidden_dims = ((256, 32), (256, 32), (256, 32), (256, 32))

    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([2.0])
    spins = (16, 16)

    init_fn, apply_fn, _ = make_fermi_net(atoms, charges, spins, cfg)

    key = jax.random.PRNGKey(0)
    params = init_fn(key)

    electrons = jax.random.normal(key, (1024, 32 * 3))  # batch of 1024

    apply_jit = jax.jit(apply_fn)

    # Warmup
    _ = apply_jit(params, electrons, jnp.arange(32) % 2, atoms, charges)
    jax.block_until_ready(_)

    start = time.time()
    for _ in range(100):
        res = apply_jit(params, electrons, jnp.arange(32) % 2, atoms, charges)
        jax.block_until_ready(res)
    end = time.time()

    print(f"Time for 100 forward passes: {end - start:.4f} seconds")


if __name__ == "__main__":
    main()
