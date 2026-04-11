import time

import jax
import jax.numpy as jnp

from ferminet.jastrows import _simple_ee_jastrow

init, apply = _simple_ee_jastrow()
params = init()
nelec = 1000
key = jax.random.PRNGKey(0)
r_ee = jax.random.uniform(key, (nelec, nelec))
spins = jnp.zeros((nelec,))


@jax.jit
def run_apply(r_ee):
    return apply(params, r_ee, spins)


# Warmup
run_apply(r_ee)

t0 = time.time()
for _ in range(1000):
    run_apply(r_ee)
jax.block_until_ready(run_apply(r_ee))
t1 = time.time()
print(f"Time: {t1 - t0:.4f}s")
