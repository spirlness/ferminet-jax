import time

import jax
import jax.numpy as jnp


def apply_jastrow_old(r_ee):
    nelec = r_ee.shape[0]
    mask = jnp.triu(jnp.ones((nelec, nelec)), k=1)
    jastrow_terms = 1.0 / (1.0 + 2.0 * r_ee) * mask
    return jnp.sum(jastrow_terms)

def apply_jastrow_new(r_ee):
    jastrow_terms = 1.0 / (1.0 + 2.0 * r_ee)
    return jnp.sum(jnp.triu(jastrow_terms, k=1))

def apply_hamiltonian_old(ee):
    n = ee.shape[0]
    r_ee = jnp.linalg.norm(ee, axis=-1) * (1.0 - jnp.eye(n))
    return jnp.sum(r_ee)

def apply_hamiltonian_new(ee):
    r_ee = jnp.fill_diagonal(jnp.linalg.norm(ee, axis=-1), 0.0, inplace=False)
    return jnp.sum(r_ee)

if __name__ == "__main__":
    n = 200
    key = jax.random.PRNGKey(0)
    r_ee = jax.random.normal(key, (n, n))
    ee = jax.random.normal(key, (n, n, 3))

    jastrow_old_jit = jax.jit(apply_jastrow_old)
    jastrow_new_jit = jax.jit(apply_jastrow_new)
    ham_old_jit = jax.jit(apply_hamiltonian_old)
    ham_new_jit = jax.jit(apply_hamiltonian_new)

    jastrow_old_jit(r_ee).block_until_ready()
    jastrow_new_jit(r_ee).block_until_ready()
    ham_old_jit(ee).block_until_ready()
    ham_new_jit(ee).block_until_ready()

    print("Benchmarking Jastrow...")
    t0 = time.time()
    for _ in range(1000):
        jastrow_old_jit(r_ee).block_until_ready()
    print("Old:", time.time() - t0)

    t0 = time.time()
    for _ in range(1000):
        jastrow_new_jit(r_ee).block_until_ready()
    print("New:", time.time() - t0)

    print("\nBenchmarking Hamiltonian...")
    t0 = time.time()
    for _ in range(1000):
        ham_old_jit(ee).block_until_ready()
    print("Old:", time.time() - t0)

    t0 = time.time()
    for _ in range(1000):
        ham_new_jit(ee).block_until_ready()
    print("New:", time.time() - t0)
