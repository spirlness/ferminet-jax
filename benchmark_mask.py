
import time
import jax
import jax.numpy as jnp
import ml_collections
from ferminet import networks

def benchmark():
    # Setup configuration
    cfg = ml_collections.ConfigDict()
    cfg.system = ml_collections.ConfigDict()
    cfg.system.ndim = 3
    cfg.network = ml_collections.ConfigDict()
    cfg.network.determinants = 16
    cfg.network.envelope_type = "isotropic"
    cfg.network.ferminet = ml_collections.ConfigDict()
    cfg.network.ferminet.hidden_dims = ((256, 32), (256, 32))
    cfg.network.ferminet.determinants = 16
    cfg.network.ferminet.hidden_activation = "tanh"
    cfg.network.ferminet.bias_orbitals = True

    # Setup system
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([2.0])
    spins = (10, 10)  # Moderate size to see impact
    n_electrons = sum(spins)

    print(f"Benchmarking with {n_electrons} electrons...")

    # Create network
    init_fn, apply_fn, _ = networks.make_fermi_net(atoms, charges, spins, cfg)
    params = init_fn(jax.random.PRNGKey(0))

    # Mock inputs
    batch_size = 128
    electrons = jax.random.normal(jax.random.PRNGKey(1), (batch_size, n_electrons * 3))
    spins_arr = jnp.concatenate([jnp.zeros(spins[0]), jnp.ones(spins[1])])
    spins_batch = jnp.tile(spins_arr, (batch_size, 1))

    # JIT compile the apply function
    # We use vmap inside apply_fn, so we just pass batched input
    # Note: apply_fn expects normalized inputs, but handles normalization internally

    # Warmup
    print("Warming up...")
    start_time = time.time()
    sign, log_psi = apply_fn(params, electrons, spins_batch, atoms, charges)
    sign.block_until_ready()
    print(f"Warmup done in {time.time() - start_time:.4f}s")

    # Benchmark loop
    n_loops = 100
    print(f"Running {n_loops} iterations...")

    # We wrap in jax.jit to test the compiled performance
    # However, if mask creation is inside the function, JIT might constant-fold it
    # IF n_electrons is static. But here n_electrons is implicit in shapes.
    # make_fermi_net fixes n_electrons.

    @jax.jit
    def step(p, e, s, a, c):
        return apply_fn(p, e, s, a, c)

    # Re-warmup with jit
    step(params, electrons, spins_batch, atoms, charges)[0].block_until_ready()

    start_time = time.time()
    for _ in range(n_loops):
        out = step(params, electrons, spins_batch, atoms, charges)
        out[0].block_until_ready()

    end_time = time.time()
    avg_time = (end_time - start_time) / n_loops
    print(f"Average time per step: {avg_time*1000:.4f} ms")

if __name__ == "__main__":
    benchmark()
