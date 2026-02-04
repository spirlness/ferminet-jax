
import time
import jax
import jax.numpy as jnp
import jax.random as random
from src.ferminet.mcmc import FixedStepMCMC

def benchmark():
    # Setup
    batch_size = 256
    n_elec = 32
    dim = 3
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    r = random.normal(subkey, (batch_size, n_elec, dim))

    # Simple dense network
    input_dim = n_elec * dim
    hidden_dim = 64

    params = {
        "w1": random.normal(key, (input_dim, hidden_dim)),
        "b1": random.normal(key, (hidden_dim,)),
        "w2": random.normal(key, (hidden_dim, 1)),
        "b2": random.normal(key, (1,))
    }

    def network_apply(p, x):
        # x is expected to have a leading batch dimension: [batch, n_elec, dim]
        # Flatten to [batch, input_dim] where input_dim = n_elec * dim
        # FixedStepMCMC and grad_batch always call this with batched inputs.

        flat_x = x.reshape(x.shape[0], -1)

        h = jnp.tanh(flat_x @ p["w1"] + p["b1"])
        out = h @ p["w2"] + p["b2"]
        return out.reshape(-1) # [batch]

    mcmc = FixedStepMCMC(step_size=0.1, n_steps=10)

    # Run loop
    n_iter = 50
    print(f"Benchmarking MCMC sample over {n_iter} iterations...")

    # Pre-compute gradient transformation ONCE
    def single_network_apply(p, x_single):
        # x_single: [n_elec, 3] -> need to batch it for network_apply or adjust network_apply
        # network_apply expects batched input [batch, ...]
        # So we create a batch of 1
        return network_apply(p, x_single[None, ...])[0]

    grad_single = jax.grad(single_network_apply, argnums=1)
    grad_batch = jax.vmap(grad_single, in_axes=(None, 0))

    start_time = time.time()

    for i in range(n_iter):
        # Simulate parameter update by slightly perturbing params
        # This mimics the training loop where params change every step
        # and a NEW closure is created for log_psi_fn

        # We don't actually need to perturb params for the benchmark logic to be valid
        # as long as we create a new closure and pass it.
        # But let's create a new closure each time as VMCTrainer does.

        def log_psi_fn(x):
            return network_apply(params, x)

        # Create grad_log_psi_fn bound to params
        # This is cheap (python lambda) and uses the pre-transformed JAX function
        grad_log_psi_fn = lambda x: grad_batch(params, x)

        key, subkey = random.split(key)
        r, _ = mcmc.sample(log_psi_fn, r, subkey, grad_log_psi_fn=grad_log_psi_fn)

        # Force wait to measure actual time including dispatch
        r.block_until_ready()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time:.4f}s")
    print(f"Time per iteration: {total_time / n_iter:.4f}s")

if __name__ == "__main__":
    benchmark()
