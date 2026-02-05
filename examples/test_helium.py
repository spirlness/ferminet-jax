"""Helium atom training test."""

import jax
import jax.numpy as jnp

from ferminet.networks import make_fermi_net, make_log_psi_apply
from ferminet.types import FermiNetData
from ferminet.configs import helium


def test_helium_training():
    print("=" * 60)
    print("FermiNet Helium Atom Training Test")
    print("=" * 60)

    cfg = helium.get_config()
    cfg.batch_size = 256
    cfg.optim.iterations = 100
    cfg.network.determinants = 4
    cfg.network.ferminet.hidden_dims = ((32, 8), (32, 8))
    cfg.log.print_every = 10

    print(f"\nConfiguration:")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Iterations: {cfg.optim.iterations}")
    print(f"  Determinants: {cfg.network.determinants}")
    print(f"  Hidden dims: {cfg.network.ferminet.hidden_dims}")

    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([2.0])
    spins = (1, 1)

    print(f"\nSystem: Helium atom")
    print(f"  Atoms: {atoms.shape}")
    print(f"  Charges: {charges}")
    print(f"  Electrons: {spins} (up, down)")

    key = jax.random.PRNGKey(42)

    print("\nInitializing network...")
    init_fn, apply_fn, orbitals_fn = make_fermi_net(atoms, charges, spins, cfg)

    key, subkey = jax.random.split(key)
    params = init_fn(subkey)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Total parameters: {n_params:,}")

    log_psi_fn = make_log_psi_apply(apply_fn)

    n_electrons = sum(spins)
    ndim = 3

    key, subkey = jax.random.split(key)
    electrons = jax.random.normal(subkey, (cfg.batch_size, n_electrons * ndim)) * 0.5
    spins_arr = jnp.array([0, 1])

    print("\nTesting forward pass...")
    sign, log_psi = apply_fn(params, electrons[0], spins_arr, atoms, charges)
    print(f"  sign: {sign}, log_psi: {log_psi:.4f}")

    @jax.jit
    def compute_local_energy(params, electron_pos):
        def psi(pos):
            return log_psi_fn(params, pos, spins_arr, atoms, charges)

        grad_psi = jax.grad(psi)

        def laplacian_psi(pos):
            def grad_component(i):
                def psi_i(x):
                    p = pos.at[i].set(x)
                    return psi(p)

                return jax.grad(psi_i)(pos[i])

            return sum(
                jax.grad(lambda p: grad_component(i))(pos[i]) for i in range(len(pos))
            )

        kinetic = -0.5 * laplacian_psi(electron_pos)

        pos_3d = electron_pos.reshape(n_electrons, ndim)

        v_en = 0.0
        for i in range(n_electrons):
            for a in range(atoms.shape[0]):
                r = jnp.linalg.norm(pos_3d[i] - atoms[a])
                v_en -= charges[a] / jnp.maximum(r, 1e-8)

        v_ee = 0.0
        for i in range(n_electrons):
            for j in range(i + 1, n_electrons):
                r = jnp.linalg.norm(pos_3d[i] - pos_3d[j])
                v_ee += 1.0 / jnp.maximum(r, 1e-8)

        return kinetic + v_en + v_ee

    print("\nStarting training loop...")
    print("-" * 60)

    import optax

    learning_rate = 0.001
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, electrons_batch):
        def single_log_psi(pos):
            return log_psi_fn(params, pos, spins_arr, atoms, charges)

        log_psis = jax.vmap(single_log_psi)(electrons_batch)
        return -jnp.mean(log_psis)

    @jax.jit
    def train_step(params, opt_state, electrons, key):
        key, subkey = jax.random.split(key)
        proposal = electrons + jax.random.normal(subkey, electrons.shape) * 0.1

        def log_prob(pos):
            return 2 * log_psi_fn(params, pos, spins_arr, atoms, charges)

        old_log_prob = jax.vmap(log_prob)(electrons)
        new_log_prob = jax.vmap(log_prob)(proposal)

        key, subkey = jax.random.split(key)
        accept = jnp.log(jax.random.uniform(subkey, (cfg.batch_size,))) < (
            new_log_prob - old_log_prob
        )

        new_electrons = jnp.where(accept[:, None], proposal, electrons)
        pmove = jnp.mean(accept)

        loss, grads = jax.value_and_grad(loss_fn)(params, new_electrons)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, new_electrons, key, loss, pmove

    history = {"loss": [], "pmove": []}

    for step in range(cfg.optim.iterations):
        params, opt_state, electrons, key, loss, pmove = train_step(
            params, opt_state, electrons, key
        )

        history["loss"].append(float(loss))
        history["pmove"].append(float(pmove))

        if (step + 1) % cfg.log.print_every == 0:
            print(f"Step {step + 1:4d}: loss = {loss:8.4f}, pmove = {pmove:.3f}")

    print("-" * 60)
    print("\nTraining completed!")
    print(f"  Final loss: {history['loss'][-1]:.4f}")
    print(f"  Final pmove: {history['pmove'][-1]:.3f}")

    print("\nSampling final wavefunction...")
    for _ in range(100):
        params, opt_state, electrons, key, loss, pmove = train_step(
            params, opt_state, electrons, key
        )

    sample_log_psis = []
    for i in range(min(10, cfg.batch_size)):
        lp = log_psi_fn(params, electrons[i], spins_arr, atoms, charges)
        sample_log_psis.append(float(lp))

    print(f"  Sample log|ψ| values: {[f'{x:.2f}' for x in sample_log_psis[:5]]}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

    return params, history


if __name__ == "__main__":
    params, history = test_helium_training()
