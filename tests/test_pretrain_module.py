import jax
import jax.numpy as jnp
import optax

from ferminet import pretrain


def test_make_pretrain_step_updates_parameters():
    def dummy_apply(params, electrons, spins, atoms, charges):
        _ = (spins, atoms, charges)
        # Using a bias term to check parameter updates
        return jnp.sum(electrons, axis=-1) + params["bias"]

    def target_fn(electrons, atoms, charges):
        # Dummy target
        return jnp.zeros(electrons.shape[0])

    optimizer = optax.sgd(learning_rate=0.1)

    # Passing batch_size=2, n_electrons=1, ndim=3
    step_fn = pretrain.make_pretrain_step(
        dummy_apply, target_fn, optimizer, batch_size=2, n_electrons=1, ndim=3
    )

    params = {"bias": jnp.array(0.0)}
    opt_state = optimizer.init(params)

    key = jax.random.PRNGKey(0)
    spins = jnp.array([1])  # n_electrons=1
    atoms = jnp.zeros((1, 3))
    charges = jnp.array([2.0])

    new_params, new_state, loss, new_key = step_fn(
        params, opt_state, key, spins, atoms, charges
    )

    assert loss > 0.0
    assert new_params["bias"] != params["bias"]
    assert not jnp.array_equal(new_key, key)


def test_pretrain_runs_with_tiny_loop():
    def init_fn(key):
        _ = key
        return jnp.array(1.0)

    def apply_fn(params, electrons, spins, atoms, charges):
        _ = (spins, atoms, charges)
        return jnp.sum(electrons, axis=-1) + params

    def target_fn(electrons, atoms, charges):
        _ = (atoms, charges)
        return jnp.zeros(electrons.shape[0])

    key = jax.random.PRNGKey(0)
    atoms = jnp.zeros((1, 3))
    charges = jnp.array([2.0])
    spins = (1, 0)

    params = pretrain.pretrain(
        init_fn,
        apply_fn,
        target_fn,
        key,
        atoms,
        charges,
        spins,
        n_iterations=2,
        batch_size=2,
        learning_rate=0.1,
    )

    assert jnp.isscalar(params)
