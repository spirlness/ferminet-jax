from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax

from ferminet import types


def make_pretrain_step(
    network_apply: types.LogFermiNetLike,
    target_orbitals_fn: Callable,
    optimizer: optax.GradientTransformation,
    batch_size: int,
    n_electrons: int,
    ndim: int = 3,
) -> Callable:
    def loss_fn(params, electrons, spins, atoms, charges, target_orbitals):
        log_psi = network_apply(params, electrons, spins, atoms, charges)
        return jnp.mean((log_psi - target_orbitals) ** 2)

    @jax.jit
    def step(params, opt_state, key, spins, atoms, charges):
        key, subkey = jax.random.split(key)
        electrons = jax.random.normal(subkey, (batch_size, n_electrons * ndim))
        target = target_orbitals_fn(electrons, atoms, charges)

        loss, grads = jax.value_and_grad(loss_fn)(
            params, electrons, spins, atoms, charges, target
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, key

    return step


def pretrain(
    init_fn: types.InitFermiNet,
    apply_fn: types.LogFermiNetLike,
    target_fn: Callable,
    key: jax.Array,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    spins: Tuple[int, int],
    n_iterations: int = 1000,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
) -> types.ParamTree:
    key, subkey = jax.random.split(key)
    params = init_fn(subkey)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    n_electrons = sum(spins)
    ndim = 3

    step_fn = make_pretrain_step(
        apply_fn, target_fn, optimizer, batch_size, n_electrons, ndim
    )

    spins_arr = jnp.array([0] * spins[0] + [1] * spins[1])

    for i in range(n_iterations):
        params, opt_state, loss, key = step_fn(
            params, opt_state, key, spins_arr, atoms, charges
        )

        if (i + 1) % 100 == 0:
            print(f"Pretrain step {i + 1}/{n_iterations}, loss: {loss:.6f}")

    return params
