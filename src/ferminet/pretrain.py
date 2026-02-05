from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax

from ferminet import types


def make_pretrain_step(
    network_apply: types.LogFermiNetLike,
    target_orbitals_fn: Callable,
    optimizer: optax.GradientTransformation,
) -> Callable:
    def loss_fn(params, electrons, spins, atoms, charges, target_orbitals):
        log_psi = network_apply(params, electrons, spins, atoms, charges)
        return jnp.mean((log_psi - target_orbitals) ** 2)

    @jax.jit
    def step(params, opt_state, batch):
        electrons, spins, atoms, charges, target = batch
        loss, grads = jax.value_and_grad(loss_fn)(
            params, electrons, spins, atoms, charges, target
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

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

    step_fn = make_pretrain_step(apply_fn, target_fn, optimizer)

    n_electrons = sum(spins)
    ndim = 3

    for i in range(n_iterations):
        key, subkey = jax.random.split(key)
        electrons = jax.random.normal(subkey, (batch_size, n_electrons * ndim))
        spins_arr = jnp.array([0] * spins[0] + [1] * spins[1])
        target = target_fn(electrons, atoms, charges)

        batch = (electrons, spins_arr, atoms, charges, target)
        params, opt_state, loss = step_fn(params, opt_state, batch)

        if (i + 1) % 100 == 0:
            print(f"Pretrain step {i + 1}/{n_iterations}, loss: {loss:.6f}")

    return params
