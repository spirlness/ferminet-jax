# pyright: reportMissingImports=false

from typing import Any, cast

import jax
import jax.numpy as jnp

from ferminet.configs import helium
from ferminet.hamiltonian import local_energy as make_local_energy
from ferminet.loss import make_loss
from ferminet.networks import make_fermi_net, make_log_psi_apply
from ferminet.types import FermiNetData, ParamTree


def _tree_l2_norm(tree: ParamTree) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(
        jnp.sum(jnp.array([jnp.sum(jnp.square(jnp.asarray(x))) for x in leaves]))
    )


def _make_batched_local_energy(
    apply_sign_log, charges: jnp.ndarray, nspins: tuple[int, int]
):
    single = make_local_energy(apply_sign_log, charges=charges, nspins=nspins)

    def batched(params, key, data: FermiNetData) -> jnp.ndarray:
        def per_config(pos: jnp.ndarray) -> jnp.ndarray:
            sample = FermiNetData(
                positions=pos,
                spins=data.spins,
                atoms=data.atoms,
                charges=data.charges,
            )
            e, _ = single(params, key, sample)
            return e

        return jax.vmap(per_config)(data.positions)

    return batched


def test_loss_depends_on_params_and_has_gradients():
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([2.0])
    nspins = (1, 1)
    spins_arr = jnp.array([0, 1])

    cfg = helium.get_config()
    cfg_any = cast(Any, cfg)
    cfg_any.network.determinants = 2
    cfg_any.network.ferminet.hidden_dims = ((16, 4),)

    init_fn, apply_fn, _ = make_fermi_net(atoms, charges, nspins, cfg)
    params = init_fn(jax.random.PRNGKey(0))

    batch = 8
    key = jax.random.PRNGKey(1)
    positions = jax.random.normal(key, (batch, sum(nspins) * 3)) * 0.5
    data = FermiNetData(
        positions=positions, spins=spins_arr, atoms=atoms, charges=charges
    )

    log_psi = make_log_psi_apply(apply_fn)
    local_energy_fn = _make_batched_local_energy(
        apply_fn, charges=charges, nspins=nspins
    )
    loss_fn = make_loss(log_psi, local_energy_fn)

    (loss_value, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, key, data
    )
    grad_norm = _tree_l2_norm(grads)

    params_scaled = jax.tree_util.tree_map(lambda x: x * 1.1, params)
    loss_value_scaled, _ = loss_fn(params_scaled, key, data)

    assert jnp.isfinite(loss_value)
    assert jnp.isfinite(loss_value_scaled)
    assert jnp.abs(loss_value - loss_value_scaled) > 1e-6
    assert grad_norm > 0.0
