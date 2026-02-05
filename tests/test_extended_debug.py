# pyright: reportMissingImports=false

from typing import Any, cast

import jax
import jax.numpy as jnp

from ferminet.configs import helium
from ferminet.hamiltonian import local_energy as make_local_energy
from ferminet.loss import make_loss
from ferminet.networks import make_fermi_net, make_log_psi_apply
from ferminet.types import FermiNetData


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


def test_loss_aux_outputs_have_expected_shapes():
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

    batch = 6
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

    loss_value, aux = loss_fn(params, key, data)

    assert jnp.isfinite(loss_value)
    assert aux.local_energy.shape == (batch,)
    assert aux.clipped_energy.shape == (batch,)
    assert jnp.isfinite(aux.variance)
    assert jnp.all(jnp.isfinite(aux.local_energy))
