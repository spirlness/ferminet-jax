# pyright: reportMissingImports=false

from typing import Any, cast

import jax
import jax.numpy as jnp

from ferminet.configs import helium
from ferminet.mcmc import make_mcmc_step
from ferminet.networks import make_fermi_net, make_log_psi_apply
from ferminet.types import FermiNetData


def test_make_fermi_net_and_mcmc_step_smoke():
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([2.0])
    nspins = (1, 1)
    spins_arr = jnp.array([0, 1])

    cfg = helium.get_config()
    cfg_any = cast(Any, cfg)
    cfg_any.network.determinants = 2
    cfg_any.network.ferminet.hidden_dims = ((16, 4),)

    init_fn, apply_fn, _ = make_fermi_net(atoms, charges, nspins, cfg)
    log_psi = make_log_psi_apply(apply_fn)
    params = init_fn(jax.random.PRNGKey(0))

    batch = 16
    key = jax.random.PRNGKey(1)
    positions = jax.random.normal(key, (batch, sum(nspins) * 3)) * 0.5
    data = FermiNetData(
        positions=positions, spins=spins_arr, atoms=atoms, charges=charges
    )

    mcmc_step = make_mcmc_step(log_psi, batch, 3, atoms)
    new_data, pmove = mcmc_step(params, data, key, 0.02)

    assert new_data.positions.shape == positions.shape
    assert jnp.isfinite(pmove)
    assert (pmove >= 0.0) & (pmove <= 1.0)
