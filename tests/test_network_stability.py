# pyright: reportMissingImports=false

from typing import Any, cast

import jax
import jax.numpy as jnp

from ferminet.configs import helium
from ferminet.networks import make_fermi_net


def test_network_outputs_finite_for_extreme_inputs():
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

    key = jax.random.PRNGKey(1)
    positions = jax.random.normal(key, (8, sum(nspins) * 3)) * 0.5
    positions_large = positions * 100.0
    base = jnp.array([0.0, 0.0, 0.0, 1e-3, 0.0, 0.0])
    positions_collision = jnp.tile(base[None, :], (positions.shape[0], 1))

    for x in (positions, positions_large, positions_collision):
        _, log_psi = apply_fn(params, x, spins_arr, atoms, charges)
        assert jnp.all(jnp.isfinite(log_psi))
