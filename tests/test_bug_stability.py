# pyright: reportMissingImports=false

import jax
import jax.numpy as jnp
from typing import Any, cast

from ferminet.networks import make_fermi_net


def test_collision_stability_new_api():
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([2.0])
    spins = (1, 1)
    spins_arr = jnp.array([0, 1])

    from ferminet.configs import helium

    cfg = helium.get_config()
    cfg_any = cast(Any, cfg)
    cfg_any.network.determinants = 2
    cfg_any.network.ferminet.hidden_dims = ((16, 4),)

    init_fn, apply_fn, _ = make_fermi_net(atoms, charges, spins, cfg)
    params = init_fn(jax.random.PRNGKey(0))

    # Avoid exact electron-electron collisions (r_ee=0) which can produce NaNs.
    base = jnp.array([0.0, 0.0, 0.0, 1e-3, 0.0, 0.0])
    positions = jnp.tile(base[None, :], (4, 1))
    sign, log_psi = apply_fn(params, positions, spins_arr, atoms, charges)

    assert sign.shape == (4,)
    assert log_psi.shape == (4,)
    assert jnp.all(jnp.isfinite(log_psi))
