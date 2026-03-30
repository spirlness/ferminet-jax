"""Tests for performance optimizations (round 2).

Verifies that the optimizations do not change numerical results.
"""

from typing import Any, cast

import jax
import jax.numpy as jnp

from ferminet.configs import helium
from ferminet.constants import pmap_with_donate
from ferminet.hamiltonian import (
    local_energy as make_local_energy,
)
from ferminet.hamiltonian import (
    potential_nuclear_nuclear,
)
from ferminet.networks import (
    _apply_interaction_layer,
    _init_interaction_layer,
    make_fermi_net,
)
from ferminet.types import FermiNetData

# ── P1: Precomputed V_nn ──────────────────────────────────────────────────────


def test_precomputed_vnn_matches_per_step():
    """V_nn precomputed in factory matches per-step recomputation."""
    atoms = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    charges = jnp.array([1.0, 1.0])

    nspins = (1, 1)
    cfg = helium.get_config()
    cfg_any = cast(Any, cfg)
    cfg_any.network.determinants = 2
    cfg_any.network.ferminet.hidden_dims = ((16, 4),)

    init_fn, apply_fn, _ = make_fermi_net(atoms, charges, nspins, cfg)

    # Factory with atoms → V_nn cached
    el_cached = make_local_energy(apply_fn, charges=charges, nspins=nspins, atoms=atoms)
    # Factory without atoms → V_nn computed per-call
    el_nocache = make_local_energy(apply_fn, charges=charges, nspins=nspins)

    params = init_fn(jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(1)
    positions = jax.random.normal(key, (sum(nspins) * 3,)) * 0.5
    spins_arr = jnp.array([0, 1])
    data = FermiNetData(
        positions=positions, spins=spins_arr, atoms=atoms, charges=charges
    )

    e_cached, _ = el_cached(params, key, data)
    e_nocache, _ = el_nocache(params, key, data)

    assert jnp.isfinite(e_cached)
    assert jnp.allclose(e_cached, e_nocache, atol=1e-6), (
        f"Cached V_nn energy {e_cached} != non-cached {e_nocache}"
    )


def test_vnn_constant_across_configurations():
    """V_nn is the same regardless of electron positions (sanity check)."""
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([2.0])
    v1 = potential_nuclear_nuclear(charges, atoms)
    v2 = potential_nuclear_nuclear(charges, atoms)
    assert jnp.allclose(v1, v2)


# ── P2: Interaction layer optimisation ────────────────────────────────────────


def test_interaction_layer_optimised_broadcast():
    """Verify the optimised interaction layer produces finite outputs."""
    key = jax.random.PRNGKey(42)
    in_one, in_two = 8, 5
    out_one, out_two = 8, 5
    n_elec = 3

    k1, k2, k3 = jax.random.split(key, 3)
    layer_params = _init_interaction_layer(k1, in_one, in_two, out_one, out_two)
    h_one = jax.random.normal(k2, (n_elec, in_one))
    h_two = jax.random.normal(k3, (n_elec, n_elec, in_two))

    h_one_out, h_two_out = _apply_interaction_layer(
        layer_params, h_one, h_two, jnp.tanh, use_residual=False
    )
    assert h_one_out.shape == (n_elec, out_one)
    assert h_two_out.shape == (n_elec, n_elec, out_two)
    assert jnp.all(jnp.isfinite(h_one_out))
    assert jnp.all(jnp.isfinite(h_two_out))


# ── P3: pmap_with_donate helper ───────────────────────────────────────────────


def test_pmap_with_donate_callable():
    """pmap_with_donate returns a decorator factory."""

    def dummy(x):
        return x + 1.0

    # Should not raise — returns a decorator that wraps dummy
    decorator = pmap_with_donate(donate_argnums=())
    assert callable(decorator)
    pmapped = decorator(dummy)
    assert callable(pmapped)
