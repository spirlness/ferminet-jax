"""Tests for the forward-over-reverse Laplacian mode in hamiltonian.py."""

import jax
import jax.numpy as jnp

from ferminet import hamiltonian, types


def _make_dummy_network():
    """Return a simple FermiNet-like function for testing kinetic energy."""

    def apply(params, pos, spins, atoms, charges):
        _ = (spins, atoms, charges)
        # log|psi| = -0.5 * sum(pos^2)  →  T = -0.5 * (n + sum(pos^2))
        return jnp.ones(()), -0.5 * jnp.sum(pos**2) * params["scale"]

    return apply


def test_forward_over_reverse_matches_default():
    """Verify that forward_over_reverse produces the same kinetic energy as default."""
    net = _make_dummy_network()
    params = {"scale": jnp.array(1.0)}
    pos = jnp.array([0.5, -0.3, 0.8, 1.0, -0.5, 0.2])
    data = types.FermiNetData(
        positions=pos,
        spins=jnp.array([0, 1]),
        atoms=jnp.array([[0.0, 0.0, 0.0]]),
        charges=jnp.array([2.0]),
    )

    ke_default = hamiltonian.local_kinetic_energy(net, laplacian_mode="default")
    ke_fwd_rev = hamiltonian.local_kinetic_energy(
        net, laplacian_mode="forward_over_reverse"
    )

    t_default = ke_default(params, data)
    t_fwd_rev = ke_fwd_rev(params, data)
    assert jnp.allclose(t_default, t_fwd_rev, atol=1e-5), (
        f"Mismatch: default={t_default}, fwd_rev={t_fwd_rev}"
    )


def test_scan_mode_matches_default():
    """Verify that scan mode produces the same kinetic energy as default."""
    net = _make_dummy_network()
    params = {"scale": jnp.array(1.0)}
    pos = jnp.array([0.5, -0.3, 0.8, 1.0, -0.5, 0.2])
    data = types.FermiNetData(
        positions=pos,
        spins=jnp.array([0, 1]),
        atoms=jnp.array([[0.0, 0.0, 0.0]]),
        charges=jnp.array([2.0]),
    )

    ke_default = hamiltonian.local_kinetic_energy(net, laplacian_mode="default")
    ke_scan = hamiltonian.local_kinetic_energy(net, laplacian_mode="scan")

    t_default = ke_default(params, data)
    t_scan = ke_scan(params, data)
    assert jnp.allclose(t_default, t_scan, atol=1e-5)


def test_legacy_use_scan_flag():
    """Verify the legacy use_scan=True flag still works via mode normalisation."""
    net = _make_dummy_network()
    params = {"scale": jnp.array(1.0)}
    pos = jnp.array([0.5, -0.3, 0.8])
    data = types.FermiNetData(
        positions=pos,
        spins=jnp.array([0]),
        atoms=jnp.array([[0.0, 0.0, 0.0]]),
        charges=jnp.array([1.0]),
    )

    ke_legacy = hamiltonian.local_kinetic_energy(net, use_scan=True)
    ke_default = hamiltonian.local_kinetic_energy(net)

    t_legacy = ke_legacy(params, data)
    t_default = ke_default(params, data)
    assert jnp.allclose(t_legacy, t_default, atol=1e-5)
