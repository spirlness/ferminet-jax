import jax.numpy as jnp
import pytest
from ferminet import hamiltonian

def test_potential_electron_electron():
    # Two electrons at (0, 0, 0) and (1, 0, 0)
    # r_ee = [[0, 1], [1, 0]] (with extra dim for features)
    r_ee = jnp.array([[[0.0], [1.0]], [[1.0], [0.0]]])
    # V_ee = 1/r_12 = 1.0
    v_ee = hamiltonian.potential_electron_electron(r_ee)
    assert jnp.allclose(v_ee, 1.0)

def test_potential_electron_nuclear():
    # One atom at (0, 0, 0) with charge 2.0
    # One electron at (1, 0, 0)
    # r_ae = [[[1.0]]] (shape (1, 1, 1))
    charges = jnp.array([2.0])
    r_ae = jnp.array([[[1.0]]])
    # V_en = -Z_1 / r_11 = -2.0 / 1.0 = -2.0
    v_en = hamiltonian.potential_electron_nuclear(charges, r_ae)
    assert jnp.allclose(v_en, -2.0)

def test_potential_nuclear_nuclear():
    # Two atoms at (0, 0, 0) and (1, 0, 0) with charges 1.0 and 2.0
    atoms = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    charges = jnp.array([1.0, 2.0])
    # V_nn = (Z_1 * Z_2) / r_12 = (1.0 * 2.0) / 1.0 = 2.0
    v_nn = hamiltonian.potential_nuclear_nuclear(charges, atoms)
    assert jnp.allclose(v_nn, 2.0)

def test_potential_energy():
    # System: H2 molecule (roughly)
    # Atoms at (0, 0, 0) and (1, 0, 0) with charges 1.0
    # Electrons at (0.5, 0, 0) and (0.5, 1, 0)
    atoms = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    charges = jnp.array([1.0, 1.0])
    pos = jnp.array([[0.5, 0.0, 0.0], [0.5, 1.0, 0.0]])

    # Manual calculation:
    # V_nn: Z1*Z2 / R_12 = 1*1 / 1.0 = 1.0
    # V_ee: 1 / r_12 = 1 / sqrt((0.5-0.5)^2 + (0-1)^2 + (0-0)^2) = 1 / 1.0 = 1.0
    # V_en:
    # e1 to a1: r=0.5 -> -1/0.5 = -2.0
    # e1 to a2: r=0.5 -> -1/0.5 = -2.0
    # e2 to a1: r=sqrt(0.5^2 + 1^2) = sqrt(1.25) -> -1/sqrt(1.25)
    # e2 to a2: r=sqrt(0.5^2 + 1^2) = sqrt(1.25) -> -1/sqrt(1.25)
    # V_en = -4.0 - 2/sqrt(1.25)

    expected_v_nn = 1.0
    expected_v_ee = 1.0
    expected_v_en = -4.0 - 2.0 / jnp.sqrt(1.25)
    expected_total = expected_v_nn + expected_v_ee + expected_v_en

    _, _, r_ae, r_ee = hamiltonian.construct_input_features(pos, atoms)
    total_v = hamiltonian.potential_energy(r_ae, r_ee, atoms, charges)

    assert jnp.allclose(total_v, expected_total)

def test_construct_input_features():
    # pos shape (n_e, 3), atoms shape (n_a, 3)
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    pos = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    ae, ee, r_ae, r_ee = hamiltonian.construct_input_features(pos, atoms)

    # ae shape (n_e, n_a, 3)
    assert ae.shape == (2, 1, 3)
    assert jnp.allclose(ae[0, 0], jnp.array([1.0, 0.0, 0.0]))
    assert jnp.allclose(ae[1, 0], jnp.array([0.0, 1.0, 0.0]))

    # ee shape (n_e, n_e, 3)
    # ee[i, j] = pos[j] - pos[i]
    assert ee.shape == (2, 2, 3)
    assert jnp.allclose(ee[0, 1], jnp.array([-1.0, 1.0, 0.0]))
    assert jnp.allclose(ee[1, 0], jnp.array([1.0, -1.0, 0.0]))

    # r_ae shape (n_e, n_a, 1)
    assert r_ae.shape == (2, 1, 1)
    assert jnp.allclose(r_ae[:, :, 0], jnp.array([[1.0], [1.0]]))

    # r_ee shape (n_e, n_e, 1)
    assert r_ee.shape == (2, 2, 1)
    assert jnp.allclose(r_ee[0, 1, 0], jnp.sqrt(2.0))
    assert jnp.allclose(r_ee[0, 0, 0], 0.0) # Masked diagonal
