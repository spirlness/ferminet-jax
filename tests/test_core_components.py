import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from ferminet.multi_determinant import MultiDeterminantOrbitals
from ferminet.jastrow import JastrowFactor
from ferminet.physics import soft_coulomb_potential, nuclear_potential, electronic_potential

@pytest.fixture
def key():
    return random.PRNGKey(42)

@pytest.fixture
def system_config():
    return {
        'n_electrons': 2,
        'n_up': 1,
        'n_determinants': 4,
        'n_nuclei': 2,
        'nuclei_pos': jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        'nuclei_charge': jnp.array([1.0, 1.0])
    }

def test_multi_determinant_orbitals_init(system_config):
    orbitals = MultiDeterminantOrbitals(
        n_electrons=system_config['n_electrons'],
        n_up=system_config['n_up'],
        n_determinants=system_config['n_determinants'],
        n_nuclei=system_config['n_nuclei']
    )
    assert orbitals.n_determinants == system_config['n_determinants']
    assert 'det_weights' in orbitals.params
    assert len(orbitals.params) == system_config['n_determinants'] + 1

def test_multi_determinant_orbitals_forward(system_config, key):
    orbitals = MultiDeterminantOrbitals(
        n_electrons=system_config['n_electrons'],
        n_up=system_config['n_up'],
        n_determinants=system_config['n_determinants'],
        n_nuclei=system_config['n_nuclei']
    )
    batch_size = 4
    r_elec = random.normal(key, (batch_size, system_config['n_electrons'], 3))
    log_psi = orbitals.log_psi(r_elec, system_config['nuclei_pos'])

    assert log_psi.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(log_psi))

def test_jastrow_factor_forward(system_config, key):
    jastrow = JastrowFactor(n_electrons=system_config['n_electrons'])
    batch_size = 4
    r_elec = random.normal(key, (batch_size, system_config['n_electrons'], 3))
    j_val = jastrow.forward(r_elec)

    assert j_val.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(j_val))

def test_physics_potentials(system_config, key):
    r_single = random.normal(key, (system_config['n_electrons'], 3))

    v_n = nuclear_potential(r_single, system_config['nuclei_pos'], system_config['nuclei_charge'])
    v_e = electronic_potential(r_single)

    assert jnp.isfinite(v_n)
    assert jnp.isfinite(v_e)
    # V_ne should be negative for attractive potential
    assert v_n < 0
    # V_ee should be positive for repulsive potential
    assert v_e > 0

def test_collision_stability(system_config):
    orbitals = MultiDeterminantOrbitals(
        n_electrons=system_config['n_electrons'],
        n_up=system_config['n_up'],
        n_determinants=system_config['n_determinants'],
        n_nuclei=system_config['n_nuclei']
    )
    # Collision at origin
    r_collision = jnp.zeros((2, system_config['n_electrons'], 3))
    log_psi = orbitals.log_psi(r_collision, system_config['nuclei_pos'])

    assert jnp.all(jnp.isfinite(log_psi))
