import jax.numpy as jnp

from ferminet.utils import statistics, system


def test_statistics_helpers():
    ema = statistics.exponential_moving_average(
        jnp.array(0.0), jnp.array(1.0), decay=0.5
    )
    assert jnp.isclose(ema, 0.5)

    count, mean, m2 = statistics.welford_update(
        0, jnp.array(0.0), jnp.array(0.0), jnp.array(2.0)
    )
    count, mean, m2 = statistics.welford_update(count, mean, m2, jnp.array(4.0))
    var = statistics.welford_finalize(count, m2)
    assert jnp.isclose(var, 2.0)
    assert jnp.isclose(statistics.welford_finalize(1, jnp.array(0.0)), 0.0)

    data = jnp.arange(8.0)
    mean_block, stderr_block = statistics.block_average(data, block_size=2)
    assert jnp.isclose(mean_block, jnp.mean(data))
    assert stderr_block >= 0


def test_system_helpers():
    molecule = [("He", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.0))]
    atoms_bohr, charges = system.parse_molecule(molecule)
    assert atoms_bohr.shape == (2, 3)
    assert jnp.array_equal(charges, jnp.array([2, 1]))

    atoms_angstrom, _ = system.parse_molecule(molecule, units="angstrom")
    assert atoms_angstrom[1, 2] == system.ANGSTROM_TO_BOHR

    spins = system.get_spin_config(charges)
    assert isinstance(spins, tuple) and len(spins) == 2
    polarized = system.get_spin_config(charges, spin_polarized=True)
    assert polarized[1] == 0
