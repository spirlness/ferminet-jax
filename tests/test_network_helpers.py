import jax
import jax.numpy as jnp
import ml_collections
import pytest

from ferminet import networks


def test_normalize_atoms_and_electrons_handles_various_shapes():
    atoms = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    normalized_atoms = networks._normalize_atoms(atoms, ndim=3)
    assert normalized_atoms.shape == (2, 3)

    electrons = jnp.arange(6.0)
    normalized_electrons = networks._normalize_electrons(
        electrons, n_electrons=2, ndim=3
    )
    assert normalized_electrons.shape == (2, 3)

    batched = networks._normalize_electrons(jnp.arange(12.0).reshape(2, 6), 2, 3)
    assert batched.shape == (2, 2, 3)

    with pytest.raises((ValueError, TypeError)):
        networks._normalize_atoms(jnp.arange(5.0), ndim=3)


def test_activation_lookup_and_pairwise_helpers():
    tanh = networks._activation_from_name("tanh")
    relu = networks._activation_from_name("relu")
    assert jnp.allclose(tanh(jnp.array(0.0)), 0.0)
    assert jnp.allclose(relu(jnp.array(-1.0)), 0.0)

    electrons = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    r_ae = networks._pairwise_electron_nuclear_vectors(electrons, atoms)
    assert r_ae.shape == (2, 1, 3)

    r_ee = networks._pairwise_electron_electron_vectors(electrons)
    assert r_ee.shape == (2, 2, 3)


def test_masked_mean_and_determinant_combination():
    values = jnp.arange(12.0).reshape(2, 2, 3)
    mask = networks._electron_electron_mask(2)
    masked_mean = networks._masked_mean(values, mask)
    assert masked_mean.shape == (2, 3)

    orbitals_up = jnp.eye(2)[None, ...]
    orbitals_down = jnp.eye(2)[None, ...]
    sign, log_abs = networks._combine_determinants(orbitals_up, orbitals_down, 2, 2)
    assert sign.shape == ()
    assert log_abs.shape == ()


def test_envelope_type_and_config_helpers():
    cfg = ml_collections.ConfigDict()
    cfg.network = ml_collections.ConfigDict()
    cfg.network.ferminet = ml_collections.ConfigDict()
    cfg.network.ferminet.hidden_dims = ((64, 16), (32, 8))
    cfg.network.ferminet.determinants = 4
    cfg.network.ferminet.hidden_activation = "gelu"

    hidden_dims = networks._hidden_dims_from_cfg(cfg)
    assert hidden_dims == ((64, 16), (32, 8))

    dets = networks._determinants_from_cfg(cfg)
    assert dets == 4

    activation = networks._activation_from_cfg(cfg)
    assert jnp.allclose(activation(jnp.array(1.0)), jax.nn.gelu(jnp.array(1.0)))

    cfg.network.envelope_type = "diagonal"
    assert networks._envelope_type_from_cfg(cfg) == "diagonal"


@pytest.mark.parametrize("envelope_type", ["isotropic", "diagonal", "full"])
def test_make_fermi_net_supports_envelope_types(envelope_type):
    cfg = ml_collections.ConfigDict()
    cfg.system = ml_collections.ConfigDict()
    cfg.system.ndim = 3
    cfg.network = ml_collections.ConfigDict()
    cfg.network.determinants = 2
    cfg.network.envelope_type = envelope_type
    cfg.network.envelope = ml_collections.ConfigDict()
    cfg.network.envelope.isotropic = ml_collections.ConfigDict()
    cfg.network.envelope.isotropic.sigma = 1.0
    cfg.network.ferminet = ml_collections.ConfigDict()
    cfg.network.ferminet.hidden_dims = ((16, 4),)
    cfg.network.ferminet.determinants = 2
    cfg.network.ferminet.hidden_activation = "tanh"
    cfg.network.ferminet.bias_orbitals = True
    cfg.network.ferminet.envelope_type = envelope_type

    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([2.0])
    spins = (1, 1)
    init_fn, apply_fn, _ = networks.make_fermi_net(atoms, charges, spins, cfg)
    params = init_fn(jax.random.PRNGKey(0))
    electrons = jnp.zeros((2, sum(spins) * 3))
    spins_arr = jnp.array([0, 1])

    sign, log_psi = apply_fn(params, electrons, spins_arr, atoms, charges)
    assert sign.shape == (2,)
    assert log_psi.shape == (2,)
    assert jnp.all(jnp.isfinite(log_psi))
