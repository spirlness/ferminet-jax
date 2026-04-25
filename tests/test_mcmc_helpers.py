import jax
import jax.numpy as jnp

from ferminet import mcmc, types


def test_harmonic_mean_matches_manual_computation():
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    x = jnp.ones((1, 2, 1, 3))
    result = mcmc._harmonic_mean(x, atoms)
    r = jnp.linalg.norm(x - atoms, axis=-1, keepdims=True)
    expected = 1.0 / jnp.mean(1.0 / r, axis=-2, keepdims=True)
    assert jnp.allclose(result, expected)


def test_harmonic_mean_is_finite_at_zero_distance():
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    x = jnp.zeros((1, 2, 1, 3))
    result = mcmc._harmonic_mean(x, atoms)
    assert jnp.all(jnp.isfinite(result))


def test_mh_accept_rejects_non_finite_proposals():
    x1 = jnp.zeros((2, 2))
    x2 = jnp.ones((2, 2))
    lp1 = jnp.zeros((2,))
    lp2 = jnp.array([jnp.inf, 0.0])
    ratio = jnp.array([1.0, 1.0])
    key = jax.random.PRNGKey(0)
    num_accepts = jnp.array(0.0)

    current = mcmc.WalkerState(positions=x1, log_prob=lp1, hmean=None)
    proposal = mcmc.WalkerState(positions=x2, log_prob=lp2, hmean=None)

    new_walker, accepts = mcmc.mh_accept(current, proposal, ratio, key, num_accepts)

    assert jnp.array_equal(
        new_walker.positions[0], x1[0]
    )  # non-finite proposal rejected
    assert jnp.array_equal(new_walker.positions[1], x2[1])
    assert accepts > 0
    assert jnp.isfinite(new_walker.log_prob[1])
    assert new_walker.hmean is None


def test_log_prob_gaussian_and_width_update_behaviour():
    x = jnp.ones((1, 1, 1, 3))
    mu = jnp.zeros_like(x)
    sigma = jnp.ones_like(x)
    log_prob = mcmc._log_prob_gaussian(x, mu, sigma)
    assert log_prob.shape == (1,)

    width, pmoves = mcmc.update_mcmc_width(
        t=0,
        width=0.1,
        adapt_frequency=20,
        pmove=0.9,
        pmoves=jnp.zeros((20,)),
        pmove_max=0.6,
        pmove_min=0.5,
    )
    assert jnp.isclose(jnp.asarray(width), jnp.asarray(0.1))
    assert pmoves[0] == jnp.asarray(0.9)

    width, _ = mcmc.update_mcmc_width(
        t=20,
        width=0.1,
        adapt_frequency=20,
        pmove=0.9,
        pmoves=jnp.full((20,), 0.9),
        pmove_max=0.6,
        pmove_min=0.5,
    )
    assert width > 0.1

    width_cool, _ = mcmc.update_mcmc_width(
        t=20,
        width=10.0,
        adapt_frequency=20,
        pmove=0.1,
        pmoves=jnp.full((20,), 0.1),
    )
    assert width_cool < 10.0


def test_mh_update_without_atoms_modifies_positions():
    def dummy_network(params, positions, spins, atoms, charges):
        _ = (params, spins, atoms, charges)
        return jnp.sum(positions, axis=-1)

    positions = jnp.zeros((2, 4))
    spins = jnp.array([0, 1])
    atoms = jnp.zeros((1, 2))
    charges = jnp.array([1.0])
    data = types.FermiNetData(
        positions=positions, spins=spins, atoms=atoms, charges=charges
    )

    lp1 = jnp.zeros((2,))
    num_accepts = jnp.array(0.0)
    state = mcmc.MCMCState(
        data=data,
        key=jax.random.PRNGKey(0),
        log_prob=lp1,
        num_accepts=num_accepts,
        hmean=None,
    )
    new_state = mcmc.mh_update(
        params={},
        f=dummy_network,
        state=state,
        stddev=0.1,
        atoms=None,
        ndim=2,
    )

    assert new_state.data.positions.shape == positions.shape
    assert new_state.num_accepts >= 0.0
    assert new_state.hmean is None


def test_mh_update_with_atoms_updates_hmean():
    def dummy_network(params, positions, spins, atoms, charges):
        return jnp.zeros((positions.shape[0],))

    ndim = 3
    nelec = 2
    batch = 2
    positions = jnp.ones((batch, nelec * ndim))
    spins = jnp.array([0, 0])
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([1.0])
    data = types.FermiNetData(
        positions=positions, spins=spins, atoms=atoms, charges=charges
    )

    lp1 = jnp.zeros((batch,))
    num_accepts = jnp.array(0.0)

    # Calculate initial hmean manually
    x_reshaped = jnp.reshape(positions, [batch, nelec, 1, ndim])
    hmean_init = mcmc._harmonic_mean(x_reshaped, atoms)

    state = mcmc.MCMCState(
        data=data,
        key=jax.random.PRNGKey(0),
        log_prob=lp1,
        num_accepts=num_accepts,
        hmean=hmean_init,
    )

    new_state = mcmc.mh_update(
        params={},
        f=dummy_network,
        state=state,
        stddev=0.1,
        atoms=atoms,
        ndim=ndim,
    )

    assert new_state.data.positions.shape == positions.shape
    assert new_state.hmean is not None
    assert new_state.hmean.shape == hmean_init.shape
