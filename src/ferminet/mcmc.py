"""Metropolis-Hastings Monte Carlo sampling.

NOTE: these functions operate on batches of MCMC configurations.
"""

from typing import Callable, TypeAlias, TypeVar, cast

import jax
import jax.numpy as jnp
from jax import lax

from ferminet.types import FermiNetData, LogFermiNetLike, ParamTree
from ferminet.utils.numerics import EPS

MCMCState: TypeAlias = tuple[
    FermiNetData, jax.Array, jnp.ndarray, jnp.ndarray, jnp.ndarray | None
]
T = TypeVar("T")


def _fori_loop(
    lower: int,
    upper: int,
    body_fun: Callable[[int, T], T],
    init_val: T,
) -> T:
    fori_loop = cast(
        Callable[[int, int, Callable[[int, T], T], T], T],
        lax.fori_loop,
    )
    return fori_loop(lower, upper, body_fun, init_val)


def _split_key(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    keys = jax.random.split(key)
    return keys[0], keys[1]


def _asarray_data(
    data: FermiNetData,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return data.positions, data.spins, data.atoms, data.charges


def _harmonic_mean(x: jnp.ndarray, atoms: jnp.ndarray) -> jnp.ndarray:
    """Harmonic mean of electron-atom distances.

    Args:
        x: Electron positions (batch, nelec, 1, ndim).
        atoms: Atom positions (natoms, ndim).

    Returns:
        Harmonic mean distances (batch, nelec, 1, 1).
    """
    ae: jnp.ndarray = x - atoms[None, ...]
    r_ae = cast(
        jnp.ndarray,
        jnp.sqrt(jnp.sum(jnp.square(ae), axis=-1, keepdims=True) + EPS),
    )
    return 1.0 / jnp.mean(1.0 / (r_ae + EPS), axis=-2, keepdims=True)


def mh_accept(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    lp_1: jnp.ndarray,
    lp_2: jnp.ndarray,
    ratio: jnp.ndarray,
    subkey: jax.Array,
    num_accepts: jnp.ndarray,
    hmean1: jnp.ndarray | None = None,
    hmean2: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Metropolis-Hastings accept/reject step with non-finite guards."""
    rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
    finite_proposal = jnp.isfinite(lp_2) & jnp.isfinite(ratio)
    cond = (ratio > rnd) & finite_proposal
    x_new = jnp.where(cond[..., None], x2, x1)
    lp_new = jnp.where(cond, lp_2, lp_1)
    num_accepts += jnp.sum(cond)

    if hmean1 is not None and hmean2 is not None:
        hmean_new = jnp.where(cond[..., None, None, None], hmean2, hmean1)
    else:
        hmean_new = None

    return x_new, lp_new, num_accepts, hmean_new


def mh_update(
    params: ParamTree,
    f: LogFermiNetLike,
    data: FermiNetData,
    key: jax.Array,
    lp_1: jnp.ndarray,
    num_accepts: jnp.ndarray,
    hmean_1: jnp.ndarray | None,
    stddev: float | jnp.ndarray = 0.02,
    atoms: jnp.ndarray | None = None,
    ndim: int = 3,
) -> tuple[FermiNetData, jax.Array, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Performs one MH step using all-electron move.

    Args:
        params: Network parameters.
        f: Log wavefunction network.
        data: MCMC configuration.
        key: RNG key.
        lp_1: Current log probability (2 * log|psi|).
        num_accepts: Running total of accepts.
        hmean_1: Current harmonic mean distances (or None).
        stddev: Proposal standard deviation. Can be a scalar (same for all
            electrons) or a per-electron array of shape ``(n_electrons,)``
            for shell-adaptive proposals.
        atoms: Atom positions for adaptive proposal.
        ndim: Dimensionality.

    Returns:
        (new_data, key, lp_new, num_accepts, hmean_new)
    """
    key, subkey, subkey_accept = jax.random.split(key, num=3)
    positions, spins, atoms_data, charges = _asarray_data(data)
    x1: jnp.ndarray = positions

    if atoms is None:
        # Symmetric proposal — stddev can be scalar or per-electron array.
        noise = jax.random.normal(subkey, shape=x1.shape)
        # Broadcast: if stddev is per-electron (shape (nelec,)), reshape to
        # (1, nelec*ndim) or (nelec, ndim) depending on x1 layout so that
        # multiplication broadcasts correctly.
        stddev_arr = jnp.asarray(stddev)
        if stddev_arr.ndim >= 1 and x1.ndim == 2:
            # stddev is per-electron, x1 is (batch, nelec*ndim)
            stddev_broad = jnp.repeat(stddev_arr, ndim)[None, :]
        else:
            stddev_broad = stddev_arr
        x2 = x1 + stddev_broad * noise
        lp_2 = 2.0 * f(params, x2, spins, atoms_data, charges)
        ratio = lp_2 - lp_1
        hmean_2 = None
        x2_flat = x2
    else:
        # Asymmetric proposal scaled by harmonic mean to atoms
        n = x1.shape[0]
        x1 = jnp.reshape(x1, [n, -1, 1, ndim])
        # Use passed hmean_1 if available, otherwise compute it
        if hmean_1 is None:
            hmean_1 = _harmonic_mean(x1, atoms)

        noise = jax.random.normal(subkey, shape=x1.shape)
        x2 = x1 + stddev * hmean_1 * noise
        x2_flat = jnp.reshape(x2, [n, -1])
        lp_2 = 2.0 * f(params, x2_flat, spins, atoms_data, charges)
        hmean_2 = _harmonic_mean(x2, atoms)

        # Forward and reverse probabilities for detailed balance
        lq_1 = _log_prob_gaussian(x1, x2, stddev * hmean_1)
        lq_2 = _log_prob_gaussian(x2, x1, stddev * hmean_2)
        ratio = lp_2 + lq_2 - lp_1 - lq_1

        x1 = jnp.reshape(x1, [n, -1])

    x_new, lp_new, num_accepts, hmean_new = mh_accept(
        x1,
        x2_flat,
        lp_1,
        lp_2,
        ratio,
        subkey_accept,
        num_accepts,
        hmean_1,
        hmean_2,
    )

    # Update data with new positions
    new_data = data._replace(positions=x_new)
    return new_data, key, lp_new, num_accepts, hmean_new


def _log_prob_gaussian(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
    """Log probability of Gaussian with diagonal covariance."""
    numer = jnp.sum(-0.5 * ((x - mu) ** 2) / (sigma**2), axis=[1, 2, 3])
    denom = x.shape[-1] * jnp.sum(jnp.log(sigma), axis=[1, 2, 3])
    return numer - denom


def make_mcmc_step(
    batch_network: LogFermiNetLike,
    batch_per_device: int,
    steps: int = 10,
    atoms: jnp.ndarray | None = None,
    ndim: int = 3,
) -> Callable[
    [ParamTree, FermiNetData, jax.Array, float], tuple[FermiNetData, jnp.ndarray]
]:
    """Creates the MCMC step function.

    Args:
        batch_network: Batched log|psi| network.
        batch_per_device: Batch size per device.
        steps: Number of MCMC steps per call.
        atoms: Atom positions for adaptive proposal.
        ndim: Dimensionality.

    Returns:
        MCMC step function.
    """

    def mcmc_step(
        params: ParamTree,
        data: FermiNetData,
        key: jax.Array,
        width: float,
    ) -> tuple[FermiNetData, jnp.ndarray]:
        def step_fn(_: int, x: MCMCState) -> MCMCState:
            return mh_update(
                params, batch_network, *x, stddev=width, atoms=atoms, ndim=ndim
            )

        positions, spins, atoms_data, charges = _asarray_data(data)
        logprob = 2.0 * batch_network(params, positions, spins, atoms_data, charges)

        if atoms is not None:
            n = positions.shape[0]
            x_reshaped = jnp.reshape(positions, [n, -1, 1, ndim])
            hmean_init = _harmonic_mean(x_reshaped, atoms)
        else:
            hmean_init = None

        num_accepts_init = jnp.array(0.0)
        new_data, key, _, num_accepts, _ = _fori_loop(
            0, steps, step_fn, (data, key, logprob, num_accepts_init, hmean_init)
        )

        pmove = num_accepts / (steps * batch_per_device)
        return new_data, pmove

    return mcmc_step


def update_mcmc_width(
    t: int,
    width: float,
    adapt_frequency: int,
    pmove: float | jnp.ndarray,
    pmoves: jnp.ndarray,
    pmove_max: float = 0.55,
    pmove_min: float = 0.5,
    width_min: float = 0.001,
    width_max: float = 10.0,
) -> tuple[float, jnp.ndarray]:
    """Adapts MCMC step width based on acceptance rate."""
    if adapt_frequency <= 0 or pmoves.size == 0:
        return width, pmoves

    idx = t % adapt_frequency
    idx = int(jnp.clip(jnp.asarray(idx), 0, pmoves.size - 1))
    pmoves = pmoves.at[idx].set(jnp.asarray(pmove, dtype=pmoves.dtype))

    if t > 0 and idx == 0:
        mean_pmove = float(jnp.mean(pmoves))
        if mean_pmove > pmove_max:
            width *= 1.1
        elif mean_pmove < pmove_min:
            width /= 1.1

    width = float(jnp.clip(jnp.asarray(width), width_min, width_max))
    return width, pmoves
