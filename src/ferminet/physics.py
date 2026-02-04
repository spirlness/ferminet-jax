"""
FermiNet Stage 1 - Optimized Physics Layer
Using JAX for wave function related physical calculations
"""

import jax
import jax.numpy as jnp
from functools import lru_cache
from typing import Callable


@jax.jit
def soft_coulomb_potential(r: jnp.ndarray, alpha: float = 0.1) -> jnp.ndarray:
    """
    Soft-core Coulomb potential to avoid singularity at r=0

    Formula: V = 1 / sqrt(r^2 + alpha^2)
    """
    return 1.0 / jnp.sqrt(r**2 + alpha**2)


@jax.jit
def nuclear_potential(
    r_elec: jnp.ndarray, nuclei_pos: jnp.ndarray, nuclei_charge: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate nuclear-electron attraction potential V_ne
    V_ne = -sum_{i,j} Z_j / |r_i - R_j|
    """
    # r_elec: (n_elec, 3)
    # nuclei_pos: (n_nuclei, 3)

    # Calculate distances from all electrons to all nuclei
    # Broadcasting: (n_elec, 1, 3) - (1, n_nuclei, 3) -> (n_elec, n_nuclei, 3)
    diff = r_elec[:, None, :] - nuclei_pos[None, :, :]
    distances = jnp.linalg.norm(diff, axis=-1)  # (n_elec, n_nuclei)

    # Soften to avoid singularity
    soft_distances = soft_coulomb_potential(distances, alpha=0.1)

    # Calculate potential: -sum(Z_j / r_ij)
    # Sum over nuclei charges for each electron, then sum over electrons
    potential = -jnp.sum(nuclei_charge[None, :] / soft_distances)

    return potential


@jax.jit
def electronic_potential(r_elec: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate electron-electron repulsion potential V_ee
    V_ee = sum_{i<j} 1 / |r_i - r_j|
    """
    n_elec = r_elec.shape[0]

    # Calculate distances between all electron pairs
    # (n_elec, 1, 3) - (1, n_elec, 3) -> (n_elec, n_elec, 3)
    diff = r_elec[:, None, :] - r_elec[None, :, :]
    distances = jnp.linalg.norm(diff, axis=-1)  # (n_elec, n_elec)

    # Create upper triangular mask (exclude diagonal and lower triangle)
    mask = jnp.triu(jnp.ones((n_elec, n_elec)), k=1)

    # Soften to avoid singularity (add large number to zeros to avoid div by zero in non-masked areas, though mask handles sum)
    # Actually simpler: just compute 1/soft_dist and mask the result.
    soft_distances = soft_coulomb_potential(distances, alpha=0.1)

    # Calculate repulsion potential
    potential = jnp.sum((1.0 / soft_distances) * mask)

    return potential


@jax.jit
def total_potential(
    r_elec: jnp.ndarray, nuclei_pos: jnp.ndarray, nuclei_charge: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate total potential energy V = V_ne + V_ee
    """
    v_ne = nuclear_potential(r_elec, nuclei_pos, nuclei_charge)
    v_ee = electronic_potential(r_elec)
    return v_ne + v_ee


def kinetic_energy(log_psi: Callable, r_r: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate kinetic energy T = -0.5 * sum_i (d^2 log_psi / dr_i^2)
    Using gradient formula: T = -0.5 * (|grad log psi|^2 + Laplacian log psi)

    Note: This function is NOT jitted by default because it takes a function argument.
    It should be jitted when part of a larger computation (e.g. inside train_step).
    """
    n_elec = int(r_r.shape[0])
    kinetic_fn = _get_kinetic_energy_fn(log_psi, n_elec)
    return kinetic_fn(r_r)


def make_kinetic_energy(
    log_psi: Callable, n_electrons: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Factory for kinetic energy computation.

    Builds and returns a function `kinetic(r_elec)` which computes the kinetic
    energy for a single electron configuration `r_elec` with shape [n_elec, 3].

    The returned function hoists JAX transforms (grad/jacfwd) outside the hot
    path so callers can reuse it across many evaluations.
    """

    def log_psi_flat(r_flat: jnp.ndarray) -> jnp.ndarray:
        return log_psi(r_flat.reshape(n_electrons, 3))

    grad_log_psi_flat = jax.grad(log_psi_flat)

    def kinetic(r_elec: jnp.ndarray) -> jnp.ndarray:
        r_flat = r_elec.reshape(-1)
        grad, grad_jvp_fn = jax.linearize(grad_log_psi_flat, r_flat)
        grad_squared_sum = jnp.sum(grad**2)

        n = r_flat.shape[0]

        def body_fun(i, val):
            e_i = jax.nn.one_hot(i, n)
            _, dgrad = jax.jvp(grad_log_psi_flat, (r_flat,), (e_i,))
            return val + dgrad[i]

        laplacian = jax.lax.fori_loop(0, n, body_fun, 0.0)
        return -0.5 * (grad_squared_sum + laplacian)

    return kinetic


@lru_cache(maxsize=128)
def _cached_kinetic_energy_fn(
    log_psi: Callable, n_electrons: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return make_kinetic_energy(log_psi, n_electrons)


def _get_kinetic_energy_fn(
    log_psi: Callable, n_electrons: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get or build a kinetic energy function.

    Falls back to uncached construction when `log_psi` is not cacheable.
    """
    try:
        return _cached_kinetic_energy_fn(log_psi, n_electrons)
    except TypeError:
        return make_kinetic_energy(log_psi, n_electrons)


def local_energy(
    log_psi: Callable,
    r_r: jnp.ndarray,
    nuclei_pos: jnp.ndarray,
    nuclei_charge: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate local energy E_L = T + V
    """
    # Calculate kinetic energy
    t = kinetic_energy(log_psi, r_r)

    # Calculate potential energy
    v = total_potential(r_r, nuclei_pos, nuclei_charge)

    # Local energy
    e_l = t + v

    return e_l


def make_batched_local_energy(log_psi: Callable, n_electrons: int) -> Callable:
    """Factory for batched local energy.

    Args:
        log_psi: Function with signature `(params, r_batch) -> log|psi|` where
            `r_batch` has shape [batch, n_elec, 3] and the return has shape [batch].
        n_electrons: Number of electrons.

    Returns:
        Function `(params, r_batch, nuclei_pos, nuclei_charge) -> local_E` where
        `local_E` has shape [batch].

    The returned function avoids Python loops and does not construct grad/hessian
    transforms inside the per-sample path.
    """

    def log_psi_flat(params, r_flat: jnp.ndarray) -> jnp.ndarray:
        r_single = r_flat.reshape(n_electrons, 3)
        return log_psi(params, r_single[None, :, :])[0]

    grad_log_psi_flat = jax.grad(log_psi_flat, argnums=1)

    def kinetic_single(params, r_single: jnp.ndarray) -> jnp.ndarray:
        r_flat = r_single.reshape(-1)
        grad, grad_jvp_fn = jax.linearize(
            lambda r: grad_log_psi_flat(params, r), r_flat
        )
        grad_squared_sum = jnp.sum(grad**2)

        n = r_flat.shape[0]

        def body_fun(i, val):
            e_i = jax.nn.one_hot(i, n)
            _, dgrad = jax.jvp(
                lambda r: grad_log_psi_flat(params, r), (r_flat,), (e_i,)
            )
            return val + dgrad[i]

        laplacian = jax.lax.fori_loop(
            0, n, body_fun, jnp.zeros((), dtype=r_flat.dtype)
        )
        return -0.5 * (grad_squared_sum + laplacian)

    def local_energy_single(
        params, r_single: jnp.ndarray, nuclei_pos, nuclei_charge
    ) -> jnp.ndarray:
        t = kinetic_single(params, r_single)
        v = total_potential(r_single, nuclei_pos, nuclei_charge)
        return t + v

    return jax.vmap(local_energy_single, in_axes=(None, 0, None, None))
