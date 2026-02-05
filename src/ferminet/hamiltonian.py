"""Evaluating the Hamiltonian on a wavefunction."""

# pyright: reportUnknownMemberType=false

from collections.abc import Callable, Sequence
from typing import Protocol, cast

import jax
import jax.numpy as jnp
from jax import lax

from ferminet import types


class LocalEnergy(Protocol):
    def __call__(
        self,
        params: types.ParamTree,
        key: jax.Array,
        data: types.FermiNetData,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Returns local energy at a configuration."""
        ...


def local_kinetic_energy(
    f: types.FermiNetLike,
    use_scan: bool = False,
    complex_output: bool = False,
) -> Callable[[types.ParamTree, types.FermiNetData], jnp.ndarray]:
    """Creates function for local kinetic energy: -1/2 nabla^2 log|psi|.

    Uses the identity:
    T = -1/2 * (nabla^2 log|psi| + |nabla log|psi||^2)

    Args:
        f: Network function returning (sign, log_magnitude).
        use_scan: Whether to use lax.scan for Laplacian.
        complex_output: Whether output is complex.

    Returns:
        Function computing kinetic energy.
    """

    def logabs_f(
        params: types.ParamTree,
        pos: jnp.ndarray,
        spins: jnp.ndarray,
        atoms: jnp.ndarray,
        charges: jnp.ndarray,
    ) -> jnp.ndarray:
        logabs = f(params, pos, spins, atoms, charges)[1]
        if complex_output:
            logabs = jnp.real(logabs)
        return logabs

    grad_f = cast(
        Callable[
            [types.ParamTree, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
            jnp.ndarray,
        ],
        jax.grad(logabs_f, argnums=1),
    )

    def kinetic(
        params: types.ParamTree,
        data: types.FermiNetData,
    ) -> jnp.ndarray:
        positions = cast(jnp.ndarray, data.positions)
        spins = cast(jnp.ndarray, data.spins)
        atoms = cast(jnp.ndarray, data.atoms)
        charges = cast(jnp.ndarray, data.charges)
        n = positions.shape[0]
        eye = jnp.eye(n)

        def grad_closure(x: jnp.ndarray) -> jnp.ndarray:
            return grad_f(params, x, spins, atoms, charges)

        primal, dgrad_f = cast(
            tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]],
            jax.linearize(grad_closure, positions),
        )

        def hessian_diagonal(i: int) -> jnp.ndarray:
            return dgrad_f(eye[i])[i]

        if use_scan:

            def scan_body(i: int, _: None) -> tuple[int, jnp.ndarray]:
                return i + 1, hessian_diagonal(i)

            _, diagonal = lax.scan(scan_body, 0, None, length=n)
            laplacian: jnp.ndarray = jnp.sum(diagonal)
        else:

            def laplacian_body(i: int, val: jnp.ndarray) -> jnp.ndarray:
                return val + hessian_diagonal(i)

            laplacian = cast(jnp.ndarray, lax.fori_loop(0, n, laplacian_body, 0.0))

        grad_squared = jnp.sum(primal**2)
        return -0.5 * (laplacian + grad_squared)

    return kinetic


def potential_electron_electron(r_ee: jnp.ndarray) -> jnp.ndarray:
    """Electron-electron repulsion: sum_{i<j} 1/r_ij."""
    r_ee_flat = r_ee[jnp.triu_indices_from(r_ee[..., 0], k=1)]
    return jnp.sum(1.0 / r_ee_flat)


def potential_electron_nuclear(
    charges: jnp.ndarray,
    r_ae: jnp.ndarray,
) -> jnp.ndarray:
    """Electron-nuclear attraction: -sum_i sum_j Z_j/r_ij."""
    return -jnp.sum(charges / r_ae[..., 0])


def potential_nuclear_nuclear(
    charges: jnp.ndarray,
    atoms: jnp.ndarray,
) -> jnp.ndarray:
    """Nuclear-nuclear repulsion: sum_{i<j} Z_i*Z_j/r_ij."""
    r_aa = cast(
        jnp.ndarray, jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
    )
    return jnp.sum(jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))


def potential_energy(
    r_ae: jnp.ndarray,
    r_ee: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
) -> jnp.ndarray:
    """Total potential energy V = V_ee + V_en + V_nn."""
    return (
        potential_electron_electron(r_ee)
        + potential_electron_nuclear(charges, r_ae)
        + potential_nuclear_nuclear(charges, atoms)
    )


def local_energy(
    f: types.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    complex_output: bool = False,
) -> LocalEnergy:
    """Creates the local energy function E_L = T + V.

    Args:
        f: Network function.
        charges: Nuclear charges.
        nspins: Number of electrons per spin.
        use_scan: Use scan for Laplacian.
        complex_output: Complex-valued output.

    Returns:
        LocalEnergy function.
    """
    _ = nspins
    kinetic_fn = local_kinetic_energy(f, use_scan, complex_output)

    def e_l(
        params: types.ParamTree,
        key: jax.Array,
        data: types.FermiNetData,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        _ = key
        positions = cast(jnp.ndarray, data.positions)
        atoms = cast(jnp.ndarray, data.atoms)
        _, _, r_ae, r_ee = construct_input_features(positions, atoms)
        potential = potential_energy(r_ae, r_ee, atoms, charges)
        kinetic = kinetic_fn(params, data)
        total_energy = potential + kinetic
        return total_energy, None

    return e_l


def construct_input_features(
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    ndim: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Construct ae, ee vectors and distances."""
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])

    r_ae = cast(jnp.ndarray, jnp.linalg.norm(ae, axis=2, keepdims=True))
    n = ee.shape[0]
    r_ee = cast(
        jnp.ndarray,
        jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)),
    )

    return ae, ee, r_ae, r_ee[..., None]
