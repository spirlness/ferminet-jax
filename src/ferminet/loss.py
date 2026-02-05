# pyright: reportAttributeAccessIssue=false, reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedFunction=false, reportUntypedClassDecorator=false

"""VMC loss with custom JVP for unbiased gradients.

This module mirrors the unbiased VMC gradient estimator used in the
DeepMind FermiNet implementation. The forward pass computes the mean local
energy, while the custom JVP replaces the parameter derivative with the
covariance-based estimator:

    dL = 2 * mean((E_L - mean(E_L)) * d_log_psi)

The auxiliary outputs provide diagnostics such as local energy statistics
and clipping for monitoring stability.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import cast

import chex  # type: ignore[reportMissingImports]
import jax
import jax.numpy as jnp
import kfac_jax

try:
    from ferminet import constants, hamiltonian, types
except ImportError:  # pragma: no cover - optional module during bootstrap.
    from ferminet import constants, types

    hamiltonian = None

_ = (constants, hamiltonian)


Array = jnp.ndarray


@dataclasses.dataclass
@chex.dataclass
class AuxiliaryLossData:
    """Auxiliary outputs for diagnostics and monitoring."""

    variance: Array
    local_energy: Array
    clipped_energy: Array
    grad_local_energy: Array | None


def clip_local_energy(
    local_energy: Array,
    clip_factor: float,
    center_at_clip: bool = True,
) -> Array:
    """Clip local energies using median absolute deviation.

    Args:
        local_energy: Local energy values with leading batch dimension.
        clip_factor: Number of MADs to keep about the center.
        center_at_clip: Center the clip window at the median (True) or mean.

    Returns:
        Clipped local energy values with the same shape as input.
    """
    if clip_factor <= 0.0:
        return local_energy

    median = jnp.median(local_energy)
    mad = jnp.median(jnp.abs(local_energy - median))
    mad = jnp.maximum(mad, jnp.asarray(1e-6, dtype=local_energy.dtype))

    center = median if center_at_clip else jnp.mean(local_energy)
    half_width = clip_factor * mad
    lower = center - half_width
    upper = center + half_width
    return jnp.clip(local_energy, lower, upper)


clip_local_energy_values = clip_local_energy


def _variance(local_energy: Array) -> Array:
    """Returns batch variance of local energy.

    The variance is computed with the population formula to keep the
    estimator consistent with the mean used in the loss.
    """
    mean_energy = jnp.mean(local_energy)
    return jnp.mean((local_energy - mean_energy) ** 2)


def _split_local_energy(
    local_energy_output: Array | tuple[Array, Array | None],
) -> tuple[Array, Array | None]:
    """Unpack local energy and optional gradients.

    Some local energy functions return (local_energy, grad_local_energy).
    This helper normalizes the output into a pair for downstream use.
    """
    if isinstance(local_energy_output, tuple) and len(local_energy_output) == 2:
        local_energy, grad_local_energy = local_energy_output
        return local_energy, grad_local_energy
    return local_energy_output, None


def _zeros_like_aux(aux: AuxiliaryLossData) -> AuxiliaryLossData:
    """Create a zero tangent for AuxiliaryLossData."""
    grad_local_energy = (
        None if aux.grad_local_energy is None else jnp.zeros_like(aux.grad_local_energy)
    )
    return AuxiliaryLossData(
        variance=jnp.zeros_like(aux.variance),
        local_energy=jnp.zeros_like(aux.local_energy),
        clipped_energy=jnp.zeros_like(aux.clipped_energy),
        grad_local_energy=grad_local_energy,
    )


LocalEnergyFn = Callable[
    [types.ParamTree, jax.Array, types.FermiNetData],
    Array | tuple[Array, Array | None],
]


def make_loss(
    network: types.LogFermiNetLike,
    local_energy_fn: LocalEnergyFn,
    clip_local_energy: float = 5.0,
) -> Callable[
    [types.ParamTree, jax.Array, types.FermiNetData], tuple[Array, AuxiliaryLossData]
]:
    """Create a VMC loss with a custom JVP for unbiased gradients.

    The returned loss has signature `(params, key, data)` and returns
    `(loss, aux)` so it can be used with `jax.value_and_grad` and `has_aux`.
    The forward pass uses the mean local energy, while the custom JVP
    supplies the unbiased estimator to avoid backprop bias.

    Args:
        network: Log wavefunction network returning log|psi|.
        local_energy_fn: Callable that returns local energy values.
        clip_local_energy: MAD clip factor for monitoring diagnostics.

    Returns:
        Callable `(params, key, data) -> (loss, aux)`.
    """

    def log_psi(params: types.ParamTree, data: types.FermiNetData) -> Array:
        return network(
            params,
            data.positions,
            data.spins,
            data.atoms,
            data.charges,
        )

    def local_energy_and_grad(
        params: types.ParamTree,
        key: jax.Array,
        data: types.FermiNetData,
    ) -> tuple[Array, Array | None]:
        return _split_local_energy(local_energy_fn(params, key, data))

    def loss_components(
        params: types.ParamTree,
        key: jax.Array,
        data: types.FermiNetData,
    ) -> tuple[Array, AuxiliaryLossData, Array]:
        local_energy, grad_local_energy = local_energy_and_grad(params, key, data)
        mean_energy = jnp.mean(local_energy)
        clipped_energy = clip_local_energy_fn(local_energy, clip_local_energy)
        variance = _variance(local_energy)
        aux = AuxiliaryLossData(
            variance=variance,
            local_energy=local_energy,
            clipped_energy=clipped_energy,
            grad_local_energy=grad_local_energy,
        )
        return mean_energy, aux, local_energy

    def clip_local_energy_fn(
        local_energy: Array,
        clip_factor: float,
    ) -> Array:
        return clip_local_energy_values(local_energy, clip_factor, center_at_clip=True)

    @jax.custom_jvp
    def loss(
        params: types.ParamTree,
        key: jax.Array,
        data: types.FermiNetData,
    ) -> tuple[Array, AuxiliaryLossData]:
        mean_energy, aux, _ = loss_components(params, key, data)
        return mean_energy, aux

    @loss.defjvp
    def loss_jvp(
        primals: tuple[types.ParamTree, jax.Array, types.FermiNetData],
        tangents: tuple[types.ParamTree, jax.Array, types.FermiNetData],
    ):
        params, key, data = primals
        params_t, key_t, data_t = tangents
        del key_t
        del data_t

        mean_energy, aux, local_energy = loss_components(params, key, data)

        def log_psi_params(p: types.ParamTree) -> Array:
            return log_psi(p, data)

        log_psi_value, d_log_psi = cast(
            tuple[Array, Array],
            jax.jvp(log_psi_params, (params,), (params_t,)),
        )

        kfac_jax.register_normal_predictive_distribution(log_psi_value[:, None])

        centered_energy = local_energy - jnp.mean(local_energy)
        d_loss = 2.0 * jnp.mean(centered_energy * d_log_psi)

        tangent_aux = _zeros_like_aux(aux)
        return (mean_energy, aux), (d_loss, tangent_aux)

    return loss
