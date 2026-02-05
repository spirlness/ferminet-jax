"""Multiplicative envelope functions for orbitals."""

import enum
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp


class EnvelopeType(enum.Enum):
    """Supported envelope types."""

    ISOTROPIC = "isotropic"
    DIAGONAL = "diagonal"
    FULL = "full"


EnvelopeParams = Mapping[str, jnp.ndarray]
EnvelopeOutputs = tuple[jnp.ndarray | None, ...]
EnvelopeInit = Callable[[int, Sequence[int], int], EnvelopeParams]
EnvelopeApply = Callable[
    [EnvelopeParams, jnp.ndarray, jnp.ndarray, Sequence[int]], EnvelopeOutputs
]


@dataclass(frozen=True)
class Envelope:
    """Envelope function container.

    Attributes:
        init: Function to initialize envelope parameters.
        apply: Function to apply envelope to orbitals.
    """

    init: EnvelopeInit
    apply: EnvelopeApply


def _normalize_output_dims(output_dims: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(dim) for dim in output_dims)


def _ensure_distance_axis(r_ae: jnp.ndarray) -> jnp.ndarray:
    if r_ae.ndim == 2:
        return r_ae[..., None]
    return r_ae


def _positive(values: jnp.ndarray) -> jnp.ndarray:
    """Ensure decay parameters remain positive."""
    return cast(jnp.ndarray, jax.nn.softplus(values))


def make_isotropic_envelope() -> Envelope:
    """Creates an isotropic (exponential decay) envelope.

    The envelope is: exp(-sum_i sigma_i * |r - R_i|)
    where sigma_i are learnable parameters per atom.
    """

    def init(natoms: int, output_dims: Sequence[int], ndim: int = 3) -> EnvelopeParams:
        del ndim
        params: dict[str, jnp.ndarray] = {}
        output_dims = _normalize_output_dims(output_dims)
        for i, norb in enumerate(output_dims):
            if norb > 0:
                params[f"sigma_{i}"] = jnp.ones((natoms, norb))
                params[f"pi_{i}"] = jnp.ones((natoms, norb))
        return params

    def apply(
        params: EnvelopeParams,
        ae: jnp.ndarray,
        r_ae: jnp.ndarray,
        output_dims: Sequence[int],
    ) -> EnvelopeOutputs:
        """Apply isotropic envelope: exp(-sigma * r_ae)."""
        del ae
        envelopes: list[jnp.ndarray | None] = []
        output_dims = _normalize_output_dims(output_dims)
        r_ae = _ensure_distance_axis(r_ae)
        for i, norb in enumerate(output_dims):
            if norb > 0:
                sigma = _positive(params[f"sigma_{i}"])
                pi = params[f"pi_{i}"]
                envelope = jnp.sum(pi * jnp.exp(-sigma * r_ae), axis=1)
                envelopes.append(envelope)
            else:
                envelopes.append(None)
        return tuple(envelopes)

    return Envelope(init=init, apply=apply)


def make_diagonal_envelope() -> Envelope:
    """Creates a diagonal (separate decay per dimension) envelope."""

    def init(natoms: int, output_dims: Sequence[int], ndim: int = 3) -> EnvelopeParams:
        params: dict[str, jnp.ndarray] = {}
        output_dims = _normalize_output_dims(output_dims)
        for i, norb in enumerate(output_dims):
            if norb > 0:
                params[f"sigma_{i}"] = jnp.ones((natoms, norb, ndim))
                params[f"pi_{i}"] = jnp.ones((natoms, norb))
        return params

    def apply(
        params: EnvelopeParams,
        ae: jnp.ndarray,
        r_ae: jnp.ndarray,
        output_dims: Sequence[int],
    ) -> EnvelopeOutputs:
        """Apply diagonal envelope with per-dimension decay rates."""
        del r_ae
        envelopes: list[jnp.ndarray | None] = []
        output_dims = _normalize_output_dims(output_dims)
        for i, norb in enumerate(output_dims):
            if norb > 0:
                sigma = _positive(params[f"sigma_{i}"])
                pi = params[f"pi_{i}"]
                ae_expanded = ae[:, :, None, :]
                decay = jnp.sum(sigma * jnp.abs(ae_expanded), axis=-1)
                envelope = jnp.sum(pi * jnp.exp(-decay), axis=1)
                envelopes.append(envelope)
            else:
                envelopes.append(None)
        return tuple(envelopes)

    return Envelope(init=init, apply=apply)


def make_full_envelope() -> Envelope:
    """Creates a full (anisotropic) envelope with learnable directions."""

    def init(natoms: int, output_dims: Sequence[int], ndim: int = 3) -> EnvelopeParams:
        params: dict[str, jnp.ndarray] = {}
        output_dims = _normalize_output_dims(output_dims)
        eye = jnp.eye(ndim)
        for i, norb in enumerate(output_dims):
            if norb > 0:
                params[f"sigma_{i}"] = jnp.ones((natoms, norb))
                params[f"pi_{i}"] = jnp.ones((natoms, norb))
                params[f"eta_{i}"] = jnp.tile(eye, (natoms, norb, 1, 1))
        return params

    def apply(
        params: EnvelopeParams,
        ae: jnp.ndarray,
        r_ae: jnp.ndarray,
        output_dims: Sequence[int],
    ) -> EnvelopeOutputs:
        """Apply full envelope with learnable linear directions."""
        del r_ae
        envelopes: list[jnp.ndarray | None] = []
        output_dims = _normalize_output_dims(output_dims)
        for i, norb in enumerate(output_dims):
            if norb > 0:
                sigma = _positive(params[f"sigma_{i}"])
                pi = params[f"pi_{i}"]
                eta = params[f"eta_{i}"]
                ae_expanded = ae[:, :, None, :, None]
                eta_expanded = eta[None, :, :, :, :]
                ae_transformed = jnp.sum(ae_expanded * eta_expanded, axis=-2)
                r_transformed = jnp.sqrt(jnp.sum(ae_transformed**2, axis=-1) + 1.0e-12)
                envelope = jnp.sum(pi * jnp.exp(-sigma * r_transformed), axis=1)
                envelopes.append(envelope)
            else:
                envelopes.append(None)
        return tuple(envelopes)

    return Envelope(init=init, apply=apply)
