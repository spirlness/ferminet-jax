from collections.abc import Iterable, MutableMapping, Sequence
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp

# Type aliases
ParamTree = jnp.ndarray | Iterable["ParamTree"] | MutableMapping[str, "ParamTree"]
Param = MutableMapping[str, jnp.ndarray]


class FermiNetData(NamedTuple):
    """Data passed to network.

    Attributes:
        positions: walker positions, shape (n_electrons * ndim).
        spins: spins of each walker, shape (n_electrons).
        atoms: atomic positions, shape (n_atoms * ndim).
        charges: atomic charges, shape (n_atoms).
    """

    positions: jnp.ndarray
    spins: jnp.ndarray
    atoms: jnp.ndarray
    charges: jnp.ndarray


class InitFermiNet(Protocol):
    def __call__(self, key: jax.Array) -> ParamTree:
        """Returns initialized parameters for the network."""
        ...


class FermiNetLike(Protocol):
    def __call__(
        self,
        params: ParamTree,
        electrons: jnp.ndarray,
        spins: jnp.ndarray,
        atoms: jnp.ndarray,
        charges: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Returns the sign and log magnitude of the wavefunction."""
        ...


class LogFermiNetLike(Protocol):
    def __call__(
        self,
        params: ParamTree,
        electrons: jnp.ndarray,
        spins: jnp.ndarray,
        atoms: jnp.ndarray,
        charges: jnp.ndarray,
    ) -> jnp.ndarray:
        """Returns the log magnitude of the wavefunction."""
        ...


class OrbitalFnLike(Protocol):
    def __call__(
        self,
        params: ParamTree,
        pos: jnp.ndarray,
        spins: jnp.ndarray,
        atoms: jnp.ndarray,
        charges: jnp.ndarray,
    ) -> Sequence[jnp.ndarray]:
        """Forward evaluation up to the orbitals."""
        ...
