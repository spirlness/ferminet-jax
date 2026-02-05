"""Jastrow factors for FermiNet."""

import enum
from collections.abc import Callable, Mapping

import jax.numpy as jnp


class JastrowType(enum.Enum):
    """Type of Jastrow factor to use."""

    NONE = "none"
    SIMPLE_EE = "simple_ee"


JastrowInit = Callable[[], Mapping[str, jnp.ndarray]]
JastrowApply = Callable[
    [Mapping[str, jnp.ndarray], jnp.ndarray, jnp.ndarray],
    jnp.ndarray,
]


def get_jastrow(jastrow_type: JastrowType | str) -> tuple[JastrowInit, JastrowApply]:
    """Returns init and apply functions for the specified Jastrow type.

    Args:
        jastrow_type: Type of Jastrow factor.

    Returns:
        Tuple of (init, apply) functions.
    """
    if jastrow_type == JastrowType.NONE or jastrow_type == "none":
        return _no_jastrow()
    if jastrow_type == JastrowType.SIMPLE_EE or jastrow_type == "simple_ee":
        return _simple_ee_jastrow()
    raise ValueError(f"Unknown Jastrow type: {jastrow_type}")


def _no_jastrow() -> tuple[JastrowInit, JastrowApply]:
    """No Jastrow factor."""

    def init() -> Mapping[str, jnp.ndarray]:
        return {}

    def apply(
        params: Mapping[str, jnp.ndarray],
        r_ee: jnp.ndarray,
        spins: jnp.ndarray,
    ) -> jnp.ndarray:
        _ = (params, r_ee, spins)
        return jnp.zeros(())

    return init, apply


def _simple_ee_jastrow() -> tuple[JastrowInit, JastrowApply]:
    """Simple electron-electron Jastrow factor.

    J = sum_{i<j} a / (1 + b * r_ij)

    where a, b are learnable parameters and r_ij is electron-electron distance.
    """

    def init() -> Mapping[str, jnp.ndarray]:
        return {
            "a": jnp.array(1.0),
            "b": jnp.array(1.0),
        }

    def apply(
        params: Mapping[str, jnp.ndarray],
        r_ee: jnp.ndarray,
        spins: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply simple e-e Jastrow.

        Args:
            params: Jastrow parameters {"a": float, "b": float}
            r_ee: Electron-electron distances (nelec, nelec)
            spins: Electron spins (nelec,)

        Returns:
            Scalar Jastrow factor value (log scale).
        """
        _ = spins
        a = params["a"]
        b = params["b"]

        nelec = r_ee.shape[0]
        mask = jnp.triu(jnp.ones((nelec, nelec)), k=1)
        jastrow_terms = a / (1.0 + b * r_ee) * mask
        return jnp.sum(jastrow_terms)

    return init, apply
