"""Compatibility shim for legacy imports."""

from ferminet.multi_determinant import (  # noqa: F401
    MultiDeterminantOrbitals,
    create_multi_determinant_orbitals,
)

__all__ = ["MultiDeterminantOrbitals", "create_multi_determinant_orbitals"]
