# Copyright 2024 FermiNet Authors.
# Licensed under the Apache License, Version 2.0.

"""Numerical utilities for FermiNet."""

import jax.numpy as jnp

# Global epsilon for numerical stability.
# Used in norms, denominators, and square roots to prevent NaNs.
EPS = 1.0e-12


def safe_norm(
    x: jnp.ndarray, axis: int = -1, keepdims: bool = False, eps: float = EPS
) -> jnp.ndarray:
    """Compute norm with epsilon stabilization inside square root."""
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims) + eps)


def safe_inv(x: jnp.ndarray, eps: float = EPS) -> jnp.ndarray:
    """Compute 1/(x + eps) for stability."""
    return 1.0 / (x + eps)
