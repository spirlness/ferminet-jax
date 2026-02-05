from typing import Callable, cast

import jax
import jax.numpy as jnp

PMAP_AXIS_NAME = "qmc_pmap_axis"


Array = jnp.ndarray


def pmap(
    fn: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> Callable[..., object]:
    """Wrapper around jax.pmap with fixed axis name."""
    pmap_fn = cast(Callable[..., Callable[..., object]], jax.pmap)
    return pmap_fn(fn, axis_name=PMAP_AXIS_NAME, *args, **kwargs)


def pmean(values: Array) -> Array:
    """Average values across all devices."""
    pmean_fn = cast(Callable[[Array, str], Array], jax.lax.pmean)
    return pmean_fn(values, PMAP_AXIS_NAME)


def all_gather(x: Array) -> Array:
    """Gather values from all devices."""
    all_gather_fn = cast(Callable[[Array, str], Array], jax.lax.all_gather)
    return all_gather_fn(x, PMAP_AXIS_NAME)
