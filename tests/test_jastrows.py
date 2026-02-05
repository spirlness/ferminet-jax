import jax.numpy as jnp
import pytest

from ferminet import jastrows


def test_get_jastrow_none_returns_zero():
    init_fn, apply_fn = jastrows.get_jastrow(jastrows.JastrowType.NONE)
    params = init_fn()
    result = apply_fn(params, jnp.ones((2, 2)), jnp.zeros((2,)))
    assert result.shape == ()
    assert result == 0.0


def test_get_jastrow_simple_ee_matches_manual_sum():
    init_fn, apply_fn = jastrows.get_jastrow("simple_ee")
    params = init_fn()
    distances = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    spins = jnp.array([0, 1])

    result = apply_fn(params, distances, spins)
    expected = params["a"] / (1.0 + params["b"] * 1.0)
    assert jnp.isclose(result, expected)


def test_unknown_jastrow_type_raises():
    with pytest.raises(ValueError):
        jastrows.get_jastrow("unsupported")
