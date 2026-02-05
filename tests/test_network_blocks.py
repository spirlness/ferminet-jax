import jax
import jax.numpy as jnp

from ferminet import network_blocks


def test_init_linear_layer_returns_expected_shapes():
    key = jax.random.PRNGKey(0)
    params = network_blocks.init_linear_layer(key, in_dim=3, out_dim=2)
    assert params["w"].shape == (3, 2)
    assert params["b"].shape == (2,)


def test_linear_layer_applies_bias():
    x = jnp.ones((1, 3))
    w = jnp.eye(3)
    b = jnp.array([1.0, 2.0, 3.0])
    result = network_blocks.linear_layer(x, w, b)
    assert jnp.allclose(result, x + b)


def test_array_partitions_and_split_into_blocks():
    partitions = network_blocks.array_partitions([2, 2])
    assert partitions == (2,)

    arr = jnp.arange(16).reshape(4, 4)
    blocks = network_blocks.split_into_blocks(arr, [2, 2])
    assert len(blocks) == 4
    assert blocks[0].shape == (2, 2)
    assert jnp.array_equal(blocks[0], jnp.array([[0, 1], [4, 5]]))
