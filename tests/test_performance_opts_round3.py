"""Tests for performance optimization around train loop."""

import jax.numpy as jnp
import pytest

from ferminet.train import ENERGY, LEARNING_RATE, PMOVE, VARIANCE


def test_train_loop_stats_indexing():
    """Verify train.py constants index into a simulated device_get array correctly."""
    stats_host = jnp.array([1.23, 4.56, 0.55, 1e-4])
    assert float(stats_host[ENERGY]) == pytest.approx(1.23)
    assert float(stats_host[VARIANCE]) == pytest.approx(4.56)
    assert float(stats_host[PMOVE]) == pytest.approx(0.55)
    assert float(stats_host[LEARNING_RATE]) == pytest.approx(1e-4)
