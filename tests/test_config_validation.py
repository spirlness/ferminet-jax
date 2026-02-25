"""Tests for validate_config in base_config.py."""

import pytest
import ml_collections

from ferminet import base_config


def _make_valid_config() -> ml_collections.ConfigDict:
    """Return a minimal valid config for validation tests."""
    cfg = base_config.default()
    cfg = base_config.resolve(cfg)
    return cfg


def test_valid_config_passes():
    """A well-formed config should not raise."""
    cfg = _make_valid_config()
    base_config.validate_config(cfg)  # should not raise


def test_invalid_batch_size():
    cfg = _make_valid_config()
    cfg.batch_size = 0
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        base_config.validate_config(cfg)


def test_invalid_optimizer():
    cfg = _make_valid_config()
    cfg.optim.optimizer = "sgd"
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        base_config.validate_config(cfg)


def test_invalid_laplacian_mode():
    cfg = _make_valid_config()
    cfg.optim.laplacian = "magic"
    with pytest.raises(ValueError, match="Unsupported laplacian mode"):
        base_config.validate_config(cfg)


def test_invalid_iterations():
    cfg = _make_valid_config()
    cfg.optim.iterations = -1
    with pytest.raises(ValueError, match="optim.iterations must be > 0"):
        base_config.validate_config(cfg)
