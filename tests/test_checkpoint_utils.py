import jax.numpy as jnp
import pytest

from ferminet import checkpoint


def test_save_and_restore_roundtrip(tmp_path):
    params = {"w": jnp.arange(3.0)}
    opt_state = {"m": jnp.zeros(3)}
    mcmc_state = {"positions": jnp.ones((2, 2))}

    ckpt_path = checkpoint.save_checkpoint(
        tmp_path, 12, params=params, opt_state=opt_state, mcmc_state=mcmc_state
    )

    restored = checkpoint.restore_checkpoint(str(ckpt_path))

    assert restored.step == 12
    assert jnp.allclose(restored.params["w"], params["w"])
    assert jnp.allclose(restored.opt_state["m"], opt_state["m"])
    assert jnp.allclose(restored.mcmc_state["positions"], mcmc_state["positions"])


def test_checkpoint_listing_and_latest(tmp_path):
    for step in (5, 10, 15):
        checkpoint.save_checkpoint(
            tmp_path,
            step,
            params={"value": jnp.array(float(step))},
            opt_state={},
            mcmc_state={},
        )

    listings = checkpoint.list_checkpoints(tmp_path)
    steps = [step for _, step in listings]
    assert steps == [5, 10, 15]

    latest_path, latest_data = checkpoint.find_latest_checkpoint(tmp_path)
    assert str(latest_path).endswith("00000015.pkl")
    assert latest_data.step == 15

    assert checkpoint.checkpoint_exists(tmp_path)
    assert checkpoint.checkpoint_exists(tmp_path, step=10)
    assert not checkpoint.checkpoint_exists(tmp_path, step=99)


def test_restore_checkpoint_missing(tmp_path):
    missing = tmp_path / "checkpoint_00000042.pkl"
    with pytest.raises(FileNotFoundError):
        checkpoint.restore_checkpoint(str(missing))


def test_checkpoint_helpers_handle_missing_directories(tmp_path):
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(OSError):
        checkpoint.find_latest_checkpoint(str(missing_dir))

    assert checkpoint.checkpoint_exists(str(missing_dir)) is False
    assert checkpoint.list_checkpoints(str(missing_dir)) == []


def test_find_latest_checkpoint_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert checkpoint.find_latest_checkpoint(str(empty_dir)) is None


def test_save_and_load_model_roundtrip(tmp_path):
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    model_path = tmp_path / "model.pkl"
    written = checkpoint.save_model(str(model_path), params)
    loaded = checkpoint.load_model(str(written))
    assert jnp.allclose(loaded["w"], params["w"])
