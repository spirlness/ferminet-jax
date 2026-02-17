import jax
import jax.numpy as jnp
import ml_collections
import pytest

from ferminet import base_config, checkpoint, train, types


def _build_cfg() -> ml_collections.ConfigDict:
    cfg = base_config.default()
    cfg.system.molecule = [("He", (0.0, 0.0, 0.0))]
    cfg.system.electrons = (1, 1)
    cfg.system.charges = (2,)
    cfg.log.save_path = "./checkpoints/test"
    cfg.log.restore = False
    cfg.log.restore_path = None
    cfg.mcmc.use_langevin = False
    return cfg


def test_make_schedule_and_filter_kwargs_behaviour():
    cfg = _build_cfg()
    cfg.optim.lr.rate = 0.01
    cfg.optim.lr.decay = 0.5
    cfg.optim.lr.delay = 10.0
    schedule = train.make_schedule(cfg)
    assert float(schedule(0)) == pytest.approx(0.0)
    assert float(schedule(1000)) > 0.0

    def fn(a, b):
        return a + b

    filtered = train._filter_kwargs(fn, {"a": 1, "b": 2, "c": 3})
    assert filtered == {"a": 1, "b": 2}


def test_prepare_system_and_init_mcmc_data_shapes():
    cfg = _build_cfg()
    atoms, charges, spins, ndim = train._prepare_system(cfg)
    assert atoms.shape == (1, 3)
    assert charges.shape == (1,)
    assert spins == (1, 1)
    assert ndim == 3

    data = train._init_mcmc_data(
        jax.random.PRNGKey(0), atoms, charges, spins, batch_size=4, ndim=ndim
    )
    assert data.positions.shape == (4, sum(spins) * ndim)
    assert data.spins.shape == (2,)


def _dummy_data() -> types.FermiNetData:
    return types.FermiNetData(
        positions=jnp.zeros((2, 6)),
        spins=jnp.array([0, 1]),
        atoms=jnp.zeros((1, 3)),
        charges=jnp.array([2.0]),
    )


def test_restore_checkpoint_with_explicit_path(tmp_path):
    cfg = _build_cfg()
    cfg.log.save_path = str(tmp_path)
    params = {"w": jnp.array([1.0])}
    opt_state = {"m": jnp.array([0.0])}
    data = _dummy_data()
    ckpt_path = checkpoint.save_checkpoint(
        tmp_path, 5, params=params, opt_state=opt_state, mcmc_state=data
    )
    cfg.log.restore_path = str(ckpt_path)

    restored = train._restore_checkpoint(cfg, None, None, None, step=0)
    restored_params, restored_opt, restored_data, restored_step = restored

    assert jnp.array_equal(restored_params["w"], params["w"])
    assert restored_step == 5
    assert restored_opt["m"].shape == opt_state["m"].shape
    assert restored_data.positions.shape == data.positions.shape


def test_restore_checkpoint_uses_latest_when_flag_set(tmp_path):
    cfg = _build_cfg()
    cfg.log.save_path = str(tmp_path)
    cfg.log.restore = True

    params = {"w": jnp.array([2.0])}
    opt_state = {"m": jnp.array([1.0])}
    data = _dummy_data()

    checkpoint.save_checkpoint(tmp_path, 1, params, opt_state, data)
    checkpoint.save_checkpoint(tmp_path, 3, params, opt_state, data)

    restored = train._restore_checkpoint(cfg, params, opt_state, data, step=0)
    _, _, _, restored_step = restored
    assert restored_step == 3


def test_prepare_system_requires_molecule():
    cfg = base_config.default()
    cfg.system.molecule = []
    with pytest.raises(ValueError):
        train._prepare_system(cfg)


def test_shard_array_and_data(monkeypatch):
    monkeypatch.setattr(
        train.kfac_jax.utils, "broadcast_all_local_devices", lambda x: x, raising=False
    )
    monkeypatch.setattr(
        train.kfac_jax.utils, "replicate_all_local_devices", lambda x: x, raising=False
    )

    arr = jnp.arange(12).reshape(4, 3)
    sharded = train._shard_array(arr, device_count=2)
    assert sharded.shape == (2, 2, 3)

    with pytest.raises(ValueError):
        train._shard_array(arr, device_count=3)

    data = _dummy_data()._replace(positions=jnp.zeros((4, 6)))
    sharded_data = train._shard_data(data, device_count=2)
    assert sharded_data.positions.shape == (2, 2, 6)


def test_build_network_returns_callable():
    cfg = _build_cfg()
    cfg.network.determinants = 2
    cfg.network.ferminet.hidden_dims = ((16, 4),)
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([2.0])
    spins = (1, 1)

    init_fn, apply_fn, extras = train._build_network(cfg, atoms, charges, spins)
    params = init_fn(jax.random.PRNGKey(0))
    electrons = jnp.zeros((2, sum(spins) * 3))
    spins_arr = jnp.array([0, 1])

    log_psi = apply_fn(params, electrons, spins_arr, atoms, charges)
    assert log_psi.shape == (2,)
    assert extras["factory"] in {
        "make_fermi_net",
        "make_ferminet",
        "make_network",
        "build_network",
    }


def test_p_split_uses_kfac_utils(monkeypatch):
    captured = {}

    def fake_p_split(key):
        captured["key"] = key
        return key, key

    monkeypatch.setattr(train.kfac_jax.utils, "p_split", fake_p_split, raising=False)

    key1, key2 = train._p_split(jax.random.PRNGKey(0))
    assert "key" in captured
    assert jnp.array_equal(key1, key2)


def test_train_loop_with_stubbed_dependencies(monkeypatch, tmp_path):
    cfg = _build_cfg()
    cfg.batch_size = 1
    cfg.optim.optimizer = "adam"
    cfg.optim.iterations = 1
    cfg.log.print_every = 1
    cfg.log.checkpoint_every = 10
    cfg.log.save_path = str(tmp_path)
    cfg.mcmc.adapt_frequency = 10
    cfg.mcmc.steps = 1
    cfg.mcmc.use_langevin = False

    def stub_build_network(cfg, atoms, charges, spins):
        def init_fn(key):
            _ = key
            return {"w": jnp.array(0.0)}

        def apply_fn(params, positions, spins_arr, atoms_arr, charges_arr):
            _ = (params, spins_arr, atoms_arr, charges_arr)
            return jnp.zeros(positions.shape[0])

        return init_fn, apply_fn, {"factory": "make_fermi_net"}

    def stub_local_energy_fn(*args, **kwargs):
        def local_energy(params, key, data):
            _ = (params, key, data)
            return jnp.zeros((data.positions.shape[0],))

        return local_energy

    class Aux:
        def __init__(self):
            self.variance = jnp.array(0.0)

    def stub_make_loss(*args, **kwargs):
        def loss_fn(params, key, data):
            _ = (params, key, data)
            return jnp.array(0.0), Aux()

        return loss_fn

    def stub_mcmc_step(*args, **kwargs):
        def step(params, data, key, width):
            _ = (params, key, width)
            return data, jnp.array(0.5)

        return step

    monkeypatch.setattr(train, "_build_network", stub_build_network)
    monkeypatch.setattr(train, "_make_local_energy_fn", stub_local_energy_fn)
    monkeypatch.setattr(train.loss, "make_loss", stub_make_loss)
    monkeypatch.setattr(train.mcmc, "make_mcmc_step", stub_mcmc_step)
    monkeypatch.setattr(train.constants, "pmap", lambda fn, *a, **k: fn, raising=False)

    def fake_pmean(x):
        return jax.tree_util.tree_map(
            lambda leaf: leaf[None] if jnp.asarray(leaf).ndim == 0 else leaf,
            x,
        )

    monkeypatch.setattr(train.constants, "pmean", fake_pmean, raising=False)
    monkeypatch.setattr(train.jax, "pmap", lambda fn: fn, raising=False)

    def fake_device_get(x):
        def _get(leaf):
            arr = jnp.asarray(leaf)
            if arr.ndim == 0:
                arr = arr[None]
            return arr

        return jax.tree_util.tree_map(_get, x)

    monkeypatch.setattr(train.jax, "device_get", fake_device_get, raising=False)
    monkeypatch.setattr(
        train.kfac_jax.utils, "replicate_all_local_devices", lambda x: x, raising=False
    )
    monkeypatch.setattr(
        train.kfac_jax.utils,
        "broadcast_all_local_devices",
        lambda x: x,
        raising=False,
    )
    monkeypatch.setattr(
        train.kfac_jax.utils,
        "make_different_rng_key_on_all_devices",
        lambda key: key,
        raising=False,
    )
    monkeypatch.setattr(
        train.kfac_jax.utils, "p_split", lambda key: (key, key), raising=False
    )
    monkeypatch.setattr(train.checkpoint, "save_checkpoint", lambda *a, **k: None)

    result = train.train(cfg)
    assert result["step"] == cfg.optim.iterations
