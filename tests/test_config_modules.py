import ml_collections

from ferminet.configs import (
    helium,
    helium_max,
    helium_quick,
    helium_scaled,
    hydrogen,
    lithium,
)


def _assert_basic_config(cfg: ml_collections.ConfigDict) -> None:
    assert cfg.system.molecule
    assert cfg.system.electrons
    assert cfg.batch_size > 0


def test_helium_configs_cover_all_variants():
    configs = {
        "helium": (helium.get_config(), 4096),
        "helium_quick": (helium_quick.get_config(), 256),
        "helium_max": (helium_max.get_config(), 4096),
        "helium_scaled": (helium_scaled.get_config(), 8192),
    }

    for name, (cfg, batch_size) in configs.items():
        _assert_basic_config(cfg)
        assert cfg.batch_size == batch_size, name

    assert configs["helium_scaled"][0].batch_size > configs["helium"][0].batch_size


def test_other_molecule_configs_have_expected_spins():
    hydrogen_cfg = hydrogen.get_config()
    lithium_cfg = lithium.get_config()

    _assert_basic_config(hydrogen_cfg)
    _assert_basic_config(lithium_cfg)

    assert hydrogen_cfg.system.electrons == (1, 0)
    assert lithium_cfg.system.electrons == (2, 1)
