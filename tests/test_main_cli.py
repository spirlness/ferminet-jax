import sys
import types

import ferminet
from ferminet import base_config
from ferminet import main as main_module


def _patch_train_module(monkeypatch, called):
    dummy_module = types.SimpleNamespace(train=lambda cfg: called.append(cfg))
    monkeypatch.setitem(sys.modules, "ferminet.train", dummy_module)
    monkeypatch.setattr(ferminet, "train", dummy_module, raising=False)


def test_main_uses_provided_config(monkeypatch):
    cfg = base_config.default()
    called = []
    _patch_train_module(monkeypatch, called)
    monkeypatch.setattr(main_module, "_CONFIG", types.SimpleNamespace(value=cfg))

    main_module.main(["prog"])

    assert called and called[0] is cfg


def test_main_falls_back_to_default_config(monkeypatch):
    called = []
    _patch_train_module(monkeypatch, called)
    monkeypatch.setattr(main_module, "_CONFIG", types.SimpleNamespace(value=None))

    main_module.main(["prog"])

    assert called and called[0].batch_size == base_config.default().batch_size
