import ml_collections
from ferminet import base_config


def get_config() -> ml_collections.ConfigDict:
    cfg = base_config.default()

    cfg.system.molecule = [("H", (0.0, 0.0, 0.0))]
    cfg.system.electrons = (1, 0)
    cfg.system.charges = (1,)

    cfg.batch_size = 2048
    cfg.optim.iterations = 100_000
    cfg.network.determinants = 4
    cfg.network.ferminet.hidden_dims = ((64, 16), (64, 16), (64, 16))

    return cfg
