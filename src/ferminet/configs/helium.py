import ml_collections
from ferminet import base_config


def get_config() -> ml_collections.ConfigDict:
    cfg = base_config.default()

    cfg.system.molecule = [("He", (0.0, 0.0, 0.0))]
    cfg.system.electrons = (1, 1)
    cfg.system.charges = (2,)

    cfg.batch_size = 4096
    cfg.optim.iterations = 200_000
    cfg.network.determinants = 8
    cfg.network.ferminet.hidden_dims = ((128, 32), (128, 32), (128, 32), (128, 32))

    return cfg
