import ml_collections
from ferminet import base_config


def get_config() -> ml_collections.ConfigDict:
    cfg = base_config.default()

    cfg.system.molecule = [("He", (0.0, 0.0, 0.0))]
    cfg.system.electrons = (1, 1)
    cfg.system.charges = (2,)

    cfg.batch_size = 256
    cfg.optim.iterations = 20
    cfg.log.print_every = 5
    cfg.network.determinants = 4
    cfg.network.ferminet.hidden_dims = ((32, 8), (32, 8))

    return cfg
