import ml_collections
from ferminet import base_config


def get_config() -> ml_collections.ConfigDict:
    cfg = base_config.default()

    cfg.system.molecule = [("Li", (0.0, 0.0, 0.0))]
    cfg.system.electrons = (2, 1)
    cfg.system.charges = (3,)

    cfg.batch_size = 4096
    cfg.optim.iterations = 500_000
    cfg.network.determinants = 16
    cfg.network.ferminet.hidden_dims = ((256, 32), (256, 32), (256, 32), (256, 32))

    return cfg
