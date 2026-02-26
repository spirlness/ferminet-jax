import ml_collections

from ferminet import base_config


def get_config() -> ml_collections.ConfigDict:
    cfg = base_config.default()

    # H₂ molecule at equilibrium bond length (1.4 bohr ≈ 0.74 Å)
    cfg.system.molecule = [("H", (0.0, 0.0, 0.0)), ("H", (1.4, 0.0, 0.0))]
    cfg.system.electrons = (1, 1)
    cfg.system.charges = (1, 1)

    cfg.network.determinants = 16
    cfg.network.ferminet.hidden_dims = (
        (128, 32),
        (128, 32),
        (128, 32),
        (128, 32),
    )

    cfg.batch_size = 4096
    cfg.optim.iterations = 100_000
    cfg.optim.lr.rate = 0.005
    cfg.optim.lr.decay = 1.0
    cfg.optim.lr.delay = 10000.0
    cfg.optim.clip_local_energy = 3.0

    cfg.mcmc.steps = 10
    cfg.mcmc.burn_in = 100
    cfg.mcmc.move_width = 0.5
    cfg.mcmc.target_acceptance = 0.55
    cfg.mcmc.adapt_frequency = 20

    cfg.log.print_every = 100
    cfg.log.checkpoint_every = 5000
    cfg.log.save_path = "./checkpoints/h2"

    return cfg
