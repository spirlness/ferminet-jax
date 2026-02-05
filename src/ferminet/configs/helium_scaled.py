import ml_collections
from ferminet import base_config


def get_config() -> ml_collections.ConfigDict:
    cfg = base_config.default()

    cfg.system.molecule = [("He", (0.0, 0.0, 0.0))]
    cfg.system.electrons = (1, 1)
    cfg.system.charges = (2,)

    cfg.network.determinants = 32
    cfg.network.ferminet.hidden_dims = (
        (256, 32),
        (256, 32),
        (256, 32),
        (256, 32),
    )
    cfg.network.full_det = True

    cfg.batch_size = 8192

    cfg.optim.iterations = 100_000
    cfg.optim.lr.rate = 0.005
    cfg.optim.lr.decay = 1.0
    cfg.optim.lr.delay = 10000.0
    cfg.optim.kfac.damping = 0.01
    cfg.optim.kfac.norm_constraint = 0.001
    cfg.optim.clip_local_energy = 3.0

    cfg.mcmc.steps = 10
    cfg.mcmc.burn_in = 100
    cfg.mcmc.move_width = 0.5
    cfg.mcmc.target_acceptance = 0.55
    cfg.mcmc.adapt_frequency = 20
    cfg.mcmc.pmove_max = 0.60
    cfg.mcmc.pmove_min = 0.50

    cfg.log.print_every = 100
    cfg.log.checkpoint_every = 5000
    cfg.log.save_path = "./checkpoints/helium_scaled"

    return cfg
