import jax
import ml_collections
from ferminet import train
from ferminet.configs import helium_quick

def run():
    cfg = helium_quick.get_config()
    cfg.optim.iterations = 20
    cfg.log.print_every = 2
    cfg.log.checkpoint_every = 10
    train.train(cfg)

run()
