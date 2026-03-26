import jax
import time
import ml_collections
from ferminet import train
from ferminet.configs import helium_quick

def run():
    cfg = helium_quick.get_config()
    cfg.optim.iterations = 200
    cfg.log.print_every = 200
    cfg.log.checkpoint_every = 200
    cfg.mcmc.adapt_frequency = 100
    t0 = time.time()
    train.train(cfg)
    t1 = time.time()
    print(f"Total time: {t1 - t0:.2f} s")

run()
