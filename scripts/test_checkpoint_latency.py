import jax
import jax.numpy as jnp
import time
from ferminet import train
from ferminet.configs import helium_quick

def run():
    cfg = helium_quick.get_config()
    cfg.optim.iterations = 10
    cfg.log.print_every = 5
    cfg.log.checkpoint_every = 1 # benchmark checkpoint
    t0 = time.time()
    train.train(cfg)
    t1 = time.time()
    print(f"Total time: {t1 - t0:.2f} s")

run()
