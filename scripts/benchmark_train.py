import jax
import jax.numpy as jnp
import time
from ferminet import train, base_config
from ml_collections import ConfigDict

def benchmark():
    cfg = base_config.default()
    cfg.system.molecule = [("H", (0.0, 0.0, 0.0))]
    cfg.system.electrons = (1, 1)
    cfg.system.charges = (1.0,)
    cfg.batch_size = 32
    cfg.optim.iterations = 100
    cfg.log.print_every = 10

    start = time.time()
    train.train(cfg)
    print("Time taken:", time.time() - start)

if __name__ == "__main__":
    benchmark()
