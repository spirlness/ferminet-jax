import time
import ml_collections

import os

# Mock JAX compilation cache dir before loading JAX
os.environ["JAX_CACHE_DIR"] = "/tmp/ferminet_jax_cache"

import jax
import jax.numpy as jnp

from ferminet import train
from ferminet.configs import helium_quick


def benchmark():
    cfg = helium_quick.get_config()
    cfg.optim.iterations = 50
    cfg.log.print_every = 10
    cfg.mcmc.adapt_frequency = 10
    cfg.log.checkpoint_every = 10

    start = time.time()
    train.train(cfg)
    end = time.time()
    print(f"Time: {end - start:.4f} seconds")


if __name__ == "__main__":
    benchmark()
