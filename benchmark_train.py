import time
import jax
import jax.numpy as jnp
from ferminet import train
from ferminet.configs import helium_quick
import os

os.environ["JAX_CACHE_DIR"] = "/tmp/ferminet_jax_cache"

cfg = helium_quick.get_config()
cfg.optim.iterations = 100
cfg.optim.optimizer = "adam"

t0 = time.time()
result = train.train(cfg)
t1 = time.time()
print(f"Total time: {t1 - t0:.2f}s")
