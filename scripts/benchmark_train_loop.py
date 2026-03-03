import time

from ferminet import train
from ferminet.configs import helium_quick

cfg = helium_quick.get_config()
cfg.optim.iterations = 50
cfg.log.print_every = 1

start = time.time()
train.train(cfg)
print(f"Total time: {time.time() - start:.3f} s")
