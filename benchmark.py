import time
from ferminet import base_config, train
from ferminet.configs import helium

cfg = helium.get_config()
cfg.optim.optimizer = "adam"
cfg.optim.iterations = 100
cfg.log.print_every = 1

t0 = time.time()
train.train(cfg)
t1 = time.time()
print(f"Total time for 100 steps (print_every=1): {t1 - t0:.3f}s")
