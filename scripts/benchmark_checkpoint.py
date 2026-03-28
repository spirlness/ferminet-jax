"""Benchmark checkpointing latency.

This script runs the core training path with checkpointing enabled every step
to benchmark the performance difference between sequential jax.device_get vs
grouped jax.device_get.
"""

import time
import jax
import sys
from ml_collections import ConfigDict

from ferminet.configs import helium_quick
from ferminet import train

def main() -> None:
    print("Starting checkpoint benchmark...")

    # Get standard config
    cfg = helium_quick.get_config()
    cfg.optim.iterations = 5
    cfg.log.checkpoint_every = 1

    t0 = time.time()
    result = train.train(cfg)
    wall_time = time.time() - t0

    print(f"Benchmark completed in {wall_time:.3f} seconds.")
    print(f"Final step: {result['step']}")

if __name__ == "__main__":
    main()
