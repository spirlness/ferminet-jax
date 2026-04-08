import time

from ferminet import train
from ferminet.configs import helium_quick


def main():
    cfg = helium_quick.get_config()
    cfg.optim.iterations = 50
    cfg.log.print_every = 100
    cfg.log.checkpoint_every = 1000

    # Warmup
    print("Warming up...")
    train.train(cfg)

    print("Benchmarking...")
    cfg.optim.iterations = 100
    t0 = time.perf_counter()
    train.train(cfg)
    t1 = time.perf_counter()

    print(f"Total loop time for 100 steps: {t1 - t0:.3f} s")


if __name__ == "__main__":
    main()
