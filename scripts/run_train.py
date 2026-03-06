import time

from ferminet import base_config, train


def main():
    cfg = base_config.default()
    cfg.system.molecule = [("He", (0.0, 0.0, 0.0))]
    cfg.system.electrons = (1, 1)
    cfg.system.charges = (2,)
    cfg.batch_size = 256
    cfg.optim.iterations = 50
    cfg.log.print_every = 1
    cfg.log.checkpoint_every = 1000
    cfg.mcmc.steps = 10

    t0 = time.time()
    train.train(cfg)
    t1 = time.time()
    print(f"Time for 50 steps: {t1 - t0:.3f} s")


if __name__ == "__main__":
    main()
