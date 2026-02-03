import jax
import jax.numpy as jnp
import jax.random as random
import time
import json
import argparse
import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from ferminet.network import ExtendedFermiNet
from ferminet.trainer import VMCTrainer
from ferminet.mcmc import FixedStepMCMC


def run_baseline_benchmark(epochs=10, output_file="baseline.json"):
    nuclei_pos = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    nuclei_charges = jnp.array([1.0, 1.0])

    config = {
        "n_electrons": 2,
        "n_up": 1,
        "nuclei_config": {"positions": nuclei_pos, "charges": nuclei_charges},
        "network": {
            "single_layer_width": 32,
            "pair_layer_width": 8,
            "num_interaction_layers": 1,
            "determinant_count": 1,
        },
        "mcmc": {
            "n_samples": 256,
            "step_size": 0.15,
            "n_steps": 5,
        },
        "training": {
            "n_epochs": epochs,
        },
        "learning_rate": 0.001,
        "seed": 42,
    }

    key = random.PRNGKey(config["seed"])

    network = ExtendedFermiNet(
        config["n_electrons"], config["n_up"], config["nuclei"], config["network"]
    )

    mcmc = FixedStepMCMC(
        step_size=config["mcmc"]["step_size"], n_steps=config["mcmc"]["n_steps"]
    )

    trainer = VMCTrainer(network, mcmc, config)

    key, init_key = random.split(key)
    n_samples = config["mcmc"]["n_samples"]
    n_electrons = config["n_electrons"]
    nuclei_pos_init = config["nuclei"]["positions"]
    indices = random.randint(
        init_key, (n_samples, n_electrons), 0, len(nuclei_pos_init)
    )
    r_elec = nuclei_pos_init[indices]
    key, noise_key = random.split(key)
    r_elec += random.normal(noise_key, r_elec.shape) * 0.2

    params = network.params
    nuclei_charge = config["nuclei"]["charges"]

    timings = {
        "forward": [],
        "mcmc": [],
        "energy_grad": [],
        "adam": [],
        "train_step": [],
    }

    @jax.jit
    def forward_jit(p, r):
        orig = network.params
        network.params = p
        res = network(r)
        network.params = orig
        return res

    def log_psi_fn(r):
        orig = network.params
        network.params = params
        res = network(r)
        network.params = orig
        return res

    grad_fn = jax.jit(jax.value_and_grad(trainer.energy_loss, has_aux=True))
    adam_jit = jax.jit(trainer._adam_update)

    print(f"Starting Baseline Benchmark: {epochs} epochs (+1 warmup)")
    print(f"Device: {jax.devices()[0]}")

    for epoch in range(epochs + 1):
        is_warmup = epoch == 0
        key, step_key = random.split(key)

        t0 = time.perf_counter()
        _ = forward_jit(params, r_elec).block_until_ready()
        t_fwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        r_elec_new, _ = mcmc.sample(log_psi_fn, r_elec, step_key)
        jax.block_until_ready(r_elec_new)
        t_mcmc = time.perf_counter() - t0
        r_elec = r_elec_new

        t0 = time.perf_counter()
        (loss, mean_E), grads = grad_fn(params, r_elec, nuclei_pos, nuclei_charge)
        jax.block_until_ready(loss)
        jax.block_until_ready(grads)
        t_grad = time.perf_counter() - t0

        t0 = time.perf_counter()
        params_new, state_new = adam_jit(params, grads, trainer.adam_state)
        jax.block_until_ready(params_new)
        t_adam = time.perf_counter() - t0

        trainer.adam_state = state_new
        params = params_new

        t0 = time.perf_counter()
        res = trainer.train_step(params, r_elec, step_key, nuclei_pos, nuclei_charge)
        jax.block_until_ready(res)
        t_step = time.perf_counter() - t0

        params, _, _, r_elec = res

        if not is_warmup:
            timings["forward"].append(t_fwd)
            timings["mcmc"].append(t_mcmc)
            timings["energy_grad"].append(t_grad)
            timings["adam"].append(t_adam)
            timings["train_step"].append(t_step)
            print(f"  Epoch {epoch}/{epochs}: step={t_step:.4f}s, energy={res[1]:.4f}")
        else:
            print("  Warmup completed.")

    results = {
        "metadata": {
            "device": str(jax.devices()[0]),
            "n_samples": n_samples,
            "n_epochs": epochs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "averages": {
            "forward_ms": float(np.mean(timings["forward"]) * 1000),
            "mcmc_ms": float(np.mean(timings["mcmc"]) * 1000),
            "energy_grad_ms": float(np.mean(timings["energy_grad"]) * 1000),
            "adam_ms": float(np.mean(timings["adam"]) * 1000),
            "train_step_ms": float(np.mean(timings["train_step"]) * 1000),
        },
        "std_dev": {"train_step_ms": float(np.std(timings["train_step"]) * 1000)},
    }

    print("\n" + "=" * 40)
    print("      Benchmark Results (Average)")
    print("=" * 40)
    for k, v in results["averages"].items():
        print(f"{k.replace('_ms', ''):15}: {v:10.2f} ms")
    print("-" * 40)
    print(f"Total Epochs recorded: {epochs}")
    print("=" * 40)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FermiNet Baseline Benchmark")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", type=str, default="baseline.json")
    args = parser.parse_args()
    run_baseline_benchmark(args.epochs, args.output)
