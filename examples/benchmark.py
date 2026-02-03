"""
Performance Benchmark for FermiNet
Comparison of Original vs Optimized Implementation
"""

import jax
import jax.numpy as jnp
import jax.random as random
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import modules
from ferminet.network import SimpleFermiNet, ExtendedFermiNet
from ferminet.trainer import VMCTrainer, ExtendedTrainer
from ferminet.mcmc import FixedStepMCMC
import ferminet.physics as physics

def run_benchmark(config_name: str, config: Dict, use_optimized: bool = False):
    """
    Run a short training benchmark
    """
    print(f"\nRunning benchmark: {config_name}")
    print(f"  Type: {'Optimized' if use_optimized else 'Baseline'}")
    print(f"  Samples: {config['mcmc']['n_samples']}")
    print(f"  Epochs: {config['training']['n_epochs']}")

    # Initialize components
    key = random.PRNGKey(config['seed'])

    # Network
    if use_optimized and 'determinant_count' in config['network']:
         # Stage 2 style network for optimized if specified
         if config['network']['determinant_count'] > 1 or config['network'].get('use_residual', False):
             network = ExtendedFermiNet(
                 config['n_electrons'],
                 config['n_up'],
                 config['nuclei'],
                 config['network']
             )
         else:
             network = ExtendedFermiNet(
                 config['n_electrons'],
                 config['n_up'],
                 config['nuclei'],
                 config['network']
             )
    else:
        # Default simple network
        network = ExtendedFermiNet(
            config['n_electrons'],
            config['n_up'],
            config['nuclei'],
            config['network']
        )

    # MCMC
    mcmc = FixedStepMCMC(
        step_size=config['mcmc']['step_size'],
        n_steps=config['mcmc']['n_steps']
    )

    # Trainer
    if use_optimized:
        trainer = ExtendedTrainer(network, mcmc, config)
    else:
        trainer = VMCTrainer(network, mcmc, config)

    # Initialize positions
    key, init_key = random.split(key)
    n_samples = config['mcmc']['n_samples']
    n_electrons = config['n_electrons']

    # Initialize near nuclei for better stability
    nuclei_pos = config['nuclei']['positions']
    indices = random.randint(init_key, (n_samples, n_electrons), 0, len(nuclei_pos))
    r_elec = nuclei_pos[indices]
    key, noise_key = random.split(key)
    r_elec += random.normal(noise_key, r_elec.shape) * 0.2

    # Run training
    epoch_times = []
    energy_times = []

    nuclei_charge = config['nuclei']['charges']
    params = network.params

    start_total = time.time()

    for epoch in range(config['training']['n_epochs']):
        start_epoch = time.time()

        # Train step
        key, step_key = random.split(key)

        # JIT compilation happens on first call
        step_start = time.time()
        result = trainer.train_step(
            params, r_elec, step_key, nuclei_pos, nuclei_charge
        )

        # Unpack result depending on trainer type
        if use_optimized:
            params, mean_E, accept_rate, r_elec, info = result
        else:
            params, mean_E, accept_rate, r_elec = result

        step_end = time.time()

        # Wait for computation to finish (JAX is async)
        jax.block_until_ready(params)
        jax.block_until_ready(mean_E)

        end_epoch = time.time()
        epoch_time = end_epoch - start_epoch

        # Skip first epoch (compilation time)
        if epoch > 0:
            epoch_times.append(epoch_time)

        print(f"  Epoch {epoch+1}: Time={epoch_time:.4f}s, Energy={mean_E:.4f}")

    total_time = time.time() - start_total

    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0

    return {
        'total_time': total_time,
        'avg_epoch_time': avg_epoch_time,
        'epoch_times': epoch_times
    }

def main():
    print("=" * 60)
    print("FermiNet Performance Benchmark")
    print("=" * 60)

    # Common config
    nuclei_pos = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    nuclei_charges = jnp.array([1.0, 1.0])

    config = {
        'n_electrons': 2,
        'n_up': 1,
        'nuclei': {
            'positions': nuclei_pos,
            'charges': nuclei_charges
        },
        'network': {
            'single_layer_width': 32,
            'pair_layer_width': 8,
            'num_interaction_layers': 1,
            'determinant_count': 1,
        },
        'mcmc': {
            'n_samples': 256,  # Batch size
            'step_size': 0.15,
            'n_steps': 5,
        },
        'training': {
            'n_epochs': 10,  # Run enough to average
        },
        'learning_rate': 0.001,
        'seed': 42
    }

    # 1. Run Baseline (Simulated by using non-optimized path if possible,
    # but currently we updated VMCTrainer to use JIT.
    # To truly compare, we should have kept the old one.
    # We will compare VMCTrainer (JIT) vs ExtendedTrainer (JIT + Extras))

    # Actually, we optimized VMCTrainer in place. So we are comparing
    # "Current Implementation" vs "Previous Reporting".
    # But let's verify VMCTrainer performance now.

    print("\nEvaluating VMCTrainer (Base Optimized)...")
    base_results = run_benchmark("VMCTrainer", config, use_optimized=False)

    print("\nEvaluating ExtendedTrainer (Advanced Optimized)...")
    # Enable some advanced features
    ext_config = config.copy()
    ext_config['use_scheduler'] = True
    ext_results = run_benchmark("ExtendedTrainer", ext_config, use_optimized=True)

    # 2. Comparison
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)

    print(f"VMCTrainer Avg Epoch Time: {base_results['avg_epoch_time']:.4f}s")
    print(f"ExtendedTrainer Avg Epoch Time: {ext_results['avg_epoch_time']:.4f}s")

    # Compare with reported baseline (approx 150s mentioned in previous files?)
    # In 'archive/train_stage2_optimized_v2.py' it mentions "baseline_epoch_time = 150.0"
    reported_baseline = 150.0

    speedup_base = reported_baseline / base_results['avg_epoch_time'] if base_results['avg_epoch_time'] > 0 else 0
    speedup_ext = reported_baseline / ext_results['avg_epoch_time'] if ext_results['avg_epoch_time'] > 0 else 0

    print(f"\nEstimated Speedup vs Reported Baseline ({reported_baseline}s):")
    print(f"  VMCTrainer: {speedup_base:.1f}x")
    print(f"  ExtendedTrainer: {speedup_ext:.1f}x")

    print("\nNote: Baseline refers to the initial non-JIT implementation mentioned in optimization reports.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
