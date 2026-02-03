"""
Simple test script to verify numerical stability of ExtendedFermiNet
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import warnings
import sys
import os

# Import local modules
from ferminet.network import ExtendedFermiNet
from ferminet.physics import local_energy
from configs.h2_stage2_config import get_stage2_config

def test_extended_network_stability():
    print("=" * 70)
    print("ExtendedFermiNet Stability Test")
    print("=" * 70)

    config = get_stage2_config('default')
    network = ExtendedFermiNet(
        n_electrons=config['n_electrons'],
        n_up=config['n_up'],
        nuclei_config=config['nuclei'],
        network_config=config['network']
    )
    
    key = random.PRNGKey(42)
    params = network.params
    
    # 1. Test normal input
    print("\n1. Testing normal input...")
    x_normal = random.normal(key, (4, config['n_electrons'], 3))
    log_psi_normal = network.apply(params, x_normal)
    print(f"   log_psi finite: {jnp.all(jnp.isfinite(log_psi_normal))}")
    print(f"   log_psi mean: {jnp.mean(log_psi_normal):.6f}")

    # 2. Test large input (potential overflow)
    print("\n2. Testing large input...")
    x_large = x_normal * 100.0
    log_psi_large = network.apply(params, x_large)
    print(f"   log_psi finite: {jnp.all(jnp.isfinite(log_psi_large))}")
    
    # 3. Test electron collision (potential singularity)
    print("\n3. Testing electron collision...")
    x_collision = jnp.zeros((4, config['n_electrons'], 3))
    log_psi_collision = network.apply(params, x_collision)
    print(f"   log_psi finite: {jnp.all(jnp.isfinite(log_psi_collision))}")

    print("\nStability tests completed!")

if __name__ == "__main__":
    test_extended_network_stability()
