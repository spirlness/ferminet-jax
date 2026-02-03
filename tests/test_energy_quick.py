"""
Quick test to verify energy calculation works with ExtendedFermiNet
"""

import jax
import jax.numpy as jnp
import jax.random as random
from network import ExtendedFermiNet
from physics import local_energy
from configs.h2_stage2_config import get_stage2_config

print("=" * 70)
print("Quick Energy Calculation Test")
print("=" * 70)

# Load configuration
config = get_stage2_config('default')

# Create network
n_electrons = config['n_electrons']
n_up = config['n_up']
nuclei_config = config['nuclei']

print("\nCreating network...")
network = ExtendedFermiNet(n_electrons, n_up, nuclei_config, config['network'])
print(f"Network created: {network.get_network_info()['total_parameters']} parameters")

# Initialize electron positions
key = random.PRNGKey(42)
n_samples = 4
nuclei_pos = nuclei_config['positions']
n_nuclei = nuclei_pos.shape[0]

nucleus_indices = random.randint(key, (n_samples, n_electrons), 0, n_nuclei)
key, subkey = random.split(key)
offsets = random.normal(subkey, (n_samples, n_electrons, 3)) * 0.1
r_elec = nuclei_pos[nucleus_indices] + offsets

print(f"\nTesting energy calculation for {n_samples} samples...")

# Test energy calculation for each sample
total_energy = 0.0
for i in range(n_samples):
    r_single = r_elec[i]

    def log_psi_single(r):
        r_batch = r[None, :, :]
        return network(r_batch)[0]

    energy = local_energy(log_psi_single, r_single, nuclei_config['positions'], nuclei_config['charges'])

    # Convert to scalar
    energy_scalar = float(jnp.ravel(energy)[0])
    total_energy += energy_scalar

    print(f"  Sample {i}: Energy = {energy_scalar:.6f} Hartree")

mean_energy = total_energy / n_samples
print(f"\nMean energy: {mean_energy:.6f} Hartree")
print(f"Target energy: {config['target_energy']:.6f} Hartree")
print(f"Error: {abs(mean_energy - config['target_energy']):.6f} Hartree")

print("\n" + "=" * 70)
print("Test completed!")
print("=" * 70)
