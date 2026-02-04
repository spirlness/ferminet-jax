"""
Minimal test script to debug ExtendedFermiNet NaN issues
"""

import jax
import jax.numpy as jnp
import jax.random as random
from network import ExtendedFermiNet
from configs.h2_stage2_config import get_stage2_config

print("=" * 70)
print("ExtendedFermiNet Minimal Debug Test")
print("=" * 70)

# Load configuration
print("\n1. Loading configuration...")
config = get_stage2_config('default')
n_electrons = config['n_electrons']
n_up = config['n_up']
nuclei_config = config['nuclei']

print(f"   Electrons: {n_electrons} (n_up={n_up})")
print(f"   Nuclei: {nuclei_config['positions'].shape}")
print(f"   Network config: {config['network']}")

# Create network
print("\n2. Creating ExtendedFermiNet...")
try:
    network = ExtendedFermiNet(n_electrons, n_up, nuclei_config, config['network'])
    print("   [OK] Network created")

    # Print network info
    net_info = network.get_network_info()
    print(f"   Type: {net_info['type']}")
    print(f"   Total parameters: {net_info['total_parameters']:,}")
    print(f"   Determinants: {net_info['determinant_count']}")
    print(f"   Layer width: {net_info['single_layer_width']}x{net_info['pair_layer_width']}")
    print(f"   Interaction layers: {net_info['num_interaction_layers']}")
except Exception as e:
    print(f"   [FAIL] Error creating network: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Check parameters for NaN
print("\n3. Checking network parameters for NaN...")
all_finite = True
nan_params = []
for path, param in jax.tree_util.tree_flatten_with_path(network.params)[0]:
    name = "/".join(str(p) for p in path)
    has_nan = jnp.any(jnp.isnan(param))
    has_inf = jnp.any(jnp.isinf(param))
    if has_nan or has_inf:
        all_finite = False
        nan_params.append(name)
        status = "NaN" if has_nan else "Inf"
        print(f"   [WARNING] {name} contains {status}")
    else:
        param_min = float(jnp.min(param))
        param_max = float(jnp.max(param))
        param_mean = float(jnp.mean(param))
        print(f"   {name}: min={param_min:.6f}, max={param_max:.6f}, mean={param_mean:.6f}")

if all_finite:
    print("   [OK] All parameters are finite")
else:
    print(f"   [FAIL] {len(nan_params)} parameters contain NaN/Inf")
    exit(1)

# Test forward pass
print("\n4. Testing forward pass...")
key = random.PRNGKey(42)
n_samples = 4

# Initialize electron positions near nuclei
nuclei_pos = nuclei_config['positions']
n_nuclei = nuclei_pos.shape[0]
nucleus_indices = random.randint(key, (n_samples, n_electrons), 0, n_nuclei)
key, subkey = random.split(key)
offsets = random.normal(subkey, (n_samples, n_electrons, 3)) * 0.1
r_elec = nuclei_pos[nucleus_indices] + offsets

print(f"   Input shape: {r_elec.shape}")
print(f"   Input sample: {r_elec[0, 0, :]}")

try:
    log_psi = network(r_elec)
    print(f"   Output shape: {log_psi.shape}")
    print(f"   Output values: {log_psi}")

    # Check for NaN
    has_nan = jnp.any(jnp.isnan(log_psi))
    has_inf = jnp.any(jnp.isinf(log_psi))

    if has_nan or has_inf:
        print(f"   [FAIL] Output contains {'NaN' if has_nan else 'Inf'}")
        exit(1)
    else:
        print(f"   [OK] Forward pass successful, all outputs finite")
        print(f"   Min log_psi: {float(jnp.min(log_psi)):.6f}")
        print(f"   Max log_psi: {float(jnp.max(log_psi)):.6f}")
        print(f"   Mean log_psi: {float(jnp.mean(log_psi)):.6f}")

except Exception as e:
    print(f"   [FAIL] Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test gradient computation
print("\n5. Testing gradient computation...")
try:
    def loss_fn(params, r):
        # Create log_psi function with current params
        original_params = network.params
        network.params = params
        log_psi = network(r)
        network.params = original_params

        # Simple loss: negative mean log_psi (maximize probability)
        return -jnp.mean(log_psi)

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(network.params, r_elec)

    print("   Gradient shapes:")
    all_grads_finite = True
    for path, grad in jax.tree_util.tree_flatten_with_path(grads)[0]:
        name = "/".join(str(p) for p in path)
        has_nan = jnp.any(jnp.isnan(grad))
        has_inf = jnp.any(jnp.isinf(grad))
        grad_norm = float(jnp.linalg.norm(jnp.ravel(grad)))
        status = "[FAIL]" if (has_nan or has_inf) else "[OK]"
        print(f"   {status} {name}: shape={grad.shape}, norm={grad_norm:.6f}")
        if has_nan or has_inf:
            all_grads_finite = False
            print(f"   [WARNING] Gradient for {name} contains NaN/Inf")

    if all_grads_finite:
        print("   [OK] All gradients are finite")
    else:
        print("   [FAIL] Some gradients contain NaN/Inf")
        exit(1)

except Exception as e:
    print(f"   [FAIL] Error computing gradient: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test energy calculation
print("\n6. Testing energy calculation...")
try:
    from physics import local_energy

    def log_psi_single(r_single):
        """Log psi function for single sample"""
        r_batch = r_single[None, :, :]
        log_psi = network(r_batch)
        return log_psi[0]

    # Test with first sample
    r_single = r_elec[0]
    energy = local_energy(log_psi_single, r_single, nuclei_config['positions'], nuclei_config['charges'])

    print(f"   Local energy: {float(energy):.6f} Hartree")

    if jnp.isnan(energy) or jnp.isinf(energy):
        print("   [FAIL] Energy is NaN or Inf")
        exit(1)
    else:
        print("   [OK] Energy is finite")

        # Check if energy is reasonable
        if abs(float(energy)) > 1e6:
            print(f"   [WARNING] Energy magnitude is very large: {float(energy):.6f}")
        else:
            print("   [OK] Energy magnitude is reasonable")

except Exception as e:
    print(f"   [FAIL] Error computing energy: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("All tests passed!")
print("=" * 70)
