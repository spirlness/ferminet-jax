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

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Import local modules
from ferminet.network import ExtendedFermiNet
from ferminet.physics import local_energy


def check_nan(x, name):
    """Check for NaN and print warning"""
    if isinstance(x, jnp.ndarray):
        has_nan = jnp.any(jnp.isnan(x))
        has_inf = jnp.any(jnp.isinf(x))
    else:
        has_nan = np.isnan(x)
        has_inf = np.isinf(x)

    if has_nan:
        warnings.warn(f"NaN detected in {name}!")
        print(f"  {name} contains NaN")
        return True
    if has_inf:
        warnings.warn(f"Inf detected in {name}!")
        print(f"  {name} contains Inf")
        return True
    return False


def init_electron_positions(key, n_electrons, nuclei_pos):
    """Initialize electron positions near nuclei"""
    n_samples = 4
    r_elec = jax.random.normal(key, (n_samples, n_electrons, 3)) * 0.1

    # Set initial positions near nuclei for each electron
    for i in range(n_electrons):
        r_elec = r_elec.at[:, i, :].set(r_elec[:, i, :] + nuclei_pos[i])

    return r_elec


def test_single_determinant():
    """Test single determinant configuration"""
    print("\n" + "=" * 70)
    print("Test 1: Single Determinant Configuration")
    print("=" * 70)

    # Simple config
    nuclei_config = {
        'positions': jnp.array([
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0]
        ]),
        'charges': jnp.array([1.0, 1.0])
    }

    network_config = {
        'single_layer_width': 8,
        'pair_layer_width': 4,
        'num_interaction_layers': 1,
        'determinant_count': 4,
        'use_jastrow': False,
        'use_residual': False,
        'jastrow_alpha': 0.5,
    }

    n_electrons = 2
    n_up = 1

    # Create network
    print("\nCreating ExtendedFermiNet...")
    network = ExtendedFermiNet(n_electrons, n_up, nuclei_config, network_config)

    # Check parameter initialization
    print("\nChecking parameter initialization...")
    for name, param in network.params.items():
        print(f"  {name}: shape={param.shape}, "
              f"mean={jnp.mean(param):.6f}, "
              f"std={jnp.std(param):.6f}, "
              f"min={jnp.min(param):.6f}, "
              f"max={jnp.max(param):.6f}")

        if check_nan(param, name):
            print(f"  ERROR: Parameter {name} contains NaN/Inf!")
            return False

    # Test forward pass
    print("\nTesting forward pass...")
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    r_elec = init_electron_positions(subkey, n_electrons, nuclei_config['positions'])

    print(f"  Input shape: {r_elec.shape}")
    print(f"  Input range: [{jnp.min(r_elec):.3f}, {jnp.max(r_elec):.3f}]")

    try:
        log_psi = network(r_elec)
        print(f"  Log psi shape: {log_psi.shape}")
        print(f"  Log psi values: {log_psi}")
        print(f"  Log psi range: [{jnp.min(log_psi):.3f}, {jnp.max(log_psi):.3f}]")

        if check_nan(log_psi, "Log psi"):
            print("  ERROR: Log psi contains NaN/Inf!")
            return False

        print("  [PASS] Forward pass successful")
    except Exception as e:
        print(f"  ERROR: Forward pass failed - {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test gradient computation
    print("\nTesting gradient computation...")
    try:
        def log_psi_fn(r_batch):
            return network(r_batch)

        def single_log_psi(r_single):
            return log_psi_fn(r_single[None, :, :])[0]

        grad_fn = jax.grad(single_log_psi)
        grad = grad_fn(r_elec[0])

        print(f"  Gradient shape: {grad.shape}")
        print(f"  Gradient range: [{jnp.min(grad):.3f}, {jnp.max(grad):.3f}]")

        if check_nan(grad, "Gradient"):
            print("  ERROR: Gradient contains NaN/Inf!")
            return False

        print("  [PASS] Gradient computation successful")
    except Exception as e:
        print(f"  ERROR: Gradient computation failed - {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test energy computation
    print("\nTesting energy computation...")
    try:
        def log_psi_single(r):
            return log_psi_fn(r[None, :, :])[0]

        for i in range(min(3, r_elec.shape[0])):
            e_l = local_energy(log_psi_single, r_elec[i],
                            nuclei_config['positions'], nuclei_config['charges'])
            print(f"  Sample {i}: E_L = {e_l:.6f}")

            if check_nan(e_l, f"Local energy sample {i}"):
                print(f"  ERROR: Local energy contains NaN/Inf!")
                return False

        print("  [PASS] Energy computation successful")
    except Exception as e:
        print(f"  ERROR: Energy computation failed - {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[PASS] Test 1 PASSED")
    return True


def test_multiple_determinants():
    """Test multiple determinants configuration"""
    print("\n" + "=" * 70)
    print("Test 2: Multiple Determinants Configuration")
    print("=" * 70)

    # Multi determinant config
    nuclei_config = {
        'positions': jnp.array([
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0]
        ]),
        'charges': jnp.array([1.0, 1.0])
    }

    network_config = {
        'single_layer_width': 16,
        'pair_layer_width': 8,
        'num_interaction_layers': 1,
        'determinant_count': 4,
        'use_jastrow': False,
        'use_residual': False,
        'jastrow_alpha': 0.5,
    }

    n_electrons = 2
    n_up = 1

    # Create network
    print("\nCreating ExtendedFermiNet (2 determinants)...")
    network = ExtendedFermiNet(n_electrons, n_up, nuclei_config, network_config)

    # Check parameter initialization
    print("\nChecking parameter initialization...")
    for name, param in network.params.items():
        if check_nan(param, name):
            print(f"  ERROR: Parameter {name} contains NaN/Inf!")
            return False

    # Test forward pass
    print("\nTesting forward pass...")
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    r_elec = init_electron_positions(subkey, n_electrons, nuclei_config['positions'])

    try:
        log_psi = network(r_elec)
        print(f"  Log psi shape: {log_psi.shape}")
        print(f"  Log psi values: {log_psi}")

        if check_nan(log_psi, "Log psi"):
            print("  ERROR: Log psi contains NaN/Inf!")
            return False

        print("  [PASS] Forward pass successful")
    except Exception as e:
        print(f"  ERROR: Forward pass failed - {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[PASS] Test 2 PASSED")
    return True


def test_residual_connections():
    """Test residual connections"""
    print("\n" + "=" * 70)
    print("Test 3: Residual Connections")
    print("=" * 70)

    nuclei_config = {
        'positions': jnp.array([
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0]
        ]),
        'charges': jnp.array([1.0, 1.0])
    }

    network_config = {
        'single_layer_width': 16,
        'pair_layer_width': 8,
        'num_interaction_layers': 2,
        'determinant_count': 4,
        'use_jastrow': False,
        'use_residual': True,  # Enable residual
        'jastrow_alpha': 0.5,
    }

    n_electrons = 2
    n_up = 1

    print("\nCreating ExtendedFermiNet (with residual connections)...")
    network = ExtendedFermiNet(n_electrons, n_up, nuclei_config, network_config)

    # Test forward pass
    print("\nTesting forward pass...")
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    r_elec = init_electron_positions(subkey, n_electrons, nuclei_config['positions'])

    try:
        log_psi = network(r_elec)
        print(f"  Log psi shape: {log_psi.shape}")
        print(f"  Log psi values: {log_psi}")

        if check_nan(log_psi, "Log psi"):
            print("  ERROR: Log psi contains NaN/Inf!")
            return False

        print("  [PASS] Forward pass successful")
    except Exception as e:
        print(f"  ERROR: Forward pass failed - {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[PASS] Test 3 PASSED")
    return True


def main():
    """Main test function"""
    print("=" * 70)
    print("ExtendedFermiNet Numerical Stability Tests")
    print("=" * 70)

    results = []

    # Test 1: Single Determinant
    results.append(("Single Determinant", test_single_determinant()))

    # Test 2: Multiple Determinants
    results.append(("Multiple Determinants", test_multiple_determinants()))

    # Test 3: Residual Connections
    results.append(("Residual Connections", test_residual_connections()))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed, check errors above")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
