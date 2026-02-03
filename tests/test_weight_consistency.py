"""
Test to verify consistency between multi_determinant.py and network.py
for determinant weight combination.
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ferminet.network import ExtendedFermiNet
from ferminet.multi_determinant import MultiDeterminantOrbitals


def test_weight_combination_consistency():
    """
    Test that both multi_determinant.py and network.py use the same
    weight combination strategy (softmax normalization).
    """
    print("=" * 70)
    print("Testing Weight Combination Consistency")
    print("=" * 70)
    
    # Setup configuration
    n_electrons = 2
    n_up = 1
    n_determinants = 4
    batch_size = 3
    
    nuclei_config = {
        'positions': jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        'charges': jnp.array([1.0, 1.0])
    }
    
    network_config = {
        'single_layer_width': 8,
        'pair_layer_width': 4,
        'num_interaction_layers': 1,
        'determinant_count': n_determinants,
        'use_jastrow': False,
    }
    
    # Create both implementations
    print("\n1. Creating MultiDeterminantOrbitals...")
    multi_det = MultiDeterminantOrbitals(
        n_electrons=n_electrons,
        n_up=n_up,
        n_determinants=n_determinants,
        n_nuclei=nuclei_config['positions'].shape[0],
        config=network_config
    )
    
    print("2. Creating ExtendedFermiNet...")
    extended_net = ExtendedFermiNet(
        n_electrons, n_up, nuclei_config, network_config
    )
    
    # Set the same det_weights for both
    test_weights = jnp.array([1.0, 2.0, 0.5, 3.0])
    multi_det.params['det_weights'] = test_weights
    extended_net.params['det_weights'] = test_weights
    
    # Copy determinant parameters from multi_det to extended_net
    for i in range(n_determinants):
        det_key = f'det_{i}'
        extended_net.params[det_key] = multi_det.params[det_key]
        extended_net.orbitals.params[det_key] = multi_det.params[det_key]
    
    print(f"\n3. Testing with det_weights: {test_weights}")
    
    # Create test electron positions
    key = jax.random.PRNGKey(42)
    r_elec = jax.random.normal(key, (batch_size, n_electrons, 3)) * 0.5
    nuclei_pos = nuclei_config['positions']
    
    # Compute using MultiDeterminantOrbitals
    print("\n4. Computing with MultiDeterminantOrbitals...")
    log_psi_multi = multi_det.log_psi(r_elec, nuclei_pos)
    print(f"   log|psi| from multi_determinant.py: {log_psi_multi}")
    
    # Compute using ExtendedFermiNet
    print("\n5. Computing with ExtendedFermiNet...")
    log_psi_extended = extended_net(r_elec)
    print(f"   log|psi| from network.py: {log_psi_extended}")
    
    # Check consistency
    print("\n6. Checking consistency...")
    diff = jnp.abs(log_psi_multi - log_psi_extended)
    max_diff = jnp.max(diff)
    print(f"   Maximum absolute difference: {max_diff}")
    print(f"   Relative difference: {max_diff / jnp.abs(log_psi_multi).mean():.6e}")
    
    # Assert they are close
    atol = 1e-5
    rtol = 1e-5
    assert jnp.allclose(log_psi_multi, log_psi_extended, atol=atol, rtol=rtol), \
        f"Results differ! Multi: {log_psi_multi}, Extended: {log_psi_extended}"
    
    print("\n   ✓ Results are consistent!")
    
    # Test with different weights including negative raw weights
    # (which will still be softmax-normalized)
    print("\n7. Testing with mixed positive/negative raw weights...")
    test_weights_2 = jnp.array([2.0, -1.0, 0.0, 1.5])
    multi_det.params['det_weights'] = test_weights_2
    extended_net.params['det_weights'] = test_weights_2
    
    print(f"   Raw weights: {test_weights_2}")
    
    # After softmax, all weights should be positive
    softmax_weights = jax.nn.softmax(test_weights_2)
    print(f"   Softmax weights: {softmax_weights}")
    print(f"   Sum of softmax weights: {jnp.sum(softmax_weights)}")
    
    log_psi_multi_2 = multi_det.log_psi(r_elec, nuclei_pos)
    log_psi_extended_2 = extended_net(r_elec)
    
    print(f"   log|psi| from multi_determinant.py: {log_psi_multi_2}")
    print(f"   log|psi| from network.py: {log_psi_extended_2}")
    
    diff_2 = jnp.abs(log_psi_multi_2 - log_psi_extended_2)
    max_diff_2 = jnp.max(diff_2)
    print(f"   Maximum absolute difference: {max_diff_2}")
    
    assert jnp.allclose(log_psi_multi_2, log_psi_extended_2, atol=atol, rtol=rtol), \
        f"Results differ with mixed weights! Multi: {log_psi_multi_2}, Extended: {log_psi_extended_2}"
    
    print("\n   ✓ Results are consistent with mixed weights!")
    
    # Verify that softmax normalization is being used
    print("\n8. Verifying softmax normalization is used...")
    
    # Get weights from both implementations
    weights_multi = multi_det.get_weights()
    print(f"   Weights from MultiDeterminantOrbitals: {weights_multi}")
    print(f"   Sum: {jnp.sum(weights_multi):.6f}")
    
    # Calculate expected softmax weights
    expected_weights = jax.nn.softmax(test_weights_2)
    print(f"   Expected softmax weights: {expected_weights}")
    
    assert jnp.allclose(weights_multi, expected_weights, atol=1e-6), \
        "MultiDeterminantOrbitals weights don't match softmax!"
    
    print("\n   ✓ Softmax normalization is correctly applied!")
    
    print("\n" + "=" * 70)
    print("All consistency tests PASSED!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    try:
        success = test_weight_combination_consistency()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
