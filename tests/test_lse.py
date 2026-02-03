import jax
import jax.numpy as jnp
import numpy as np
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ferminet.network import ExtendedFermiNet

class TestLogSumExp(unittest.TestCase):
    def setUp(self):
        # Minimal configuration to instantiate ExtendedFermiNet
        self.n_electrons = 2
        self.n_up = 1
        self.nuclei_config = {
            'positions': jnp.array([[0.0, 0.0, 0.0]])
        }
        # Updated to satisfy assertion (n_determinants must be 4-8)
        self.n_dets = 4
        self.network_config = {
            'determinant_count': self.n_dets,
            'use_jastrow': False
        }
        
        self.net = ExtendedFermiNet(
            self.n_electrons, 
            self.n_up, 
            self.nuclei_config, 
            self.network_config
        )
        
        # Initialize dummy parameters
        self.net.params['det_weights'] = jnp.ones(self.n_dets)

    def test_extreme_determinants(self):
        print("\nTesting Log-Sum-Exp with extreme values...")
        
        batch_size = 2
        
        # Target log determinants: 
        # Det 0: normal (0)
        # Det 1: very large (80)
        # Det 2: very small (-80)
        # Det 3: normal (0)
        
        c_normal = 1.0
        c_large = jnp.exp(40.0) 
        c_small = jnp.exp(-40.0)
        
        I = jnp.eye(self.n_electrons)
        
        orb0 = jnp.tile(I[None, ...], (batch_size, 1, 1)) * c_normal
        orb1 = jnp.tile(I[None, ...], (batch_size, 1, 1)) * c_large
        orb2 = jnp.tile(I[None, ...], (batch_size, 1, 1)) * c_small
        orb3 = jnp.tile(I[None, ...], (batch_size, 1, 1)) * c_normal
        
        orbitals_list = [orb0, orb1, orb2, orb3]
        
        # Weights all 1.0 (will be softmax-normalized to 0.25 each)
        self.net.params['det_weights'] = jnp.ones(self.n_dets)
        
        # Run LSE
        log_psi = self.net.multi_determinant_slater(orbitals_list)
        
        print(f"Log Psi result: {log_psi}")
        
        # Check for NaN/Inf
        self.assertFalse(jnp.any(jnp.isnan(log_psi)), "Result contains NaN")
        self.assertFalse(jnp.any(jnp.isinf(log_psi)), "Result contains Inf")
        
        # Expected value with softmax-normalized weights (each weight = 0.25):
        # log_psi = log(0.25 * exp(0) + 0.25 * exp(80) + 0.25 * exp(-80) + 0.25 * exp(0))
        #         = log(0.25 * exp(80) * (exp(-80) + 1 + exp(-160) + exp(-80)))
        #         ≈ log(0.25 * exp(80))
        #         = log(0.25) + 80
        #         ≈ -1.386 + 80 ≈ 78.614
        
        expected = jnp.log(0.25) + 80.0
        print(f"Expected approx: {expected}")
        
        self.assertTrue(jnp.allclose(log_psi, expected, atol=0.5), 
                        f"Expected {expected}, got {log_psi}")

    def test_softmax_normalization(self):
        """Test that weights are properly softmax-normalized."""
        print("\nTesting softmax normalization...")
        batch_size = 2
        
        # All determinants have log|det| = 0 (identity matrices)
        I = jnp.eye(self.n_electrons)
        orb = jnp.tile(I[None, ...], (batch_size, 1, 1))
        orbitals_list = [orb, orb, orb, orb]
        
        # Set different raw weights
        raw_weights = jnp.array([1.0, 2.0, 3.0, 4.0])
        self.net.params['det_weights'] = raw_weights
        
        # Calculate expected softmax weights
        expected_weights = jax.nn.softmax(raw_weights)
        print(f"Raw weights: {raw_weights}")
        print(f"Expected softmax weights: {expected_weights}")
        print(f"Sum of softmax weights: {jnp.sum(expected_weights)}")
        
        # With all det = 1 (log|det| = 0), we have:
        # log_psi = log(sum(w_i * 1)) = log(sum(w_i))
        # Since softmax weights sum to 1: log_psi = log(1) = 0
        log_psi = self.net.multi_determinant_slater(orbitals_list)
        
        print(f"Log Psi result: {log_psi}")
        
        # Check no NaN/Inf
        self.assertFalse(jnp.any(jnp.isnan(log_psi)), "Result contains NaN")
        self.assertFalse(jnp.any(jnp.isinf(log_psi)), "Result contains Inf")
        
        # log_psi should be 0 since sum of normalized weights times 1 = 1
        expected = 0.0
        self.assertTrue(jnp.allclose(log_psi, expected, atol=1e-4), 
                        f"Expected {expected}, got {log_psi}")

    def test_dominant_weight(self):
        """Test behavior when one weight dominates."""
        print("\nTesting dominant weight...")
        batch_size = 1
        
        # Det 0: log|det| = 10
        # Det 1: log|det| = 0  
        # Det 2: log|det| = 0
        # Det 3: log|det| = 0
        
        I = jnp.eye(self.n_electrons)
        orb0 = jnp.tile(I[None, ...], (batch_size, 1, 1)) * jnp.exp(5.0)  # det = exp(10)
        orb1 = jnp.tile(I[None, ...], (batch_size, 1, 1))
        orb2 = jnp.tile(I[None, ...], (batch_size, 1, 1))
        orb3 = jnp.tile(I[None, ...], (batch_size, 1, 1))
        
        orbitals_list = [orb0, orb1, orb2, orb3]
        
        # Make first weight much larger: [20, 0, 0, 0]
        # After softmax: [~1.0, ~0, ~0, ~0]
        raw_weights = jnp.array([20.0, 0.0, 0.0, 0.0])
        self.net.params['det_weights'] = raw_weights
        
        log_weights = jax.nn.log_softmax(raw_weights)
        print(f"Log weights after softmax: {log_weights}")
        
        # log_psi ≈ log_w_0 + 10 (since w_0 ≈ 1 and others ≈ 0)
        # log_w_0 ≈ 0 (since exp(log_w_0) ≈ 1)
        # So log_psi ≈ 10
        
        log_psi = self.net.multi_determinant_slater(orbitals_list)
        print(f"Log Psi (dominant weight): {log_psi}")
        
        expected = log_weights[0] + 10.0
        print(f"Expected: {expected}")
        
        self.assertTrue(jnp.allclose(log_psi, expected, atol=0.1), 
                        f"Expected {expected}, got {log_psi}")

if __name__ == '__main__':
    unittest.main()
