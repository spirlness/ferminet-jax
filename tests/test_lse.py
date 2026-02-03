import jax
import jax.numpy as jnp
import numpy as np
import unittest
from network import ExtendedFermiNet

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
        
        # Weights all 1.0
        self.net.params['det_weights'] = jnp.ones(self.n_dets)
        
        # Run LSE
        log_psi = self.net.multi_determinant_slater(orbitals_list)
        
        print(f"Log Psi result: {log_psi}")
        
        # Check for NaN/Inf
        self.assertFalse(jnp.any(jnp.isnan(log_psi)), "Result contains NaN")
        self.assertFalse(jnp.any(jnp.isinf(log_psi)), "Result contains Inf")
        
        # Expected value:
        # Sum = exp(0) + exp(80) + exp(-80) + exp(0)
        #     ≈ exp(80)
        # log_psi ≈ 80.0
        
        expected = 80.0
        print(f"Expected approx: {expected}")
        
        self.assertTrue(jnp.allclose(log_psi, expected, atol=1e-1), 
                        f"Expected {expected}, got {log_psi}")

    def test_mixed_signs(self):
        print("\nTesting Log-Sum-Exp with mixed signs...")
        batch_size = 2
        
        # Det 0: Positive, mag 1 (log=0)
        # Det 1: Negative, mag 1 (log=0)
        # Det 2: Negligible
        # Det 3: Negligible
        
        I = jnp.eye(self.n_electrons)
        I_neg = I.at[0].set(I[1]).at[1].set(I[0]) # Swap row 0 and 1 -> det = -1
        
        orb0 = jnp.tile(I[None, ...], (batch_size, 1, 1))
        orb1 = jnp.tile(I_neg[None, ...], (batch_size, 1, 1))
        orb2 = jnp.tile(I[None, ...], (batch_size, 1, 1)) * 1e-10
        orb3 = jnp.tile(I[None, ...], (batch_size, 1, 1)) * 1e-10
        
        orbitals_list = [orb0, orb1, orb2, orb3]
        
        # Weights set to favor cancellation
        # [1, 1, 0, 0]
        # Sum = 1*1 + 1*(-1) + ... ≈ 0
        weights = jnp.array([1.0, 1.0, 0.0, 0.0])
        self.net.params['det_weights'] = weights
        
        log_psi = self.net.multi_determinant_slater(orbitals_list)
        print(f"Log Psi (cancellation): {log_psi}")
        
        # Should not be NaN
        self.assertFalse(jnp.any(jnp.isnan(log_psi)), "Result contains NaN on cancellation")
        
        # Result should be log(abs(sum)) + max_log
        # max_log = 0 (from det 0 and 1)
        # sum = 1 - 1 + small ≈ 0
        # log(0) is -inf, but we have +1e-20 stabilizer
        # so should be around log(1e-20) ≈ -46
        
        self.assertTrue(jnp.all(log_psi < -40.0), "Result should be small (large negative log)")

    def test_weights_handling(self):
        print("\nTesting Weight handling...")
        batch_size = 1
        
        # Det 0: 1.0 (log 0)
        # Det 1: 1.0 (log 0)
        # Det 2: 1.0
        # Det 3: 1.0
        
        orb = jnp.eye(self.n_electrons)[None, ...]
        orbitals_list = [orb, orb, orb, orb]
        
        # Weights: [exp(20), 0, 0, 0]
        # Note: using raw weights as per implementation
        self.net.params['det_weights'] = jnp.array([jnp.exp(20.0), 0.0, 0.0, 0.0])
        
        # Sum = exp(20)*1 + 0 + 0 + 0 = exp(20)
        # log_psi = 20.0
        
        log_psi = self.net.multi_determinant_slater(orbitals_list)
        print(f"Log Psi (weighted): {log_psi}")
        
        self.assertTrue(jnp.allclose(log_psi, 20.0, atol=1e-4), 
                        f"Expected 20.0, got {log_psi}")

if __name__ == '__main__':
    unittest.main()
