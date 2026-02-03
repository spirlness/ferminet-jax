"""
Test to verify that the trainer correctly uses passed parameters in network_forward.
This test validates the fix for Issue 1: 训练损失未使用传入参数，导致梯度计算与优化失效
"""

import jax
import jax.numpy as jnp
import jax.random as random
import sys
import os

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from ferminet.network import ExtendedFermiNet
from ferminet.trainer import VMCTrainer
from ferminet.mcmc import FixedStepMCMC


def test_trainer_uses_passed_params():
    """
    Test that the trainer's energy_loss function responds to different parameters.
    
    The test:
    1. Creates a network and trainer
    2. Computes energy with original parameters
    3. Modifies parameters
    4. Computes energy with modified parameters
    5. Verifies that the energies are different (params matter!)
    """
    print("\n" + "=" * 60)
    print("Testing: Trainer uses passed parameters correctly")
    print("=" * 60)
    
    # H2 configuration
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
            "n_samples": 16,  # Small batch for testing
            "step_size": 0.15,
            "n_steps": 5,
        },
        "learning_rate": 0.001,
        "seed": 42,
    }
    
    key = random.PRNGKey(config["seed"])
    
    # Initialize components
    network = ExtendedFermiNet(
        config["n_electrons"],
        config["n_up"],
        config["nuclei_config"],
        config["network"],
    )
    
    mcmc = FixedStepMCMC(
        step_size=config["mcmc"]["step_size"], n_steps=config["mcmc"]["n_steps"]
    )
    
    trainer = VMCTrainer(network, mcmc, config)
    
    # Initialize electron positions
    key, init_key = random.split(key)
    n_samples = config["mcmc"]["n_samples"]
    n_electrons = config["n_electrons"]
    nuclei_pos_init = config["nuclei_config"]["positions"]
    indices = random.randint(
        init_key, (n_samples, n_electrons), 0, len(nuclei_pos_init)
    )
    r_elec = nuclei_pos_init[indices]
    key, noise_key = random.split(key)
    r_elec += random.normal(noise_key, r_elec.shape) * 0.2
    
    params_original = network.params
    nuclei_charge = config["nuclei_config"]["charges"]
    
    # Compute energy with original parameters
    loss_original, energy_original = trainer.energy_loss(
        params_original, r_elec, nuclei_pos, nuclei_charge
    )
    
    print(f"Energy with original params: {energy_original:.6f}")
    print(f"Loss with original params: {loss_original:.6f}")
    
    # Create modified parameters by scaling all weights by a factor
    import jax.tree_util as jtu
    scale_factor = 1.5
    params_modified = jtu.tree_map(
        lambda x: x * scale_factor if x.shape else x,  # Scale arrays but not scalars
        params_original
    )
    
    # Compute energy with modified parameters
    loss_modified, energy_modified = trainer.energy_loss(
        params_modified, r_elec, nuclei_pos, nuclei_charge
    )
    
    print(f"Energy with modified params: {energy_modified:.6f}")
    print(f"Loss with modified params: {loss_modified:.6f}")
    
    # Verify that parameters affect the result
    energy_diff = jnp.abs(energy_original - energy_modified)
    print(f"\nEnergy difference: {energy_diff:.6f}")
    
    # The energies should be different when params change
    assert energy_diff > 0.001, (
        f"Energy should change when parameters change! "
        f"Difference: {energy_diff:.6f} (expected > 0.001)"
    )
    
    print("\n✓ Test PASSED: Energy calculation correctly uses passed parameters")
    
    # Additional test: Verify gradients are non-zero
    print("\n" + "-" * 60)
    print("Additional check: Gradients with respect to parameters")
    print("-" * 60)
    
    grad_fn = jax.grad(lambda p: trainer.energy_loss(p, r_elec, nuclei_pos, nuclei_charge)[0])
    grads = grad_fn(params_original)
    
    # Flatten all gradients and check if any are non-zero
    grad_leaves = jtu.tree_leaves(grads)
    grad_norms = [jnp.linalg.norm(g) for g in grad_leaves if g.size > 0]
    max_grad_norm = max(grad_norms) if grad_norms else 0.0
    
    print(f"Maximum gradient norm: {max_grad_norm:.6f}")
    
    assert max_grad_norm > 1e-6, (
        f"Gradients should be non-zero! "
        f"Max gradient norm: {max_grad_norm:.6f} (expected > 1e-6)"
    )
    
    print("✓ Gradients are non-zero and correctly computed")
    
    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


def test_gradient_flow_through_params():
    """
    Test that gradients flow correctly through parameter updates.
    
    This verifies that:
    1. Computing gradients works
    2. Parameters can be updated
    3. Updated parameters affect subsequent energy calculations
    """
    print("\n" + "=" * 60)
    print("Testing: Gradient flow through parameter updates")
    print("=" * 60)
    
    # Simple H2 configuration
    nuclei_pos = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    nuclei_charges = jnp.array([1.0, 1.0])
    
    config = {
        "n_electrons": 2,
        "n_up": 1,
        "nuclei_config": {"positions": nuclei_pos, "charges": nuclei_charges},
        "network": {
            "single_layer_width": 16,
            "pair_layer_width": 8,
            "num_interaction_layers": 1,
            "determinant_count": 1,
        },
        "mcmc": {
            "n_samples": 8,
            "step_size": 0.15,
            "n_steps": 3,
        },
        "learning_rate": 0.001,
        "seed": 123,
    }
    
    key = random.PRNGKey(config["seed"])
    
    # Initialize components
    network = ExtendedFermiNet(
        config["n_electrons"],
        config["n_up"],
        config["nuclei_config"],
        config["network"],
    )
    
    mcmc = FixedStepMCMC(
        step_size=config["mcmc"]["step_size"], n_steps=config["mcmc"]["n_steps"]
    )
    
    trainer = VMCTrainer(network, mcmc, config)
    
    # Initialize electron positions
    key, init_key = random.split(key)
    n_samples = config["mcmc"]["n_samples"]
    n_electrons = config["n_electrons"]
    nuclei_pos_init = config["nuclei_config"]["positions"]
    indices = random.randint(
        init_key, (n_samples, n_electrons), 0, len(nuclei_pos_init)
    )
    r_elec = nuclei_pos_init[indices]
    key, noise_key = random.split(key)
    r_elec += random.normal(noise_key, r_elec.shape) * 0.2
    
    params = network.params
    nuclei_charge = config["nuclei_config"]["charges"]
    
    # Initial energy
    _, energy_0 = trainer.energy_loss(params, r_elec, nuclei_pos, nuclei_charge)
    print(f"Initial energy: {energy_0:.6f}")
    
    # Perform one optimization step
    (loss, energy_1), grads = trainer._loss_and_grad(
        params, r_elec, nuclei_pos, nuclei_charge
    )
    
    print(f"Loss after gradient: {loss:.6f}")
    print(f"Energy after gradient: {energy_1:.6f}")
    
    # Update parameters using Adam
    params_updated, state_updated = trainer._adam_update(
        params, grads, trainer.adam_state
    )
    
    # Energy with updated parameters
    _, energy_2 = trainer.energy_loss(
        params_updated, r_elec, nuclei_pos, nuclei_charge
    )
    
    print(f"Energy after Adam update: {energy_2:.6f}")
    
    # The parameters should be different after update
    import jax.tree_util as jtu
    param_diff = jtu.tree_map(
        lambda p1, p2: jnp.linalg.norm(p1 - p2) if p1.shape else 0.0,
        params,
        params_updated
    )
    total_param_change = sum(jtu.tree_leaves(param_diff))
    
    print(f"\nTotal parameter change: {total_param_change:.6f}")
    
    assert total_param_change > 1e-6, (
        f"Parameters should change after update! "
        f"Change: {total_param_change:.6f} (expected > 1e-6)"
    )
    
    print("✓ Parameters correctly updated by optimizer")
    
    print("\n" + "=" * 60)
    print("Gradient flow test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_trainer_uses_passed_params()
    test_gradient_flow_through_params()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
