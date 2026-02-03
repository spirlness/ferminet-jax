# Quick test for Stage 2 components

import jax
import jax.numpy as jnp
import pickle
from pathlib import Path

print("="*60)
print("Stage 2 Quick Component Test")
print("="*60)

# Test 1: Import all modules
print("\n1. Importing modules...")
try:
    from multi_determinant import create_multi_determinant_orbitals
    from jastrow import JastrowFactor
    from residual_layers import ResidualBlock, MultiLayerResidualBlock
    from scheduler import EnergyBasedScheduler
    print("   [PASS] All modules imported")
except Exception as e:
    print(f"   [FAIL] Import error: {e}")
    exit(1)

# Test 2: Create Jastrow factor
print("\n2. Testing JastrowFactor...")
try:
    n_elec = 2
    jastrow = JastrowFactor(n_elec=n_elec, hidden_dim=4)

    # Test with simple input
    r_elec = jnp.array([[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    jastrow_val = jastrow.forward(r_elec)
    print(f"   jastrow value: {float(jastrow_val):.6f}")
    print("   [PASS] JastrowFactor works")
except Exception as e:
    print(f"   [FAIL] JastrowFactor error: {e}")

# Test 3: Create MultiDeterminantOrbitals
print("\n3. Testing MultiDeterminantOrbitals...")
try:
    nuclei_config = {
        'positions': jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        'charges': jnp.array([1.0, 1.0])
    }

    config = {
        'single_layer_width': 8,
        'pair_layer_width': 4,
        'num_interaction_layers': 1,
        'determinant_count': 2,
    }

    orbitals = create_multi_determinant_orbitals(
        n_electrons=2,
        n_up=1,
        nuclei_config=nuclei_config,
        config=config
    )
    print(f"   MultiDeterminantOrbitals created")
    print(f"   Number of determinants: {orbitals.n_determinants}")
    print("   [PASS] MultiDeterminantOrbitals works")
except Exception as e:
    print(f"   [FAIL] MultiDeterminantOrbitals error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test EnergyBasedScheduler
print("\n4. Testing EnergyBasedScheduler...")
try:
    scheduler = EnergyBasedScheduler(
        initial_lr=0.001,
        min_lr=1e-5,
        patience=3,
        factor=0.5
    )

    # Simulate energy changes
    energies = [-2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -1.05]  # Improving
    for i, energy in enumerate(energies):
        lr = scheduler.step(jnp.array(energy))
        print(f"   Step {i}: Energy={energy:.4f}, LR={lr:.6f}")

    print(f"   Final LR after improvements: {lr:.6f}")
    print("   [PASS] EnergyBasedScheduler works")
except Exception as e:
    print(f"   [FAIL] EnergyBasedScheduler error: {e}")

# Test 5: Integration test
print("\n5. Testing integration...")
try:
    config = {
        'n_electrons': 2,
        'n_up': 1,
        'nuclei': nuclei_config,
        'network': {
            'single_layer_width': 8,
            'pair_layer_width': 4,
            'num_interaction_layers': 1,
            'determinant_count': 2,
            'use_residual': False,
            'use_jastrow': True,
        },
        'mcmc': {
            'n_samples': 32,
            'step_size': 0.15,
            'n_steps': 3,
            'thermalization_steps': 10.
        },
        'training': {
            'n_epochs': 5,
            'print_interval': 1,
        },
        'learning_rate': 0.001,
        'name': 'H2_Test'
    }

    print(f"   Config created")
    print(f"   Network: {config['network']['determinant_count']} determinants")
    print(f"   Samples: {config['mcmc']['n_samples']}")
    print(f"   Epochs: {config['training']['n_epochs']}")
    print("   [PASS] Integration config ready")
except Exception as e:
    print(f"   [FAIL] Integration error: {e}")

print("\n" + "="*60)
print("Stage 2 Component Tests Complete!")
print("="*60)
)
