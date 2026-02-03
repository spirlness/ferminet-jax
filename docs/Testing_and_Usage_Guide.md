# FermiNet Testing and Usage Guide

**Comprehensive guide for testing, training, debugging, and analyzing results with FermiNet**

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Testing Guide](#2-testing-guide)
3. [Training Guide](#3-training-guide)
4. [Debugging Guide](#4-debugging-guide)
5. [Results Analysis](#5-results-analysis)
6. [Best Practices](#6-best-practices)

---

## 1. Getting Started

### Installation Instructions

#### Prerequisites

Before installing FermiNet, ensure you have the following installed:

- Python 3.8 or higher
- Git (for cloning the repository)
- CUDA-capable GPU (optional but recommended for training)

#### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ferminet.git
cd FermiNet
```

#### Step 2: Create a Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

#### Step 3: Install Dependencies

FermiNet uses JAX for automatic differentiation and GPU acceleration. Install the appropriate version:

**For CPU-only:**
```bash
pip install jax[cpu] jaxlib
```

**For GPU (CUDA):**
```bash
# Install JAX with CUDA support
pip install jax[cuda] jaxlib
```

**For GPU (ROCm - AMD):**
```bash
pip install jax[rocm] jaxlib
```

#### Step 4: Install Additional Dependencies

```bash
pip install numpy matplotlib scipy
```

#### Step 5: Verify Installation

```bash
cd demo
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
```

Expected output:
```
JAX version: 0.4.x
Devices: [cuda(id=0)]  # or [cpu(id=0)] for CPU-only
```

### Quick Start Example

#### Run a Quick Test

```bash
cd G:\FermiNet\demo
python test_stage2_quick.py
```

This will test all Stage 2 components and verify they work correctly.

#### Run Energy Calculation Test

```bash
python test_energy_quick.py
```

This tests the energy calculation functionality with ExtendedFermiNet.

### Running First Training

#### Stage 1 Training (Quick Demo)

```bash
cd G:\FermiNet\demo
python main.py
```

This runs a quick training demo with:
- Single-determinant FermiNet
- 20 training epochs
- 64 MCMC samples
- H2 molecule at equilibrium bond length

Expected training time: ~1-5 minutes (depending on hardware)

#### Stage 2 Training (Extended Architecture)

```bash
python train_stage2.py
```

This runs extended training with:
- Multi-determinant FermiNet (4 determinants)
- 200 training epochs
- 2048 MCMC samples
- Residual connections and gradient clipping

Expected training time: ~15-30 minutes (depending on hardware)

---

## 2. Testing Guide

### Available Test Scripts

The project includes several test scripts for validating different components:

| Test Script | Purpose | Duration |
|-------------|---------|----------|
| `test_stage2_quick.py` | Quick component tests | ~10 seconds |
| `test_energy_quick.py` | Energy calculation tests | ~30 seconds |
| `test_network_stability.py` | Numerical stability tests | ~1 minute |
| `test_stage2.py` | Full integration tests | ~2 minutes |
| `test_extended_debug.py` | Debugging and NaN checks | ~1 minute |

### How to Run Each Test

#### Test 1: Quick Component Tests

```bash
cd G:\F\fermiNet\demo
python test_stage2_quick.py
```

**Purpose**: Tests all Stage 2 components individually

**Expected Output**:
```
============================================================
Stage 2 Quick Component Test
============================================================

1. Importing modules...
   [PASS] All modules imported

2. Testing JastrowFactor...
   jastrow value: -0.023456
   [PASS] JastrowFactor works

3. Testing MultiDeterminantOrbitals...
   MultiDeterminantOrbitals created
   Number of determinants: 2
   [PASS] MultiDeterminantOrbitals works

4. Testing EnergyBasedScheduler...
   Step 0: Energy=-2.0000, LR=0.001000
   Step 1: Energy=-1.8000, LR=0.001000
   ...
   [PASS] EnergyBasedScheduler works

============================================================
Stage 2 Component Tests Complete!
============================================================
```

**Validation Checklist**:
- [ ] All modules import successfully
- [ ] JastrowFactor produces finite values
- [ ] MultiDeterminantOrbitals creates correct number of determinants
- [ ] EnergyBasedScheduler adjusts learning rate correctly
- [ ] No NaN or Inf values detected

---

#### Test 2: Energy Calculation Tests

```bash
python test_energy_quick.py
```

**Purpose**: Validates energy computation with ExtendedFermiNet

**Expected Output**:
```
======================================================================
Quick Energy Calculation Test
======================================================================

Creating network...
Network created: 52,000 parameters

Testing energy calculation for 4 samples...
  Sample 0: Energy = -1.123456 Hartree
  Sample 1: Energy = -1.234567 Hartree
  Sample 2: Energy = -1.345678 Hartree
  Sample 3: Energy = -1.112233 Hartree

Mean energy: -1.203983 Hartree
Target energy: -1.174000 Hartree
Error: 0.029983 Hartree

======================================================================
Test completed!
======================================================================
```

**Validation Checklist**:
- [ ] Network initializes with correct number of parameters
- [ ] Energy values are finite (no NaN/Inf)
- [ ] Mean energy is reasonable (close to target)
- [ ] Energy error is within acceptable range (< 0.1 Hartree for)

---

#### Test 3: Network Stability Tests

```bash
python test_network_stability.py
```

**Purpose**: Tests numerical stability across different configurations

**Expected Output**:
```
======================================================================
ExtendedFermiNet 数值稳定性测试
======================================================================

======================================================================
Test 1: Single Determinant Configuration
======================================================================

创建ExtendedFermiNet...

检查参数初始化...
  w_single: shape=(16,), mean=0.000123, std=0.098765, min=-0.234567, max=0.234567
  w_pair: shape=(8,), mean=0.000234, std=0.087654, min=-0.234567, max=0.234567
  ...

测试前向传播...
  [OK] 前向传播成功

测试梯度计算...
  [OK] 梯度计算成功

测试能量计算...
  [OK] 能量计算成功

[OK] Test 1 PASSED

======================================================================
测试总结
======================================================================
  Single Determinant: [PASSED]
  Multiple Determinants: [PASSED]
  Residual Connections: [PASSED]
```

**Validation Checklist**:
- [ ] All test configurations pass
- [ ] Parameters are initialized with reasonable values
- [ ] Forward pass produces finite outputs
- [ ] Gradients are finite and reasonable
- [ ] Energy calculations are stable

---

#### Test 4: Full Integration Tests

```bash
python test_stage2.py
```

**Purpose**: Tests complete training pipeline integration

**Expected Output**:
```
======================================================================
FermiNet Stage 2 扩展功能测试
======================================================================

======================================================================
测试 1: ExtendedFermiNet
======================================================================

网络信息:
  类型: ExtendedFermiNet
  总参数: 52,000
  单电子层宽度: 128
  双电子层宽度: 16
  相互作用层数: 3
  行列式数: 4

[PASS] ExtendedFermiNet测试通过!

======================================================================
测试 2: EnergyBasedScheduler
======================================================================

[PASS] EnergyBasedScheduler测试通过!

======================================================================
测试 3: ExtendedTrainer
======================================================================

[PASS] ExtendedTrainer测试通过!

======================================================================
测试 4: 完整集成测试
======================================================================

执行训练步骤...
  Step 1: 能量=-1.234567, 接受率=0.650, 梯度范数=0.123456, 学习率=0.001000
  Step[2: 能量=-1.245678, 接受率=0.680, 梯度范数=0.098765, 学习率=0.001000
  Step 3: 能量=-1.256789, 接受率=0.670, 梯度范数=0.087654, 学习率=0.001000

[PASS] 完整集成测试通过!

======================================================================
测试结果: 4 通过, 0 失败
======================================================================
```

**Validation Checklist**:
- [ ] All 4 tests pass
- [ ] Network creates successfully
- [ ] Trainer initializes correctly
- [ ] Training steps execute without errors
- [ ] Energy decreases during training

---

#### Test 5: Extended Debug Tests

```bash
python test_extended_debug.py
```

**Purpose**: Detailed debugging with NaN/Inf checks

**Expected Output**:
```
======================================================================
ExtendedFermiNet Minimal Debug Test
======================================================================

1. Loading configuration...
   Electrons: 2 (n_up=1)
   Nuclei: (2, 3)
   Network config: {...}

2. Creating ExtendedFermiNet...
   [OK] Network created
   Type: ExtendedFermiNet
   Total parameters: 52,000

3. Checking network parameters for NaN...
   w_single: min=-0.234567, max=0.234567, mean=0.000123
   w_pair: min=-0.234567, max=0.234567, mean=0.000234
   ...
   [OK] All parameters are finite

4. Testing forward pass...
   [OK] Forward pass successful, all outputs finite
   Min log_psi: -2.345678
   Max log_psi: -1.234567
   Mean log_psi: -1.789123

5. Testing gradient computation...
   [OK] All gradients are finite

6. Testing energy calculation...
   Local energy: -1.234567 Hartree
   [OK] Energy is finite
   [OK] Energy magnitude is reasonable

======================================================================
All tests passed!
======================================================================
```

**Validation Checklist**:
- [ ] No NaN or Inf in any parameter
- [ ] Forward pass produces finite outputs
- [ ] Gradients are finite
- [ ] Energy is finite and reasonable
- [ ] No numerical instabilities detected

---

### Running All Tests sequentially

```bash
cd G:\FermiNet\demo

# Run all tests in sequence
python test_stage2_quick.py && \
python test_energy_quick.py && \
python test_network_stability.py && \
python test_stage2.py && \
python test_extended_debug.py

echo "All tests completed!"
```

### Test Result Interpretation

#### Success Indicators

- All tests show `[PASS]` or `[OK]`
- No NaN or Inf values
- Energy values are reasonable (near target energy)
- Accept rates are between 0.3 and 0.9
- Gradient norms are stable (< 10.0)

#### Failure Indicators

- Any test shows `[FAIL]` or error messages
- NaN or Inf values detected
- Energy diverges (very large or infinite)
- Accept rate is too low (< 0.2) or too high (> 0.95)
- Gradient norms explode (> 100.0)

---

## 3. Training Guide

### Stage 1 Training Commands

#### Basic Training (H2 Molecule)

```bash
cd G:\FermiNet\demo
python main.py
```

**Configuration Details**:
- Network: Single-determinant, 16x4 layers
- Samples: 64
- Epochs: 20
- Learning rate: 0.001
- Target energy: -1.174 Hartree

**Expected Output**:
```
================================================================================
                    FermiNet Stage 1 - Demo Training
================================================================================
Simplified single-determinant FermiNet implementation
Target: Variational Monte Carlo training for electronic structure
================================================================================

Initializing system...
  Molecule: H2
  Electrons: 2 (up: 1, down: 1)
  Nuclei: 2 with charges [1.0, 1.0]

Initializing FermiNet network...
  Network structure:
    Single layer width: 16
    Pair layer width: 4
    Interaction layers: 1
    Determinant count: 1

Thermalizing MCMC sampler...
  Step 10/20, Avg accept rate: 0.650
  Step 20/20, Avg accept rate: 0.672
  Thermalization completed in 2.34s

Starting training...
Epoch | Energy (Ha) | Accept Rate | Error vs FHF (mHa)
------------------------------------------------------------
    10 |    -1.100000 |      0.650 |         74.000
    20 |    -1.120000 |      0.680 |         54.000
------------------------------------------------------------
Training completed in 15.67s

Final Results:
  Final Energy: -1.120000 Ha
  Target Energy: -1.174000 Ha
  Error: 54.000 mHa

Results saved to: G:\FermiNet\demo\results\H2_results_1234567890.pkl
```

#### Custom Configuration Training

Edit `G:\FermiNet\demo\configs\h2_config.py` to customize:

```python
# Example: Increase network size
H2_CONFIG = {
    'n_electrons': 2,
    'n_up': 1,
    'nuclei': {
        'positions': jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        'charges': jnp.array([1.0, 1.0])
    },
    'network': {
        'single_layer_width': 32,      # Increased from 16
        'pair_layer_width': 8,          # Increased from 4
        'num_interaction_layers': 2,     # Increased from 1
        'determinant_count': 1,
    },
    'mcmc': {
        'n_samples': 256,              # Increased from 64
        'step_size': 0.15,
        'n_steps': 10,                  # Increased from 5
        'thermalization_steps': 100,    # Increased from 20
    },
    'training': {
        'n_epochs': 100,               # Increased from 20
        'print_interval': 5,
    },
    'learning_rate': 0.001,
    'seed': 42,
    'target_energy': -1.174,
    'name': 'H2_Custom'
}
```

Then run:
```bash
python main.py
```

---

### Stage 2 Training Commands

#### Default Configuration Training

```bash
cd G:\FermiNet\demo
python train_stage2.py
```

**Configuration Details**:
- Network: 4 determinants, 128x16 layers
- Samples: 2048
- Epochs: 200
- Learning rate: 0.001 with scheduler
- Residual connections: Enabled
- Gradient clipping: 1.0

**Expected Output**:
```
======================================================================
FermiNet Stage 2 - Extended Training
======================================================================

配置信息:
  分子系统: H2_Stage2
  电子数: 2
  网络宽度: 128x16
  行列式数: 4
  相互作用层数: 3
  样本数: 2048
  训练轮数: 200
  梯度裁剪: 1.0

创建扩展FermiNet网络...
  网络类型: ExtendedFermiNet
  总参数: 52,000
  使用残差连接: True
  使用Jastrow因子: False

预热MCMC采样器...
  预热完成，当前电子位置形状: (2048, 2, 3)

计算初始能量...
  初始能量: -1.123456 Hartree
  初始标准差: 0.234567
  目标能量: -1.174000 Hartree
  能量误差: 0.050544 Hartree

======================================================================
开始训练
======================================================================

Epoch 10/200
  能量: -1.150000 Hartree
  方差: 0.012345
  标准差: 0.111111
  MCMC接受率: 0.680
  学习率: 0.001000
  梯度范数: 0.123456
  能量误差: 0.024000
  历时: 45.67秒

Epoch 20/200
  能量: -1.160000 Hartree
  方差: 0.010000
  标准差: 0.100000
  MCMC接受率: 0.690
  学习率: 0.001000
  梯度范数: 0.098765
  能量误差: 0.014000
  历时: 91.34秒

...

======================================================================
训练完成!
======================================================================

最终结果:
  最终能量: -1.170000 Hartree
  最佳能量: -1.172345 Hartree
  目标能量: -1.174000 Hartree
  能量误差: 0.004000 Hartree
  能量方差: 0.005000
  能量标准差: 0.070711
  总训练时间: 915.23秒 (15.25分钟)

训练结果已保存至: G:\FermiNet\demo\results\stage2\H2_Stage2_results.pkl
训练历史图已保存至: G:\FermiNet\demo\results\stage2\H2_Stage2_training_history.png
网络参数已保存至: G:\FermiNet\demo\results\stage2\H2_Stage2_params.pkl
```

#### Aggressive Configuration Training

Edit `train_stage2.py` to use aggressive config:

```python
# Line 191: Change from
config = get_stage2_config('default')

# To:
config config = get_stage2_config('aggressive')
```

Then run:
```bash
python train_stage2.py
```

**Aggressive Configuration**:
- 8 determinants (more expressive)
- Jastrow factor enabled (explicit correlation)
- Faster learning rate decay
- 200 epochs

**Best for**: Quick convergence testing, moderate accuracy

#### Fine Configuration Training

```python
config = get_stage2_config('fine')
```

**Fine Configuration**:
- 6 determinants
- 4096 MCMC samples (higher precision)
- 15 MCMC steps per sample
- 300 epochs
- Stricter gradient clipping (0.5)
- Smaller initial learning rate (0.0005)

**Best for**: Highest accuracy, final convergence

---

### Hyperparameter Tuning Guidelines

#### 1. Learning Rate Tuning

**Initial Learning Rate**: Start with 0.001

**Symptoms and Solutions**:

| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| Energy oscillates wildly | Learning rate too high | Reduce by factor of 2-5 |
| Energy decreases very slowly | Learning rate too low | Increase by factor of 2 |
| Energy plateaus early | Learning rate too low | Increase or use scheduler |

**Recommended values**:
- Small networks (Stage 1): 0.001 - 0.002
- Large networks (Stage 2): 0.0005 - 0.001
- Fine-tuning: 0.0001 - 0.0005

#### 2. Network Size Tuning

**Single/Pair Layer Width**:

```python
network_config = {
    'single_layer_width': 128,  # Try: 32, 64, 128, 256
    'pair_layer_width': 16,      # Try: 8, 16, 32, 64
    'num_interaction_layers': 3, # Try: 1, 2, 3, 4
    'determinant_count': 4,      # Try: 1, 2, 4, 8
}
```

**Guidelines**:
- Start small (Stage 1: 16x4, 1 det) -> test
- Increase gradually -> monitor convergence
- Stop when accuracy stops improving (diminishing returns)
- Larger networks need more samples and epochs

**Parameter Count Estimation**:
```
params ≈ n_det × [single_width × (n_elec + n_nuclei × 3)
                + pair_width × (n_elec² + n_elec × n_nuclei × 3)
                + interaction_layers × (single_width × pair_width)]
```

#### 3. MCMC Tuning

**Step Size**:

```python
mcmc_config = {
    'step_size': 0.15,  # Typical: 0.1 - 0.3
    'n_steps': 10,      # Typical: 5 - 20
    'n_samples': 2048,  # Typical: 256 - 4096
    'thermalization_steps': 100  # Typical: 50 - 200
}
```

**Accept Rate Target**: 0.5 - 0.7

**Symptoms and Solutions**:

| Symptom | Accept Rate | Solution |
|---------|-------------|----------|
| < 0.3 | Too low | Increase step_size by 1.5x |
| > 0.8 | Too high | Decrease step_size by 0.7x |

**Sample Size Guidelines**:
- Small molecules (H2, LiH): 256 - 512 samples
- Medium molecules (H2O, CH4): 1024 - 2048 samples
- Large molecules: 2048 - 4096 samples

#### 4. Training Duration Tuning

**Epoch Count**:

```python
training_config = {
    'n_epochs': 200,  # Typical: 100 - 500
}
```

**Guidelines**:
- Stage 1 (quick test): 20 - 50 epochs
- Stage 2 (normal): 200 - 300 epochs
- Stage 2 (fine): 300 - 500 epochs

**Early Stopping**:
- Stop when energy change < 0.1 mHa for 10 consecutive epochs
- Or when learning rate reaches minimum

---

### Monitoring Convergence

#### Key Metrics to Monitor

1. **Energy**: Should decrease monotonically (mostly)
2. **Energy Variance**: Should decrease to near zero
3. **MCMC Accept Rate**: Should be 0.5 - 0.7
4. **Gradient Norm**: Should be stable (< 10.0)
5. **Learning Rate**: Should decay appropriately

#### Convergence Indicators

**Good Convergence**:
```
Epoch | Energy (Ha) | Variance | Accept Rate | Grad Norm
-------------------------------------------------------
  10  |   -1.100000 |  0.100   |    0.65     |   1.00
  20  |   -1.120000 |  0.080   |    0.67     |   0.80
  50  |   -1.150000 |  0.050   |    0.68     |   0.50
 100  |   -1.165000 |  0.020   |    0.69     |   0.20
 200  |   -1.172000 |  0.005   |    0.70     |   0.10
```

**Poor Convergence**:
```
Epoch | Energy (Ha) | Variance | Accept Rate | Grad Norm
-------------------------------------------------------
  10  |   -1.100000 |  0.100   |    0.65     |   1.00
  20  |   -1.050000 |  0.150   |    0.40     |  10.00  ← Increasing energy!
  30  |   -0.950000 |  0.300   |    0.20     |  50.00  ← Diverging!
```

#### Real-time Monitoring Script

Create `monitor_training.py`:

```python
import time
import pickle
import matplotlib.pyplot as plt

def monitor_results_file(results_path):
    """Monitor training results in real-time"""
    energies = []
    timestamps = []

    print(f"Monitoring {results_path}...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            try:
                with open(results_path, 'rb') as f:
                    results = pickle.load(f)

                if 'training_history' in results:
                    history = results['training_history']
                    energies = history['energies']
                    epochs = history['epochs']

                    # Print latest status
                    if len(energies) > 0:
                        latest_energy = energies[-1]
                        latest_epoch = epochs[-1]
                        print(f"\rEpoch {latest_epoch}: E = {latest_energy:.6f} Ha", end='')

                        # Plot every 10 epochs
                        if latest_epoch % 10 == 0:
                            plt.figure(figsize=(10, 6))
                            plt.plot(epochs, energies, 'b-', linewidth=2)
                            plt.axhline(y=results['target_energy'], color='r',
                                        linestyle='--', label='Target')
                            plt.xlabel('Epoch')
                            plt.ylabel('Energy (Hartree)')
                            plt.title('Training Progress')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            plt.savefig('training_monitor.png', dpi=100)
                            plt.close()

            except (EOFError, pickle.UnpicklingError, FileNotFoundError):
                pass

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nMonitoring stopped")

if __name__ == "__main__":
    monitor_results_file("G:/FermiNet/demo/results/stage2/H2_Stage2_results.pkl")
```

Run in a separate terminal:
```bash
python monitor_training.py
```

---

## 4. Debugging Guide

### Common Error Messages

#### 1. ImportError: No module named 'jax'

**Error**:
```
ImportError: No module named 'jax'
```

**Cause**: JAX not installed

**Solution**:
```bash
pip install jax[cuda] jaxlib  # For GPU
# or
pip install jax[cpu] jaxlib  # For CPU
```

---

#### 2. CUDA Out of Memory

**Error**:
```
ResourceExhaustedError: OOM when allocating tensor
```

**Cause**: GPU memory exhausted

**Solution**:
```python
# Reduce number of samples
mcmc_config['n_samples'] = 512  # Reduce from 2048

# Reduce network size
network_config['single_layer_width'] = 64  # Reduce from 128
network_config['pair_layer_width'] = 8    # Reduce from 16

# Reduce batch size in training
# (Modify trainer.py to process in smaller batches)
```

---

#### 3. NaN Detected in Energy

**Error**:
```
WARNING: NaN detected in Local energy!
```

**Cause**: Numerical instability, usually from:
- Too large learning rate
- Poorly initialized parameters
- Exploding gradients
- Invalid wave function (e.g., zero determinant)

**Solutions**:

1. **Reduce learning rate**:
```python
config['learning_rate'] = 0.0005  # Reduce from 0.001
```

2. **Enable gradient clipping**:
```python
config['gradient_clip'] = 0.5  # Stricter clipping
```

3. **Reduce network size**:
```python
config['network']['determinant_count'] = 2  # Reduce from 4
```

4. **Add parameter regularization** (manual):
```python
# In network.py, add weight decay to gradients
grads[name] = grads[name] + 0.0001 * params[name]
```

5. **Use residual connections**:
```python
config['network']['use_residual'] = True
```

---

#### 4. Singular Matrix / Zero Determinant

**Error**:
```
WARNING: Singular matrix detected!
ValueError: Matrix is singular
```

**Cause**: Orbital matrix became singular (degenerate orbitals)

**Solutions**:

1. **Add Jastrow factor**:
```python
config['network']['use_jastrow'] = True
config['network']['jastrow_alpha'] = 0.5
```

2. **Increase numerical epsilon**:
```python
# In network.py, modify logdet calculation
logdet = jnp.linalg.slogdet(matrix + eps)
eps = 1e-6  # Increase from 1e-8
```

3. **Reinitialize network parameters**:
```python
# Change random seed
config['seed'] = 999  # Different initialization
```

---

#### 5. MCMC Accept Rate Too Low

**Error**:
```
WARNING: MCMC accept rate too low: 0.15
```

**Cause**: Step size too small or wave function not well-behaved

**Solution**:
```python
# Increase step size
config['mcmc']['step_size'] = 0.2  # Increase from 0.15

# Or reduce MCMC steps
config['mcmc']['n_steps'] = 5  # Reduce from 10
```

---

### NaN/Inf Troubleshooting

#### Step 1: Identify Where NaN Occurs

Run debug test:
```bash
python test_extended_debug.py
```

This will check:
- Parameter initialization
- Forward pass outputs
- Gradient computation
- Energy calculation

#### Step 2: Add NaN Detection to Training

Modify `train_stage2.py` to add checks:

```python
def check_nan(x, name):
    """Check for NaN/Inf and raise error if found"""
    import jax.numpy as jnp
    if jnp.any(jnp.isnan(x)):
        raise ValueError(f"NaN detected in {name}!")
    if jnp.any(jnp.isinf(x)):
        raise ValueError(f"Inf detected in {name}!")

# Add in training loop
check_nan(energy, f"Energy at epoch {epoch}")
check_nan(r_elec, f"Electron positions at epoch {epoch}")
```

#### Step 3: Check Parameter Ranges

Add to training loop:
```python
if epoch % 10 == 0:
    print("\nParameter statistics:")
    for name, param in network.params.items():
        print(f"  {name}: mean={jnp.mean(param):.6f}, "
              f"std={jnp.std(param):.6f}, "
              f"max={jnp.max(jnp.abs(param)):.6f}")
```

Expected ranges:
- Weights: mean ≈ 0, std < 1.0, max < 5.0
- Biases: mean ≈ 0, std < 1.0, max < 5.0

#### Step 4: Gradient Clipping

Ensure gradient clipping is enabled:

```python
config['gradient_clip'] = 1.0
config['gradient_clip_norm'] = 'inf'
```

Monitor gradient norms:

```python
# In training loop, add:
if 'grad_norm' in train_info:
    grad_norm = train_info['grad_norm']
    if grad_norm > 10.0:
        print(f"WARNING: Large gradient norm: {grad_norm:.3f}")
```

#### Step 5: Learning Rate Schedule

Use learning rate scheduler:

```python
config['use_scheduler'] = True
config['scheduler_patience'] = 20
config['decay_factor'] = 0.5
config['min_lr'] = 1e-5
```

---

### Performance Profiling

#### Profile Training Time

Add profiling to `train_stage2.py`:

```python
import time

# Before training loop
start_time = time.time()
epoch_times = []

# In training loop
epoch_start = time.time()

# ... training code ...

epoch_time = time.time() - epoch_start
epoch_times.append(epoch_time)

if epoch % 10 == 0:
    avg_epoch_time = sum(epoch_times[-10:]) / len(epoch_times[-10:])
    print(f"Average epoch time (last 10): {avg_epoch_time:.2f}s")

# After training loop
total_time = time.time() - start_time
time_per_epoch = total_time / n_epochs
print(f"\nTiming statistics:")
print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f}min)")
print(f"  Time per epoch: {time_per_epoch:.2f}s")
print(f"  Estimated time for {n_epochs*2} epochs: {time_per_epoch*n_epochs*2/60:.2f}min")
```

#### Profile MCMC Sampling

Add to MCMC warmup/sampling:

```python
# In mcmc.py, add timing
def sample(self, log_psi_fn, r_elec, key):
    sample_start = time.time()

    # ... MCMC code ...

    sample_time = time.time() - sample_start
    return r_elec, accept_rate, sample_time  # Return timing
```

#### Profile Memory Usage

Create `profile_memory.py`:

```python
import psutil
import time

def monitor_memory(duration=60, interval=1):
    """Monitor memory usage for given duration"""
    process = psutil.Process()
    print(f"Monitoring memory for {duration}s...")

    for i in range(duration // interval):
        mem_info = process.memory_info()
        print(f"\rTime {i*interval:3d}s | "
              f"RSS: {mem_info.rss/1e9:.2f}GB | "
              f"VMS: {mem_info.vms/1e9:.2f}GB", end='')
        time.sleep(interval)

    print()

if __name__ == "__main__":
    monitor_memory(duration=300, interval=5)
```

Run while training:
```bash
python profile_memory.py &
python train_stage2.py
```

---

### Memory Optimization Tips

#### 1. Reduce MCMC Sample Count

```python
mcmc_config['n_samples'] = 512  # Reduce from 2048
```

Trade-off: Lower precision vs. less memory

#### 2. Use Smaller Batch Sizes

Modify trainer to process in mini-batches:

```python
# In trainer.py
batch_size = 256
n_samples = r_elec.shape[0]

for batch_idx in range(0, n_samples, batch_size):
    batch_r_elec = r_elec[batch_idx:batch_idx+batch_size]
    # Process batch
```

#### 3. Clear JAX Cache

```python
import jax
jax.clear_backends()  # Clear JAX memory cache
```

#### 4. Use CPU for Large Molecules

```bash
# For very large molecules, CPU may have more memory
# Set JAX to use CPU
export JAX_PLATFORMS=cpu
python train_stage2.py
```

#### 5. Disable Gradient Accumulation

If not needed, compute gradients on-the-fly:

```python
# Instead of accumulating gradients over multiple samples
# Compute gradient per sample (slower but less memory)
for i in range(n_samples):
    grad_i = jax.grad(loss_fn)(params, r_elec[i])
    grad = {k: grad[k] + grad_i[k] for k in grad.keys()}
```

---

## 5. Results Analysis

### Loading and Visualizing Results

#### Load Training Results

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open('G:/FermiNet/demo/results/stage2/H2_Stage2_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Extract data
config = results['config']
history = results['training_history']
final_energy = results['final_energy']
best_energy = results['best_energy']
target_energy = results['target_energy']
final_variance = results['final_variance']
training_time = results['training_time']

print(f"Training Results:")
print(f"  Final energy: {final_energy:.6f} Hartree")
print(f"  Best energy: {best_energy:.6f} Hartree")
print(f"  Target energy: {target_energy:.6f} Hartree")
print(f"  Energy error: {abs(final_energy - target_energy):.6f} Hartree")
print(f"  Energy variance: {final_variance:.6f}")
print(f"  Training time: {training_time:.2f}s")
```

#### Visualize Training History

```python
def plot_training_history(history):
    """Plot comprehensive training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Energy convergence
    axes[0, 0].plot(history['epochs'], history['energies'],
                    'b-', linewidth=2, label='Energy')
    axes[0, 0].axhline(y=history['target_energy'], color='r',
                        linestyle='--', linewidth=2, label='Target')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Energy (Hartree)', fontsize=12)
    axes[0, 0].set_title('Energy Convergence', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Energy variance
    axes[0, 1].plot(history['epochs'], history['variances'],
                    'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Variance', fontsize=12)
    axes[0, 1].set_title('Energy Variance', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    # MCMC accept rate
    axes[1, 0].plot(history['epochs'], history['accept_rates'],
                    'orange', linewidth=2)
    axes[1, 0].axhline(y=0.5, color='r', linestyle='--',
                        linewidth=1, label='Target')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Accept Rate', fontsize=12)
    axes[1, 0].set_title('MCMC Accept Rate', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate schedule
    axes[1, 1].plot(history['epochs'], history['learning_rates'],
                    'purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_training_history(history)
```

#### Load Network Parameters

```python
# Load trained parameters
with open('G:/FermiNet/demo/results/stage2/H2_Stage2_params.pkl', 'rb') as f:
    params = pickle.load(f)

print(f"Loaded parameters:")
for name, param in params.items():
    print(f"  {name}: shape={param.shape}, "
          f"mean={np.mean(param):.6f}, "
          f"std={np.std(param):.6f}")

# Use for further training or evaluation
network = ExtendedFermiNet(n_electrons=2, n_up=1,
                          nuclei_config=config['nuclei'],
                          network_config=config['network'])
network.params = params
```

---

### Interpreting Training Metrics

#### Energy Metrics

**Final Energy**: The energy at the last training epoch

- **Good**: Within 1-5 mHa of target
- **Excellent**: Within 0.1-1 mHa of target
- **Poor**: > 10 mHa error

**Best Energy**: The minimum energy achieved during training

- Usually better than final energy
- Network can use best parameters after training

**Energy Variance**: Variance of local energy across samples

- **Target**: As close to zero as possible
- **Good**: < 0.01 Hartree
- **Excellent**: < 0.001 Hartree
- High variance indicates wave function not optimal

**Energy Error**: |E_train - E_target|

- **Chemical accuracy**: 1 mHa ≈ 0.001 Hartree
- **Goal**: Achieve chemical accuracy

#### MCMC Metrics

**Accept Rate**: Fraction of proposed moves accepted

- **Ideal range**: 0.5 - 0.7
- **Too low (< 0.3)**: Step size too small, poor mixing
- **Too high (> 0.8)**: Step size too large, inefficient

**Sample Quality**: Measure of how well samples represent |ψ|²

Check by:
1. Energy variance should be low
2. Accept rate should be 0.5 - 0.7
3. Energy should converge smoothly

#### Gradient Metrics

**Gradient Norm**: Magnitude of parameter gradients

- **Good**: 0.1 - 1.0
- **Too high (> 10.0)**: May cause instability
- **Too low (< 0.01)**: May indicate convergence

**Gradient Clipping**: Number of gradients clipped

- Should be low (< 5% of gradients)
- High clipping indicates instability

---

### Comparing to Reference Values

#### Reference Energies for Common Molecules

| Molecule | Reference (Hartree) | Method |
|----------|---------------------|--------|
| H        | -0.500000           | Exact |
| He       | -2.903724           | Exact |
| H2 (1.4 Å) | -1.174474     | FCI    |
| LiH      | -8.0723            | CCSD(T)|
| H2O      | -76.4389           | CCSD(T)|

#### Energy Comparison Function

```python
def compare_energy(calculated_energy, reference_energy, molecule_name):
    """Compare calculated energy to reference"""
    error = abs(calculated_energy - reference_energy)
    error_mha = error * 1000  # Convert to milli-Hartree

    print(f"\nEnergy Comparison for {molecule_name}:")
    print(f"  Calculated energy: {calculated_energy:.6f} Hartree")
    print(f"  Reference energy: {reference_energy:.6f} Hartree")
    print(f"  Absolute error: {error:.6f} Hartree ({error_mha:.3f} mHa)")

    # Assess accuracy
    if error_mha < 1.0:
        accuracy = "Chemical accuracy (excellent!)"
    elif error_mha < 10.0:
        accuracy = "Good accuracy"
    elif error_mha < 50.0:
        accuracy = "Moderate accuracy"
    else:
        accuracy = "Poor accuracy - needs improvement"

    print(f"  Accuracy: {accuracy}")

    return error_mha

# Example usage
error_mha = compare_energy(final_energy, -1.174474, "H2")
```

#### Convergence Analysis

```python
def analyze_convergence(history):
    """Analyze training convergence"""
    energies = np.array(history['energies'])
    epochs = np.array(history['epochs'])

    # Energy change per epoch
    energy_changes = np.abs(np.diff(energies))

    # Find convergence point
    converged = False
    convergence_epoch = None

    for i in range(len(energy_changes) - 10):
        # Check if energy change < 0.1 mHa for 10 consecutive epochs
        if np.all(energy_changes[i:i+10] < 0.0001):
            converged = True
            convergence_epoch = epochs[i]
            break

    print("\nConvergence Analysis:")
    print(f"  Initial energy: {energies[0]:.6f} Hartree")
    print(f"  Final energy: {energies[-1]:.6f} Hartree")
    print(f"  Total energy change: {energies[-1] - energies[0]:.6f} Hartree")

    if converged:
        print(f"  Converged at epoch: {convergence_epoch}")
        print(f"  Convergence time: {convergence_epoch} epochs")
    else:
        print(f"  Not fully converged within {epochs[-1]} epochs")

    # Energy trend
    if energy_changes[-1] < energy_changes[0]:
        print("  Trend: Decreasing (good)")
    else:
        print("  Trend: Oscillating or increasing (concern)")

analyze_convergence(history)
```

#### Multiple Run Comparison

```python
def compare_multiple_runs(run_paths):
    """Compare results from multiple training runs"""
    results_list = []

    for path in run_paths:
        with open(path, 'rb') as f:
            results = pickle.load(f)
        results_list.append(results)

    # Extract energies
    energies = [r['final_energy'] for r in results_list]
    best_energies = [r['best_energy'] for r in results_list]
    training_times = [r['training_time'] for r in results_list]

    print("\nMultiple Run Comparison:")
    print("-" * 70)
    print(f"{'Run':<10} {'Final E (Ha)':<15} {'Best E (Ha)':<15} {'Time (s)':<10}")
    print("-" * 70)

    for i, (e, be, t) in enumerate(zip(energies, best_energies, training_times)):
        print(f"{i:<10} {e:<15.6f} {be:<15.6f} {t:<10.2f}")

    print("-" * 70)

    # Statistics
    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    print(f"\nStatistics:")
    print(f"  Mean final energy: {mean_energy:.6f} Hartree")
    print(f"  Std final energy: {std_energy:.6f} Hartree")
    print(f"  Best run energy: {min(best_energies):.6f} Hartree")

# Example usage
run_paths = [
    'G:/FermiNet/demo/results/stage2/H2_Stage2_results.pkl',
    'G:/FermiNet/demo/results/stage2/H2_Stage2_run2.pkl',
    'G:/FermiNet/demo/results/stage2/H2_Stage2_run3.pkl',
]
compare_multiple_runs(run_paths)
```

---

## 6. Best Practices

### Recommended Workflow

#### 1. Initial Testing Phase

```bash
# Step 1: Run quick component tests
cd G:\FermiNet\demo
python test_stage2_quick.py

# Step 2: Run energy test
python test_energy_quick.py

# Step 3: Run stability test
python test_network_stability.py
```

**Goal**: Verify all components work before training

#### 2. Stage 1 Training (Quick Demo)

```bash
# Edit configs/h2_config.py for quick test
# Set n_epochs=20, n_samples=64

python main.py
```

**Goal**: Establish baseline, verify training loop works

#### 3. Stage 2 Training (Extended)

```bash
# Start with default config
python train_stage2.py

# Monitor convergence
# Check training_history.png
```

**Goal**: Achieve moderate accuracy (~10 mHa)

#### 4. Hyperparameter Tuning

```bash
# Try different configurations in configs/h2_stage2_config.py

# Test aggressive config (faster convergence)
# Edit train_stage2.py: config = get_stage2_config('aggressive')
python train_stage2.py

# Test fine config (highest accuracy)
# Edit train_stage2.py: config = get_stage2_config('fine')
python train_stage2.py
```

**Goal**: Find optimal hyperparameters for your molecule

#### 5. Final Training

```bash
# Use best hyperparameters
# Increase training epochs
# Use more samples

# Run multiple times with different seeds
for seed in 42 123 456 789; do
    # Edit config seed
    python train_stage2.py
    mv results/stage2/H2_Stage2_results.pkl \
       results/stage2/H2_Stage2_seed${seed}.pkl
done
```

**Goal**: Achieve chemical accuracy (1 mHa)

#### 6. Results Analysis

```python
# Analyze all runs
python analyze_results.py

# Compare to reference
python compare_to_reference.py

# Generate report
python generate_report.py
```

**Goal**: Validate results, document findings

---

### Common Pitfalls to Avoid

#### 1. Starting with Too Large a Network

**Pitfall**: Using large network for first training

**Problem**:
- Slow training
- More prone to instability
- Difficult to debug

**Solution**:
```python
# Start small
network_config = {
    'single_layer_width': 16,   # Not 128
    'pair_layer_width': 4,       # Not 16
    'determinant_count': 1,      # Not 4
}
```

---

#### 2. Insufficient MCMC Thermalization

**Pitfall**: Not thermalizing MCMC long enough

**Problem**:
- Samples not from equilibrium distribution
- Biased energy estimates
- Poor convergence

**Solution**:
```python
# Use adequate thermalization
mcmc_config['thermalization_steps'] = 100  # Not 10

# Monitor acceptance rate during thermalization
# Should stabilize before training
```

---

#### 3. Ignoring Learning Rate Schedule

**Pitfall**: Using constant learning rate

**Problem**:
- Oscillations near convergence
- Slower convergence
- May diverge

**Solution**:
```python
# Enable learning rate scheduler
config['use_scheduler'] = True
config['scheduler_patience'] = 20
config['decay_factor'] = 0.5
```

---

#### 4. Not Monitoring Gradient Norms

**Pitfall**: Not checking gradients during training

**Problem**:
- Exploding gradients go unnoticed
- Training diverges
- Unstable results

**Solution**:
```python
# Add gradient norm monitoring in training loop
if epoch % 10 == 0:
    print(f"Gradient norm: {train_info['grad_norm']:.6f}")

# Enable gradient clipping
config['gradient_clip'] = 1.0
```

---

#### 5. Training for Too Few Epochs

**Pitfall**: Stopping training too early

**Problem**:
- Suboptimal energy
- Not achieving accuracy goals
- Wasted compute time

**Solution**:
```python
# Use adequate epochs
training_config['n_epochs'] = 200  # Not 20

# Or use early stopping
# Stop when energy change < threshold for N epochs
```

---

#### 6. Only Training Once

**Pitfall**: Running single training run

**Problem**:
- Random initialization affects results
- No way to assess reliability
- May get lucky/unlucky

**Solution**:
```bash
# Run multiple times with different seeds
for seed in 42 123 456 789 999; do
    # Set config['seed'] = seed
    python train_stage2.py
done

# Compare results, use mean/standard deviation
```

---

#### 7. Not Saving Checkpoints

**Pitfall**: Only saving final results

**Problem**:
- If training crashes, lose everything
- Can't resume training
- Can't inspect intermediate states

**Solution**:
```python
# Save checkpoints periodically
if epoch % 50 == 0:
    checkpoint_path = f"results/stage2/checkpoint_epoch{epoch}.pkl"
    save_checkpoint(params, checkpoint_path)

# Resume from checkpoint
# checkpoint = load_checkpoint(checkpoint_path)
# params = checkpoint['params']
```

---

#### 8. Ignoring Accept Rate

**Pitfall**: Not monitoring MCMC accept rate

**Problem**:
- Poor mixing
- Inefficient sampling
- Biased results

**Solution**:
```python
# Monitor accept rate
if accept_rate < 0.3:
    print("WARNING: Low accept rate! Increase step_size")
elif accept_rate > 0.8:
    print("WARNING: High accept rate! Decrease step_size")

# Auto-adjust step_size
# step_size *= accept_rate / target_accept_rate
```

---

### Performance Optimization Tips

#### 1. Use GPU Acceleration

```bash
# Verify GPU is available
python -c "import jax; print(jax.devices())"

# Expected: [cuda(id=0)]

# If not, install GPU version
pip install jax[cuda] jaxlib
```

**Speedup**: 10-50x compared to CPU

---

#### 2. Enable JAX Compilation

```python
import jax

# JIT-compile critical functions
@jax.jit
def compute_energy(params, r_elec, ...):
    # Energy calculation
    return energy

@jaxOnly.jit
def mcmc_step(r_elec, key, ...):
    # MCMC step
    return r_elec_new, accept_rate
```

**Speedup**: 2-5x for compiled functions

---

#### 3. Use Vectorized Operations

```python
# BAD: Loop over samples
for i in range(n_samples):
    energy_i = compute_energy(params, r_elec[i])
    energies.append(energy_i)

# GOOD: Vectorized
energies = compute_energy_batch(params, r_elec)  # All at once
```

**Speedup**: 10-100x for batch operations

---

#### 4. Optimize MCMC Steps

```python
# Balance: MCMC steps vs. gradient updates
# Too few MCMC steps: poor sampling
# Too many MCMC steps: wasted time

# Find optimal by profiling
mcmc_config['n_steps'] = 10  # Good default

# Profile: try 5, 10, 15, 20 steps
# Measure: energy convergence per unit time
```

---

#### 5. Reduce Logging Overhead

```python
# BAD: Print every epoch
for epoch in range(n_epochs):
    # ... training ...
    print(f"Epoch {epoch}: E={energy}")  # Slows training

# GOOD: Print periodically
print_interval = 10
for epoch in range(n_epochs):
    # ... training ...
    if epoch % print_interval == 0:
        print(f"Epoch {epoch}: E={energy}")
```

**Speedup**: 5-10% for large training runs

---

#### 6. Use Efficient Data Structures

```python
# Use JAX arrays, not Python lists
import jax.numpy as jnp

# BAD: Python lists
energies = []
for e in energy_list:
    energies.append(e)

# GOOD: JAX arrays
energies = jnp.array(energy_list)

# Use in-place updates when possible
array = array.at[i].set(new_value)  # Creates new array
# Better: use array mutation when possible
```

---

#### 7. Batch Process Energy Calculations

```python
# Instead of computing energy per sample
# Compute in batches

def compute_energy_batch(network, r_elec, nuclei_pos, nuclei_charge, batch_size=256):
    """Compute energy in batches"""
    n_samples = r_elec.shape[0]
    energies = []

    for i in range(0, n_samples, batch_size):
        r_batch = r_elec[i:i+batch_size]
        energy_batch = compute_energy(network, r_batch, nuclei_pos, nuclei_charge)
        energies.append(energy_batch)

    return jnp.concatenate(energies)
```

**Speedup**: Better GPU utilization

---

#### 8. Memory Management

```python
# Clear unused arrays periodically
if epoch % 100 == 0:
    import gc
    gc.collect()

# For JAX, clear backends if memory issues
if jax.devices()[0].platform == 'gpu':
    jax.clear_backends()
```

---

### Quick Reference Checklist

#### Before Training

- [ ] All tests pass
- [ ] JAX GPU enabled (if available)
- [ ] Configuration reviewed
- [ ] Output directories exist
- [ ] Results directory writable

#### During Training

- [ ] Energy decreasing
- [ ] Energy variance decreasing
- [ ] Accept rate 0.5 - 0.7
- [ ] Gradient norms stable
- [ ] Learning rate appropriate
- [ ] No NaN/Inf values

#### After Training

- [ ] Results saved correctly
- [ ] Energy close to target
- [ ] Variance acceptably low
- [ ] Training plots generated
- [ ] Parameters saved
- [ ] Results analyzed

#### For Publication

- [ ] Multiple runs with different seeds
- [ ] Chemical accuracy achieved
- [ ] Results reproducible
- [ ] Error bars reported
- [ ] Comparison to reference values
- [ ] Methodology documented

---

## Troubleshooting Quick Reference

| Problem | Symptom | Quick Fix |
|---------|----------|------------|
| High memory usage | OOM error | Reduce n_samples, network size |
| NaN energy | "NaN detected" | Reduce learning rate, enable gradient clipping |
| Low accept rate | < 0.3 | Increase step_size |
| Energy diverging | Energy increasing | Reduce learning rate, check gradients |
| Slow convergence | Energy plateaus | Increase epochs, use scheduler |
| Poor accuracy | > 10 mHa error | Increase network size, more samples |

---

## Getting Help

1. **Check logs**: `G:\FermiNet\logs\`
2. **Review tests**: Run test scripts to diagnose
3. **Check documentation**: `G:\FermiNet\docs\`
4. **Review examples**: See `G:\FermiNet\demo\`
5. **Profile performance**: Use timing and memory profiling

---

**End of Guide**

For more information, see:
- `G:\FermiNet\FermiNet_Implementation_Report.md`
- `G:\FermiNet\demo\TRAINING_GUIDE.md`
- `G:\FermiNet\demo\STAGE2_README.md`
