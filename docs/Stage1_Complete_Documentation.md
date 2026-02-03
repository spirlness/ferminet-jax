# FermiNet Stage 1 - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Network Architecture](#network-architecture)
4. [Physics Layer](#physics-layer)
5. [MCMC Sampler](#mcmc-sampler)
6. [Trainer](#trainer)
7. [Configuration](#configuration)
8. [Training Results](#training-results)
9. [Known Issues and Solutions](#known-issues-and-solutions)
10. [Future Improvements](#future-improvements)

---

## 1. Overview

### 1.1 Objectives

FermiNet Stage 1 implements a simplified single-determinant Fermi Neural Network for solving the electronic Schrödinger equation using Variational Monte Carlo (VMC) methods. The primary objectives are:

- **Implement FermiNet Architecture**: Deep neural network that respects fermionic antisymmetry through Slater determinants
- **Variational Monte Carlo**: Optimize trial wave function to approximate ground state energy
- **Molecular Systems**: Calculate ground state energies for small molecules (H2, H, He)
- **Demonstrate Feasibility**: Establish working training pipeline with Langevin MCMC sampling

### 1.2 Technical Stack

| Component | Technology | Version/Notes |
|-----------|-----------|---------------|
| **Language** | Python 3.x | - |
| **Numerical Computing** | JAX | 0.4.x |
| **Array Operations** | JAX NumPy (jnp) | Autodiff-enabled |
| **Random Numbers** | JAX Random | Reproducible PRNGKey |
| **Optimization** | Custom Adam | Manual implementation |
| **Hardware** | CPU/GPU | JAX-compatible |

### 1.3 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Main Training Loop                   │
│                      (main.py)                           │
└────────────────────────┬────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  SimpleFermi │  │ FixedStepMCMC│  │  VMCTrainer  │
│      Net     │  │   (Sampler)  │  │ (Optimizer)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       │                 │                 │
       ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────┐
│                Physics Layer                         │
│  (kinetic_energy, local_energy, potentials)         │
└─────────────────────────────────────────────────────┘
```

**Key Components:**

1. **SimpleFermiNet** (`network.py`): Neural network architecture
   - One-body electron-nucleus features
   - Two-body electron-electron features
   - Interaction layers
   - Slater determinant output

2. **FixedStepMCMC** (`mcmc.py`): Langevin dynamics sampler
   - Drift term from wave function gradient
   - Gaussian noise term
   - Metropolis acceptance/rejection

3. **VMCTrainer** (`trainer.py`): VMC optimization
   - Energy loss computation
   - Adam optimizer
   - Parameter updates

4. **Physics Layer** (`physics.py`): Physical quantities
   - Kinetic energy via automatic differentiation
   - Coulomb potentials (softened)
   - Local energy computation

---

## 2. Network Architecture

### 2.1 SimpleFermiNet Class

The `SimpleFermiNet` class implements a single-determinant FermiNet with simplified architecture for rapid prototyping.

#### Initialization Parameters

```python
class SimpleFermiNet:
    def __init__(self, n_electrons, n_up, nuclei_config, config):
        """
        Parameters
        ----------
        n_electrons : int
            Total number of electrons
        n_up : int
            Number of spin-up electrons (determines spin configuration)
        nuclei_config : dict
            {
                'positions': jnp.ndarray  # shape: (n_nuclei, 3)
                'charges': jnp.ndarray     # shape: (n_nuclei,)
            }
        config : dict
            {
                'single_layer_width': int,      # Width of one-body features (default: 32)
                'pair_layer_width': int,        # Width of two-body features (default: 8)
                'num_interaction_layers': int,   # Number of interaction layers (default: 1)
                'determinant_count': int,        # Number of determinants (default: 1)
            }
        """
```

#### Network Parameters

The network initializes the following trainable parameters:

```python
params = {
    # One-body feature transformation
    'w_one_body': (n_nuclei, single_layer_width),
    'b_one_body': (single_layer_width,),

    # Two-body feature transformation
    'w_two_body': (n_electrons, pair_layer_width),
    'b_two_body': (pair_layer_width,),

    # Interaction layer
    'w_interaction_h': (single_layer_width, single_layer_width),
    'w_interaction_g': (pair_layer_width, pair_layer_width),
    'b_interaction_h': (single_layer_width,),
    'b_interaction_g': (pair_layer_width,),

    # Orbital output layer
    'w_orbital': (single_layer_width + pair_layer_width, n_electrons),
    'b_orbital': (n_electrons,)
}
```

### 2.2 Forward Pass Flow

The network forward pass follows this sequence:

```
Input: r_elec [batch, n_elec, 3]
  │
  ├─► one_body_features(r_elec)
  │   Output: [batch, n_elec, n_nuclei]
  │   Formula: |r_i - R_j|  (electron-nucleus distances)
  │
  ├─► Transform to h: h = tanh(one_body @ w_one_body + b_one_body)
  │   Output: [batch, n_elec, single_layer_width]
  │
  ├─► two_body_features(r_elec)
  │   Output: [batch, n_elec, n_elec]
  │   Formula: |r_i - r_j|  (electron-electron distances)
  │
  ├─► Transform to g: g = tanh(two_body * w_two_body + b_two_body)
  │   Output: [batch, n_elec, n_elec, pair_layer_width]
  │
  ├─► Interaction layers (num_interaction_layers times):
  │   h = tanh(h @ w_interaction_h + b_interaction_h)
  │   g = tanh(g @ w_interaction_g + b_interaction_g)
  │
  ├─► orbital_network(h, g):
  │   g_sum = sum(g, axis=2)  # [batch, n_elec, pair_layer_width]
  │   combined = concat([h, g_sum], axis=-1)
  │   orbitals = tanh(combined @ w_orbital + b_orbital)
  │   Output: [batch, n_elec, n_elec]
  │
  ├─► slater_determinant(orbitals):
  │   det = det(orbitals)  # [batch]
  │   log_psi = log|det|
  ▼   Output: [batch]

Return: log|ψ(r)|  [batch]
```

### 2.3 Key Methods

#### One-Body Features

```python
def one_body_features(self, r_elec, nuclei_pos):
    """
    Calculate electron-nucleus distances

    Formula: f_i^j = |r_i - R_j|

    Parameters
    ----------
    r_elec : jnp.ndarray
        Electron positions [batch, n_elec, 3]
    nuclei_pos : jnp.ndarray
        Nuclei positions [n_nuclei, 3]

    Returns
    -------
    jnp.ndarray
        One-body features [batch, n_elec, n_nuclei]
    """
    # Broadcasting for batch computation
    nuclei_pos_batch = nuclei_pos[None, None, :, :]  # [1, 1, n_nuclei, 3]
    r_elec_expanded = r_elec[:, :, None, :]          # [batch, n_elec, 1, 3]

    diff = r_elec_expanded - nuclei_pos_batch
    distances = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))

    return distances
```

#### Two-Body Features

```python
def two_body_features(self, r_elec):
    """
    Calculate electron-electron distances

    Formula: f_{ij} = |r_i - r_j|

    Parameters
    ----------
    r_elec : jnp.ndarray
        Electron positions [batch, n_elec, 3]

    Returns
    -------
    jnp.ndarray
        Two-body features [batch, n_elec, n_elec]
    """
    r_i = r_elec[:, :, None, :]  # [batch, n_elec, 1, 3]
    r_j = r_elec[:, None, :, :]  # [batch, 1, n_elec, 3]

    diff = r_i - r_j
    distances = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-10)

    return distances
```

#### Interaction Layers

```python
def interaction_layers(self, h, g):
    """
    Update one-body (h) and two-body (g) features

    h_new = tanh(h @ w_interaction_h + b_interaction_h)
    g_new = tanh(g @ w_interaction_g + b_interaction_g)

    Parameters
    ----------
    h : jnp.ndarray
        One-body features [batch, n_elec, single_layer_width]
    g : jnp.ndarray
        Two-body features [batch, n_elec, n_elec, pair_layer_width]

    Returns
    -------
    tuple
        Updated (h, g)
    """
    # Update h
    h_new = jnp.dot(h, params['w_interaction_h']) + params['b_interaction_h']
    h_new = jnp.tanh(h_new)

    # Update g (reshape for matrix multiplication)
    g_reshaped = g.reshape(-1, self.pair_layer_width)
    g_new = jnp.dot(g_reshaped, params['w_interaction_g']) + params['b_interaction_g']
    g_new = jnp.tanh(g_new)
    g_new = g_new.reshape(g.shape)

    return h_new, g_new
```

#### Slater Determinant

```python
def slater_determinant(self, orbitals):
    """
    Compute Slater determinant (ensures antisymmetry)

    Ψ(r_1, ..., r_N) = det[φ_i(r_j)]

    Parameters
    ----------
    orbitals : jnp.ndarray
        Orbital values [batch, n_elec, n_elec]

    Returns
    -------
    jnp.ndarray
        log|determinant| [batch]
    """
    determinants = jax.vmap(jnp.linalg.det)(orbitals)
    log_det = jnp.log(jnp.abs(determinants) + 1e-10)

    return log_det
```

### 2.4 Fermionic Antisymmetry

The Slater determinant ensures the wave function is antisymmetric under particle exchange, a fundamental property of fermions:

```
Ψ(r_1, r_2) = -Ψ(r_2, r_1)
```

For N electrons:

```
Ψ(r_1, r_2, ..., r_N) = det[φ_i(r_j)]

where:
- φ_i(r_j) = value of orbital i at electron j's position
- det = Σ_{π} sign(π) Π_i φ_i(r_{π(i)})
- π = permutation of {1, ..., N}
- sign(π) = ±1 depending on permutation parity
```

---

## 3. Physics Layer

### 3.1 Overview

The physics layer implements all physical quantities required for Variational Monte Carlo:

1. **Coulomb Potentials**: Nuclear-electron attraction and electron-electron repulsion
2. **Kinetic Energy**: Computed via automatic differentiation of the wave function
3. **Local Energy**: Sum of kinetic and potential energies

### 3.2 Soft-Core Coulomb Potential

To avoid singularities at zero distance, a soft-core Coulomb potential is used:

```python
def soft_coulomb_potential(r: jnp.ndarray, alpha: float = 0.1) -> jnp.ndarray:
    """
    Soft-core Coulomb potential

    V(r) = 1 / sqrt(r^2 + α^2)

    Regularized at r=0: V(0) = 1/α (finite value)
    """
    return 1.0 / jnp.sqrt(r**2 + alpha**2)
```

**Properties:**
- α = 0.1 (default softening parameter)
- Avoids division by zero at r = 0
- Approximates 1/r for r >> α

### 3.3 Nuclear-Electron Potential

```python
def nuclear_potential(r_elec, nuclei_pos, nuclei_charge):
    """
    Nuclear-electron attraction energy

    V_ne = -Σ_{i=1}^{N_e} Σ_{j=1}^{N_n} Z_j / |r_i - R_j|

    where:
    - r_i = position of electron i
    - R_j = position of nucleus j
    - Z_j = charge of nucleus j

    Returns: float (scalar energy)
    """
    # Compute distances: [n_elec, n_nuclei, 3] -> [n_elec, n_nuclei]
    diff = r_elec[:, None, :] - nuclei_pos[None, :, :]
    distances = jnp.linalg.norm(diff, axis=-1)

    # Soft potential to avoid singularity
    soft_distances = soft_coulomb_potential(distances, alpha=0.1)

    # Sum over all electron-nucleus pairs
    potential = -jnp.sum(nuclei[None, :] / soft_distances)

    return potential
```

### 3.4 Electron-Electron Potential

```python
def electronic_potential(r_elec):
    """
    Electron-electron repulsion energy

    V_ee = Σ_{i<j}^{N_e} 1 / |r_i - r_j|

    Only sums over unique pairs (upper triangular matrix)

    Returns: float (scalar energy)
    """
    n_elec = r_elec.shape[0]

    # Compute distances: [n_elec, n_elec, 3] -> [n_elec, n_elec]
    diff = r_elec[:, None, :] - r_elec[None, :, :]
    distances = jnp.linalg.norm(diff, axis=-1)

    # Upper triangular mask (exclude diagonal i=j)
    mask = jnp.triu(jnp.ones_like(distances), k=1)
    masked_distances = distances * mask

    # Soft potential and sum
    soft_distances = soft_coulomb_potential(masked_distances, alpha=0.1)
    potential = jnp.sum(1.0 / soft_distances)

    return potential
```

### 3.5 Total Potential Energy

```python
def total_potential(r_elec, nuclei_pos, nuclei_charge):
    """
    Total potential energy

    V = V_ne + V_ee

    Returns: float
    """
    v_ne = nuclear_potential(r_elec, nuclei_pos, nuclei_charge)
    v_ee = electronic_potential(r_elec)

    return v_ne + v_ee
```

### 3.6 Kinetic Energy

The kinetic energy is computed using automatic differentiation. For a trial wave function ψ(r), the local kinetic energy is:

```
T_L = -½ ∇² log|ψ(r)| - ½ |∇ log|ψ(r)||²
```

**Implementation:**

```python
def kinetic_energy(log_psi, r_r):
    """
    Calculate kinetic energy using automatic differentiation

    T = -0.5 * (|∇ log ψ|² + ∇² log ψ)

    Parameters
    ----------
    log_psi : callable
        Function returning log|ψ(r)| given positions
    r_r : jnp.ndarray
        Electron positions [n_elec, 3]

    Returns
    -------
    float
        Kinetic energy
    """
    # Calculate gradient of log_psi
    grad_fn = jax.grad(log_psi)
    grad_log_psi = grad_fn(r_r)  # [n_elec, 3]

    # Gradient squared: |∇ log ψ|² = Σ_i,j (∂ log ψ / ∂ r_ij)²
    grad_squared_sum = jnp.sum(grad_log_psi ** 2)

    # Laplacian: ∇² log ψ = Σ_i (∂²/∂r_i² log ψ)
    n_elec = r_r.shape[0]
    laplacian = jnp.array(0.0)
    for i in range(n_elec):
        for j in range(3):
            def extract_coord(r):
                return grad_fn(r)[i, j]
            second_deriv = jax.grad(extract_coord)(r_r)
            laplacian = laplacian + second_deriv

    # Kinetic energy
    t = -0.5 * (grad_squared_sum + laplacian)

    return t
```

**Derivation:**

Starting from the Schrödinger equation: Hψ = Eψ

For local energy E_L = Hψ/ψ:

```
E_L = T_L + V_L

T_L = (-½ ∇² ψ) / ψ

Using ψ = exp(log ψ):

∇ ψ = ψ ∇ log ψ
∇² ψ = ∇ · (ψ ∇ log ψ)
      = ψ (∇ log ψ)² + ψ ∇² log ψ

T_L = -½ [(∇ log ψ)² + ∇² log ψ]
```

### 3.7 Local Energy

The local energy is the core quantity for VMC optimization:

```python
def local_energy(log_psi, r_r, nuclei_pos, nuclei_charge):
    """
    Calculate local energy E_L = T_L + V_L

    E_L = (-½ ∇² + V) ψ / ψ
       = -½ ∇² log|ψ| + V

    This is the Hamiltonian acting on ψ, divided by ψ

    Returns: float
    """
    # Kinetic energy
    t = kinetic_energy(log_psi, r_r)

    # Potential energy
    v = total_potential(r_r, nuclei_pos, nuclei_charge)

    # Local energy
    e_l = t + v

    return e_l
```

**Physical Interpretation:**

- For an exact eigenstate: E_L is constant (equal to eigenvalue)
- For a trial wave function: E_L varies with electron positions
- VMC optimizes ψ to minimize variance of E_L (makes it closer to eigenstate)

---

## 4. MCMC Sampler

### 4.1 Overview

The `FixedStepMCMC` class implements Langevin dynamics for sampling electron configurations from |ψ(r)|² distribution. This is essential for computing statistical averages in VMC.

### 4.2 Langevin Dynamics

Langevin dynamics combines:

1. **Drift term**: Follows gradient of probability density
2. **Diffusion term**: Random Gaussian noise

**Update Equation:**

```
r' = r + 0.5 * ∇ log|ψ(r)| * Δt + N(0, Δt)

where:
- r' = proposed position
- r = current position
- ∇ log|ψ(r)| = gradient of log wave function
- Δt = time step (step_size)
- N(0, Δt) = Gaussian noise with variance Δt
```

**Derivation:**

For sampling from P(r) = |ψ(r)|² = exp(2 log|ψ(r)|):

The Fokker-Planck equation for Langevin dynamics:

```
∂P/∂t = -∇ · [P(r) * drift(r)] + 0.5 Δt ∇² P(r)

Stationary solution P(r) when:
drift(r) = 0.5 Δt ∇ log P(r) = 0.5 Δt * 2 ∇ log|ψ(r)|
        = Δt ∇ log|ψ(r)|

Our implementation uses 0.5 factor for stability:
drift(r) = 0.5 Δt ∇ log|ψ(r)|
```

### 4.3 Metropolis Acceptance

After Langevin proposal, apply Metropolis-Hastings acceptance:

```python
def _metropolis_accept(self, log_psi_current, log_psi_proposed, key):
    """
    Metropolis acceptance probability

    accept_prob = min(1, |ψ(r')|² / |ψ(r)|²)
               = min(1, exp[2 * (log|ψ(r')| - log|ψ(r)|)])

    Returns: boolean mask [batch]
    """
    # Log acceptance ratio
    log_accept_ratio = 2.0 * (log_psi_proposed - log_psi_current)

    # Clip to avoid overflow
    log_accept_ratio = jnp.clip(log_accept_ratio, -100, 100)

    # Generate uniform random numbers
    u = random.uniform(key, shape=log_psi_current.shape)

    # Accept condition
    accepted = u < jnp.exp(log_accept_ratio)

    return accepted
```

**Detailed Balance:**

Metropolis algorithm satisfies detailed balance:

```
P(r) T(r → r') = P(r') T(r' → r)

P(r')/P(r) = |ψ(r')|²/|ψ(r)|²

T(r → r') = min(1, P(r')/P(r))
```

### 4.4 Complete Sampling Algorithm

```python
class FixedStepMCMC:
    def __init__(self, step_size=0.15, n_steps=10):
        """
        Parameters
        ----------
        step_size : float
            Langevin time step Δt (typically 0.1-0.2)
        n_steps : int
            Number of Langevin steps per sample() call
        """

    def sample(self, log_psi_fn, r_elec, key):
        """
        Perform MCMC sampling

        Parameters
        ----------
        log_psi_fn : callable
            Function returning log|ψ(r)| given positions
        r_elec : jnp.ndarray
            Current positions [batch, n_elec, 3]
        key : jnp.ndarray
            JAX random key

        Returns
        -------
        tuple
            (r_new, accept_rate)
        """
        r_current = r_elec
        log_psi_val = log_psi_fn(r_elec)
        accepted = 0.0
        total = 0.0

        # Multiple Langevin steps
        for step in range(self.n_steps):
            key, subkey = random.split(key)
            r_proposed, log_psi_proposed, mask = self._langevin_step(
                r_current, log_psi_val, grad_log_psi_fn, log_psi_fn, subkey
            )

            # Update statistics
            accepted = accepted + mask.sum()
            total = total + mask.size

            # Accept/reject
            r_current = r_proposed
            log_psi_val = log_psi_proposed

        accept_rate = accepted / total
        return r_current, accept_rate
```

### 4.5 Tuning Guidelines

**Step Size (Δt):**

- Too small: Slow exploration, high acceptance rate
- Too large: Large jumps, low acceptance rate
- Optimal: ~50-60% acceptance rate
- Typical values: 0.1 - 0.2 for atomic systems

**Number of Steps:**

- Balance between decorrelation and computational cost
- More steps = better mixing = higher computational cost
- Typical: 5-10 steps per training iteration

**Thermalization:**

- Initial steps to reach equilibrium distribution
- Discard before starting training
- Typical: 20-100 steps for small systems

---

## 5. Trainer

### 5.1 Overview

The `VMCTrainer` class implements the Variational Monte Carlo training loop with:

1. **Energy loss computation**
2. **Gradient calculation via automatic differentiation**
3. **Adam optimizer**
4. **Parameter updates (training step)**

### 5.2 Energy Loss Function

The training objective is to minimize the variance of the local energy:

```python
def energy_loss(self, params, r_elec, nuclei_pos, nuclei_charge):
    """
    Energy loss: L = E_L - <E_L>

    Uses control variate technique:
    Loss = E_L - <E_L>

    Expected value: <E_L - <E_L>> = 0
    Optimization goal: minimize variance

    Returns
    -------
    tuple
        (loss, mean_energy)
    """
    # Create wave function with current parameters
    log_psi_fn = self._make_log_psi_fn(params)

    # Compute local energies for all samples
    local_E = self._compute_local_energy_batch(
        log_psi_fn, r_elec, nuclei_pos, nuclei_charge
    )  # [n_samples]

    # Handle numerical issues
    local_E = jnp.where(jnp.isnan(local_E), mean
```

**Rationale:**

```
Variational Principle: <ψ|H|ψ> / <ψ|ψ> ≥ E_ground

For trial wave function ψ_θ with parameters θ:

E(θ) = <E_L>_ψ = ∫ |ψ_θ|² E_L dr / ∫ |ψ_θ|² dr

Optimization strategies:
1. Minimize <E_L> directly (energy minimization)
2. Minimize Var(E_L) (variance minimization)

Our implementation uses:
Loss = E_L - <E_L>

This minimizes variance because:
Var(E_L) = <(E_L - <E_L>)²> = <Loss²>

Variance → 0 when ψ is exact eigenstate
```

### 5.3 Batch Local Energy Computation

```python
def _compute_local_energy_batch(self, log_psi_fn, r_elec,
                                nuclei_pos, nuclei_charge):
    """
    Compute local energy for batch of configurations

    Uses vectorization for efficiency
    """
    from physics import local_energy

    def compute_one(r_single):
        """Compute local energy for single configuration"""
        def log_psi_single(r):
            r_batch = r[None, :, :]
            return log_psi_fn(r_batch)[0]

        return local_energy(log_psi_single, r_single,
                          nuclei_pos, nuclei_charge)

    # Vectorize over batch
    compute_vmap = jax.vmap(compute_one)
    local_energies = compute_vmap(r_elec)

    return local_energies
```

### 5.4 Adam Optimizer

Adam (Adaptive Moment Estimation) combines:

1. **Momentum**: Exponential moving average of gradients
2. **Adaptive learning rates**: Per-parameter scaling based on gradient variance

**Algorithm:**

```python
def _adam_update(self, params, grads, state):
    """
    Adam optimization step

    Update rules:
    m_t = β₁ * m_{t-1} + (1 - β₁) * g_t       (first moment)
    v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²      (second moment)
    m̂_t = m_t / (1 - β₁^t)                     (bias correction)
    v̂_t = v_t / (1 - β₂^t)                     (bias correction)
    θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)

    where:
    - g_t = gradient at step t
    - m_t = first moment estimate
    - v_t = second moment estimate
    - α = learning rate
    - β₁, β₂ = exponential decay rates
    - ε = numerical stability constant
    """
    m = state['m']
    v = state['v']
    t = state['t'] + 1

    params_new = {}
    m_new = {}
    v_new = {}

    for key in params.keys():
        # Update first moment
        m_new[key] = self.beta1 * m[key] + (1 - self.beta1) * grads
```

**Default Hyperparameters:**

```python
learning_rate = 0.001    # α
beta1 = 0.9              # β₁ (momentum)
beta2 = 0.999            # β₂ (adaptive rate)
epsilon = 1e-8           # ε (stability)
```

### 5.5 Training Step

```python
def train_step(self, params, r_elec, key, nuclei_pos, nuclei_charge):
    """
    Execute single training step

    Steps:
    1. MCMC sample new electron configurations
    2. Compute energy loss and gradients
    3. Update parameters with Adam

    Returns
    -------
    tuple
        (params_new, mean_energy, accept_rate, r_elec_new)
    """
    # 1. MCMC sampling
    log_psi_fn = self._make_log_psi_fn(params)
    r_elec_new, accept_rate = self.mcmc.sample(log_psi_fn, r_elec, key)

    # 2. Compute loss and gradients
    grad_fn = jax.value_and_grad(self.energy_loss, has_aux=True)
    (loss, mean_E), grads = grad_fn(params, r_elec_new, nuclei_pos,
                                   nuclei_charge)

    # 3. Adam update
    params_new, self.adam_state = self._adam_update(params, grads,
                                                   self.adam_state)

    return params_new, mean_E, accept_rate, r_elec_new
```

### 5.6 Numerical Stability

Several mechanisms ensure numerical stability:

```python
# 1. NaN handling in local energies
nan_mask = jnp.isnan(local_E)
if jnp.any(nan_mask):
    valid_mask = ~nan_mask
    mean_valid = jnp.mean(local_E[valid_mask])
    local_E = jnp.where(nan_mask, mean_valid, local_E)

# 2. Inf clipping
inf_mask = jnp.isinf(local_E)
local_E = jnp.clip(local_E, -1e6, 1e6)

# 3. Loss stability
loss = jnp.where(jnp.isnan(loss), 0.0, loss)
loss = jnp.where(jnp.isinf(loss), 1e6, loss)
```

---

## 6. Configuration

### 6.1 Complete H2 Configuration

```python
H2_CONFIG = {
    # ========== Molecular System ==========
    'n_electrons': 2,              # Total number of electrons
    'n_up': 1,                     # Spin-up electrons (1 up, 1 down)
    'nuclei': {
        'positions': jnp.array([
            [0.0, 0.0, 0.0],      # H nucleus 1 at origin
            [1.4, 0.0, 0.0]       # H nucleus 2 at x=1.4 Bohr
        ]),
        'charges': jnp.array([1.0, 1.0])  # Z = +1 each
    },

    # ========== Network Architecture ==========
    'network': {
        'single_layer_width': 16,     # One-body feature width
        'pair_layer_width': 4,        # Two-body feature width
        'num_interaction_layers': 1,  # Interaction layers
        'determinant_count': 1,       # Single determinant
    },

    # ========== MCMC Sampling ==========
    'mcmc': {
        'n_samples': 64,             # Batch size
        'step_size': 0.15,           # Langevin Δt
        'n_steps': 5,                # Steps per sample
        'thermalization_steps': 20,  # Warmup steps
    },

    # ========== Training ==========
    'training': {
        'n_epochs': 20,              # Training epochs
        'print_interval': 2,         # Progress print frequency
    },

    # ========== Optimizer ==========
    'learning_rate': 0.001,         # Adam learning rate

```

### 6.2 Parameter Descriptions

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| **n_electrons** | Total electrons in system | 1-10 for small systems |
| **n_up** | Spin-up electrons | n_electrons // 2 (closed shell) |
| **nuclei.positions** | Nuclear coordinates (Bohr) | - |
| **nuclei.charges** | Nuclear charges Z | Positive integers |
| **single_layer_width** | One-body network width | 16-128 |
| **pair_layer_width** | Two-body network width | 4-32 |
| **num_interaction_layers** | FermiNet depth | 1-4 |
| **determinant_count** | Number of determinants | 1-8 |
| **n_samples** | MCMC batch size | 64-1000 |
| **step_size** | Langevin Δt | 0.1-0.2 |
| **n_steps** | MCMC steps per training step | 5-20 |
| **thermalization_steps** | MCMC warmup | 20-200 |
| **n_epochs** | Training iterations | 20-1000 |
| **learning_rate** | Adam α | 0.0001-0.01 |
| **beta1** | Adam momentum decay | 0.9 |
| **beta2** | Adam adaptive decay | 0.999 |

### 6.3 System-Specific Configurations

#### Hydrogen Atom (H)

```python
H_CONFIG = {
    'n_electrons': 1,
    'n_up': 1,
    'nuclei': {
        'positions': [[0.0, 0.0, 0.0]],
        'charges': [1.0]
    },
    'network': {
        'single_layer_width': 32,
        'pair_layer_width': 8,
        'num_interaction_layers': 1,
        'determinant_count': 1,
    },
    'mcmc': {
        'n_samples': 256,
        'step_size': 0.15,
        'n_steps': 10,
        'thermalization_steps': 100,
    },
    'training': {
        'n_epochs': 500,
        'print_interval': 10,
    },
    'learning_rate': 0.001,
    'target_energy': -0.5,  # Exact ground state
}
```

#### Helium Atom (He)

```python
HE_CONFIG = {
    'n_electrons': 2,
    'n_up': 1,
    'nuclei': {
        'positions': [[0.0, 0.0, 0.0]],
        'charges': [2.0]
    },
    'network': {
        'single_layer_width': 32,
        'pair_layer_width': 8,
        'num_interaction_layers': 1,
        'determinant_count': 1,
    },
    'mcmc': {
        'n_samples': 256,
        'step_size': 0.15,
        'n_steps': 10,
        'thermalization_steps': 100,
    },
    'training': {
        'n_epochs': 500,
        'print_interval': 10,
    },
    'learning_rate': 0.001,
    'target_energy': -2.903,  # FCI ground state
}
```

### 6.4 Configuration Tuning Tips

**For Faster Training:**
- Reduce `n_samples` (64-128)
- Reduce `n_steps` (5-10)
- Reduce `thermalization_steps` (20-50)
- Reduce network width (16-32)

**For Higher Accuracy:**
- Increase `n_samples` (256-1000)
- Increase `n_steps` (10-20)
- Increase `thermalization_steps` (100-200)
- Increase network width (64-128)
- Train for more epochs

**For Stability:**
- Use smaller learning rate (0.0001-0.001)
- Ensure MCMC acceptance rate ~50-60%
- Add gradient clipping if needed

---

## 7. Training Results

### 7.1 H2 Molecule Training

**Configuration:**
- 2 electrons (1 up, 1 down)
- 2 hydrogen nuclei at (0,0,0) and (1.4,0,0) Bohr
- Single determinant network (16×4 width, 1 interaction layer)
- 64 MCMC samples, 5 Langevin steps per training step
- 20 training epochs
- Adam optimizer (lr=0.001, β₁=0.9, β₂=0.999)

**Results:**
```
Final Energy: NaN
Target Energy: -1.174 Hartree
Error: NaN (mHa)
Average Accept Rate: 0.015 (1.5%)
Training Time: 74.8 seconds
```

### 7.2 Issue Analysis

**Problem: Energy converges to NaN**

**Root Causes:**

1. **Numerical Instability in Determinant Calculation:**
   - Slater determinant can become very small (near zero)
   - log(det) can diverge to -∞
   - Gradients explode

2. **Low MCMC Acceptance Rate (1.5%):**
   - Expected: 50-60%
   - Too low: sampling inefficient
   - Wave function changes rapidly, sampling can't keep up

3. **Small Network Capacity:**
   - 16×4 network may be insufficient for H2
   - Cannot represent accurate wave function

4. **Initialization Issues:**
   - Random initialization may put network in poor region
   - Gradients can be extremely large initially

### 7.3 Expected vs. Actual Performance

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Final Energy | -1.174 Ha | NaN | ⚠️ Failed |
| Energy Error | < 1 mHa | NaN | ⚠️ Failed |
| Accept Rate | 50-60% | 1.5% | ⚠️ Failed |
| Convergence | Stable | Divergent | ⚠️ Failed |
| Training Time | ~1-2 min | 75 sec | ✓ Reasonable |

### 7.4 Comparison with Reference

**FCI (Full Configuration Interaction) for H2 at R=1.4 Bohr:**
```
E_FCI = -1.174 Hartree
```

**Hartree-Fock (single determinant) for H2:**
```
E_HF ≈ -1.13 to -1.16 Hartree (depending on R)
```

**Our Implementation:**
```
E_FermiNet = NaN (diverged)
```

### 7.5 Successful Training Indicators

For a successful training run, you should observe:

1. **Energy Convergence:**
   ```
   Epoch 1:  Energy = -0.8 Ha, Variance = 0.5
   Epoch 10: Energy = -1.1 Ha, Variance = 0.1
   Epoch 20: Energy = -1.17 Ha, Variance = 0.01
   ```

2. **Acceptance Rate:**
   ```
   Accept Rate: 0.55 ± 0.10
   ```

3. **Gradient Norms:**
   ```
   |∇L|: Decreasing over training
   |∇L|_final < 0.01
   ```

4. **Variance Reduction:**
   ```
   Var(E_L): Decreasing
   Var(E_L)_final ≈ 0.001
   ```

---

## 8. Known Issues and Solutions

### 8.1 Numerical Instability in Determinant

**Issue:**
```python
determinants = jnp.linalg.det(orbitals)
log_det = jnp.log(jnp.abs(determinants) + 1e-10)  # Can still be -∞
```

**Problems:**
- Determinant can be exactly zero (orthogonal orbitals)
- log(0) = -∞ → NaN in gradients
- Gradients explode → training diverges

**Solutions:**

1. **Add log-sum-exp stabilization:**
```python
# Before determinant, subtract mean from each row
orbitals_centered = orbitals - jnp.mean(orbitals, axis=-1, keepdims=True)
determinants = jnp.linalg.det(orbitals_centered)

# Log with clipping
log_det = jnp.log(jnp.abs(determinants) + 1e-8)
log_det = jnp.clip(log_det, -100, 100)  # Prevent explosion
```

2. **Use symmetric regularization:**
```python
# Add small multiple of identity to matrix
epsilon = 1e-6
orbitals_reg = orbitals + epsilon * jnp.eye(orbitals.shape[-1])
```

3. **Log-determinant via LU decomposition:**
```python
# More stable than direct det()
def log_det_stable(M):
    sign, logabsdet = jnp.linalg.slogdet(M)
    return sign * logabsdet
```

### 8.2 Low MCMC Acceptance Rate

**Issue:** Acceptance rate ~1.5% instead of 50-60%

**Causes:**
1. Wave function changes too rapidly between steps
2. Step size too large for current optimization stage
3. Gradients causing large proposed moves

**Solutions:**

1. **Adaptive step size:**
```python
def adaptive_step_size(accept_rate, current_step_size):
    target = 0.55
    if accept_rate < 0.4:
        return current_step_size * 0.9  # Reduce
    elif accept_rate > 0.7:
        return current_step_size * 1.1  # Increase
    return current_step_size
```

2. **Gradient clipping in proposal:**
```python
drift = 0.5 * grad_log_psi * self.step_size
drift_norm = jnp.linalg.norm(drift)
if drift_norm > 1.0:
    drift = drift / drift_norm  # Normalize
```

3. **Separate optimization and sampling:**
```python
# During training, fix MCMC parameters
# Only update network parameters
# This reduces correlation
```

### 8.3 NaN Propagation in Gradients

**Issue:** NaN values propagate through backpropagation

**Causes:**
1. log(0) in determinant
2. Division by zero in potentials
3. Gradient of clipped values
4. Exploding gradients

**Solutions:**

1. **NaN-safe operations:**
```python
def safe_divide(a, b, eps=1e-10):
    return a / (jnp.abs(b) + eps)
```

2. **Gradient clipping:**
```python
def clip_gradients(grads, max_norm=1.0):
    total_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grads.values()))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        grads = {k: g * scale for k, g in grads.items()}
    return grads
```

3. **Loss function with NaN protection:**
```python
def safe_loss(loss):
    loss = jnp.where(jnp.isnan(loss), jnp.array(0.0), loss)
    loss = jnp.where(jnp.isinf(loss), jnp.array(1e6), loss)
    return loss
```

### 8.4 Initialization Issues

**Issue:** Random initialization leads to unstable training

**Causes:**
- Large random weights cause large gradients
- Network starts in region with poor wave function
- Determinant values can be extremely small

**Solutions:**

1. **Xavier/Glorot initialization:**
```python
def xavier_init(key, shape):
    fan_in = shape[0]
    fan_out = shape[-1]
    scale = jnp.sqrt(2.0 / (fan_in + fan_out))
    return jax.random.normal(key, shape) * scale
```

2. **Small-scale initialization:**
```python
# Initialize with smaller variance
params['w_one_body'] = jax.random.normal(key, shape) * 0.01
```

3. **Orbital-wise initialization:**
```python
# Initialize orbital weights to approximate hydrogenic orbitals
# This provides better starting point
```

### 8.5 Soft-Core Potential Artifacts

**Issue:** Soft-core potential (α=0.1) introduces errors

**Problems:**
- Potential not truly Coulombic at small r
- Affects cusp conditions
- Systematic energy errors

**Solutions:**

1. **Use cusp conditions:**
```python
# Enforce proper electron-nucleus cusp:
# (∂ψ/∂r)_{r→0} = -Z ψ(0)

# Add cusp correction to wave function
```

2. **Adaptive softening:**
```python
def adaptive_softening(r, alpha_base=0.1):
    # Use smaller α for larger Z
    return alpha_base / (1 + jnp.abs(r))
```

3. **Exact Coulomb with regularization:**
```python
def regularized_coulomb(r):
    # Use exact 1/r but handle r=0 specially
    return jnp.where(jnp.abs(r) < 1e-10,
                     jnp.array(1e10),  # Large but finite
                     1.0 / jnp.abs(r))
```

### 8.6 Memory Efficiency

**Issue:** Large memory usage with many samples

**Causes:**
- Storing full orbital matrices [batch, n_elec, n_elec]
- Multiple intermediate arrays in forward pass
- MCMC chain storage

**Solutions:**

1. **Chunked batch processing:**
```python
def process_in_chunks(fn, data, chunk_size=32):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        results.append(fn(chunk))
    return jnp.concatenate(results)
```

2. **In-place operations:**
```python
# Use jax.lax.scan instead of Python loops for state updates
```

3. **Garbage collection:**
```python
import gc
gc.collect()  # Periodic cleanup
```

### 8.7 Reproducibility

**Issue:** Different results on different runs

**Causes:**
- JAX's PRNGKey splitting order
- Floating-point non-associativity
- Different thread scheduling

**Solutions:**

1. **Explicit seed management:**
```python
# Set global seed at start
key = jax.random.PRNGKey(42)

# Always use consistent splitting pattern
key, subkey1, subkey2 = jax.random.split(key, 3)
```

2. **Deterministic operations:**
```python
# Use jax.lax.tie_in for ordering
result = jax.lax.tie_in(x, y)  # Forces x before y
```

3. **Disable parallelism for testing:**
```python
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_disable_jit', True)  # For debugging
```

---

## 9. Future Improvements

### 9.1 Stage 2 Enhancements

The `ExtendedFermiNet` class (already implemented but not used in Stage 1) includes:

1. **Multi-determinant support:**
   - 4-8 Slater determinants
   - Better electron correlation
   - Learnable combination weights

2. **Jastrow correlation factor:**
   - Explicit electron-electron correlation
   - Satisfies cusp conditions
   - Optional neural network parametrization

3. **Residual connections:**
   - Deeper networks without vanishing gradients
   - Better gradient flow
   - Improved training stability

4. **Larger capacity:**
   - 128×16 network (vs 16×4)
   - 3 interaction layers (vs 1)
   - More expressive power

### 9.2 Training Improvements

1. **Learning rate scheduling:**
   - Decay when energy plateaus
   - Warm-up phase
   - Cyclical learning rates

2. **Gradient clipping:**
   - Global norm clipping
   - Per-parameter clipping
   - Adaptive clipping

3. **Better loss functions:**
   - Energy minimization: L = <E_L>
   - Combined: L = Var(E_L) + λ <E_L>
   - Natural gradient descent

4. **Variance reduction:**
   - Control variates
   - Importance sampling
   - Stratified sampling

### 9.3 MCMC Improvements

1. **Adaptive step size:**
   - Target 50% acceptance
   - Automatic tuning

2. **Multiple walkers:**
   - Parallel independent chains
   - Better mixing

3. **Advanced proposals:**
   - Hamiltonian Monte Carlo
   - Replica exchange
   - Shadow Hamiltonian

4. **Decorrelation analysis:**
   - Monitor autocorrelation time
   - Optimize step count

### 9.4 Architectural Improvements

1. **Enforcing cusp conditions:**
   - Explicit electron-nucleus cusp
   - Explicit electron-electron cusp

2. **Symmetry enforcement:**
   - Spatial symmetries (rotation, reflection)
   - Spin symmetry
   - Permutation invariance

3. **Better initialization:**
   - Hydrogenic orbitals
   - Hartree-Fock orbitals
   - Physically motivated parameters

4. **Hybrid approaches:**
   - FermiNet + Jastrow
   - FermiNet + backflow
   - FermiNet + geminals

### 9.5 Evaluation Improvements

1. **Variance estimates:**
   - Bootstrap error bars
   - Block averaging

2. **Convergence diagnostics:**
   - Energy autocorrelation
   - Gradient norms
   - Parameter drift

3. **Visualization:**
   - Energy vs epoch
   - Electron density plots
   - Orbital visualizations

4. **Comparison methods:**
   - Compare with FCI
   - Compare with DFT
   - Benchmark against other VMC methods

---

## 10. Summary

### 10.1 Achievements

FermiNet Stage 1 successfully implements:

1. **Complete neural network architecture** with Fermionic antisymmetry
2. **Automatic differentiation** for kinetic energy
3. **Langevin MCMC sampling** for VMC
4. **Adam optimizer** for parameter optimization
5. **Modular design** with clear separation of concerns

### 10.2 Challenges

The implementation faces several numerical challenges:

1. **Determinant instability** leading to NaN
2. **Low MCMC acceptance rate** (1.5% vs 50% target)
3. **Small network capacity** limiting expressiveness
4. **Soft-core potential** introducing systematic errors

### 10.3 Next Steps

To achieve working training:

1. **Implement stable determinant calculation**
   - Use slogdet instead of log(det)
   - Add proper regularization

2. **Fix MCMC acceptance rate**
   - Implement adaptive step size
   - Add gradient clipping

3. **Improve initialization**
   - Use Xavier/Glorot initialization
   - Consider physics-based initialization

4. **Increase network capacity**
   - Use ExtendedFermiNet (128×16)
   - Add multi-determinant support

5. **Add better diagnostics**
   - Monitor gradient norms
   - Track energy variance
   - Visualize training dynamics

### 10.4 Lessons Learned

1. **Numerical stability is crucial** in quantum Monte Carlo
2. **MCMC and optimization are coupled** - must be tuned together
3. **Small systems still require careful implementation**
4. **Automatic differentiation enables physics calculations** but introduces new numerical issues
5. **Modular design aids debugging** and incremental improvements

---

## References

1. **Original FermiNet Paper:**
   - Pfau, D., Spencer, J. S., Jones, A. G., & Mehta, P. (2020).
   - "Ab-Initio Solution of the Many-Electron Schrödinger Equation with Deep Neural Networks."
   - Physical Review Research, 2(3), 033426.

2. **Variational Monte Carlo:**
   - Foulkes, W. M. C., Mitas, L., Needs, R. J., & Rajagopal, G. (2001).
   - "Quantum Monte Carlo methods in physics and chemistry."
   - Reviews of Modern Physics, 73(1), 33.

3. **Adam Optimizer:**
   - Kingma, D. P., & Ba, J. (2014).
   - "Adam: A method for stochastic optimization."
   - arXiv preprint arXiv:1412.6980.

4. **Langevin Dynamics:**
   - Roberts, G. O., & Rosenthal, J. S. (2001).
   - "Optimal scaling for various Metropolis-Hastings algorithms."
   - Statistical Science, 16(4), 351-367.

5. **JAX Documentation:**
   - https://jax.readthedocs.io/

---

## Appendix: Code Organization

```
FermiNet/
├── demo/
│   ├── main.py                  # Main training loop
│   ├── network.py               # FermiNet architecture
│   ├── physics.py               # Physical calculations
│   ├── mcmc.py                  # Langevin MCMC sampler
│   ├── trainer.py               # VMC trainer with Adam
│   ├── configs/
│   │   └── h2_config.py        # H2 molecule configuration
│   └── results/                 # Training outputs
│       └── H2_results_*.pkl

docs/
└── Stage1_Complete_Documentation.md  # This document
```

---

## Contact and Support

For questions, issues, or contributions related to FermiNet Stage 1:

1. Review the "Known Issues and Solutions" section for common problems
2. Check JAX documentation for numerical computing questions
3. Refer to the original FermiNet paper for theoretical background

---

**Document Version:** 1.0
**Last Updated:** 2026-01-28
**FermiNet Stage:** 1 (Single Determinant Implementation)
