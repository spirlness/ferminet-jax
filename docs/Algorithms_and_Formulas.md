# FermiNet: Algorithms and Mathematical Formulas

## Table of Contents
1. [Wave Function Theory](#wave-function-theory)
2. [Energy Formulas](#energy-formulas)
3. [Variational Principle](#variational-principle)
4. [MCMC Algorithms](#mcmc-algorithms)
5. [Optimization Algorithms](#optimization-algorithms)
6. [Numerical Considerations](#numerical-considerations)

---

## Wave Function Theory

### 1.1 FermiNet Ansatz

The FermiNet wave function ansatz represents the many-electron wave function as:

$$
\Psi_{\theta}(\mathbf{r}_1, \ldots, \mathbf{r}_N) = \Psi_{\text{Pauli}}(\mathbf{r}_1, \ldots, \mathbf{r}_N) \times \Psi_{\text{Jastrow}}(\mathbf{r}_1, \ldots, \mathbf{r}_N)
$$

where:
- $\mathbf{r}_i \in \mathbb{R}^3$ is the position of electron $i$
- $N$ is the total number of electrons
- $\theta$ represents the neural network parameters
- $\Psi_{\text{Pauli}}$ ensures antisymmetry under electron exchange
- $\Psi_{\text{Jastrow}}$ captures electron correlation

**Key Properties:**
- Antisymmetry: $\Psi(\ldots, \mathbf{r}_i, \ldots, \mathbf{r}_j, \ldots) = -\Psi(\ldots, \mathbf{r}_j, \ldots, \mathbf{r}_i, \ldots)$
- Normalization: $\int |\Psi(\mathbf{r}_1, \ldots, \mathbf{r}_N)|^2 d\mathbf{r}_1 \ldots d\mathbf{r}_N = 1$
- Continuity: $\Psi$ must be continuous everywhere

### 1.2 Slater Determinant Formulation

The Pauli part is constructed as a sum over multiple Slater determinants:

$$
\Psi_{\text{Pauli}}(\mathbf{r}_1, \ldots, \mathbf{r}_N) = \sum_{k=1}^{K} c_k \det \begin{pmatrix}
\phi_{k,1}(\mathbf{r}_1) & \phi_{k,2}(\mathbf{r}_1) & \cdots & \phi_{k,N}(\mathbf{r}_1) \\
\phi_{k,1}(\mathbf{r}_2) & \phi_{k,2}(\mathbf{r}_2) & \cdots & \phi_{k,N}(\mathbf{r}_2) \\
\vdots & \vdots & \ddots & \vdots \\
\phi_{k,1}(\mathbf{r}_N) & \phi_{k,2}(\mathbf{r}_N) & \cdots & \phi_{k,N}(\mathbf{r}_N)
\end{pmatrix}
$$

where:
- $K$ is the number of determinants
- $c_k$ are linear combination coefficients (often set to 1)
- $\phi_{k,i}$ are single-electron orbitals produced by neural networks

**Determinant Evaluation:**

For determinant $k$, let $\mathbf{M}^{(k)}$ be the $N \times N$ matrix with entries $M_{ij}^{(k)} = \phi_{k,i}(\mathbf{r}_j)$. The determinant is:

$$
\det \mathbf{M}^{(k)} = \sum_{\sigma \in S_N} \text{sgn}(\sigma) \prod_{i=1}^N M_{i,\sigma(i)}^{(k)}
$$

where $S_N$ is the symmetric group of $N$ elements.

**Spin Adaptation:**

For spin-restricted calculations with $N_\uparrow$ spin-up electrons and $N_\downarrow$ spin-down electrons:

$$
\Psi_{\text{Pauli}} = \det(\mathbf{M}_\uparrow) \det(\mathbf{M}_\downarrow)
$$

For spin-unrestricted (different spatial orbitals for different spins):

$$
\Psi_{\text{Pauli}} = \det \begin{pmatrix}
\mathbf{M}_\uparrow & \mathbf{0} \\
\mathbf{0} & \mathbf{M}_\downarrow
\end{pmatrix}
$$

### 1.3 Single-Electron Orbital Construction

Each single-electron orbital $\phi_{k,i}$ is constructed using a neural network with electron exchange symmetry built in:

$$
\phi_{k,i}(\mathbf{r}_j) = \sum_{m=1}^{M} w_{k,i,m} \cdot \xi_m(\mathbf{r}_j, \{\mathbf{r}_\ell\}_{\ell \neq j})
$$

where:
- $M$ is the number of envelope functions
- $w_{k,i,m}$ are learned weights
- $\xi_m$ are envelope functions that depend on electron $j$ and all other electrons

**Envelope Functions:**

The envelope functions $\xi_m$ are constructed through a multi-layer perceptron (MLP):

1. **Input Features:**
   - One-electron features (electron-nucleus distances):
     $$h_i^a = \frac{|\mathbf{r}_i - \mathbf{R}_a|}{n_a}$$
     where $\mathbf{R}_a$ is nucleus $a$ position and $n_a$ is a normalization constant
   - Two-electron features (electron-electron distances):
     $$h_{ij} = |\mathbf{r}_i - \mathbf{r}_j|$$

2. **Mean Pooling:**
   For each electron $i$, aggregate information about other electrons:
   $$\bar{h}_i = \frac{1}{N-1} \sum_{j \neq i} \sigma(\mathbf{W}_1 h_{ij} + \mathbf{b}_1)$$

3. **Combined Features:**
   $$f_i = \text{concatenate}([h_i^1, h_i^2, \ldots, h_i^A, \bar{h}_i])$$
   where $A$ is the number of nuclei

4. **Neural Network Layers:**
   $$\mathbf{x}^{(l+1)} = \sigma(\mathbf{W}^{(l)} \mathbf{x}^{(l)} + \mathbf{b}^{(l)}), \quad l = 1, \ldots, L$$

5. **Envelope Output:**
   $$\xi_m(\mathbf{r}_j, \{\mathbf{r}_\ell\}_{\ell \neq j}) = \exp(-\eta_m^2 |\mathbf{r}_j - \mathbf{R}_\text{center}(m)|^2) \cdot g_m(f_j)$$

where $\eta_m$ is a learned scale parameter and $g_m$ is the neural network output.

### 1.4 Multi-Determinant Combination

The multi-determinant expansion improves flexibility and captures static correlation:

$$
\Psi_{\text{Pauli}} = \sum_{k=1}^{K} c_k D_k
$$

**Determination of Coefficients:**

In practice, the coefficients $c_k$ can be:
1. Fixed to 1 (equal weight combination)
2. Learned as additional neural network parameters
3. Determined via orthogonalization schemes

**Practical Implementation:**

```python
# Pseudocode for multi-determinant evaluation
def evaluate_pauli_wavefunction(electron_positions, theta):
    """Evaluate sum of K Slater determinants."""
    total = 0.0

    for k in range(K):  # K determinants
        # Build determinant matrix
        M = np.zeros((N, N))
        for i in range(N):  # electron i
            for j in range(N):  # orbital j
                # Evaluate single-electron orbital
                features = compute_features(electron_positions, i)
                orbital = neural_network(features, theta)
                M[i, j] = orbital

        # Compute determinant
        det_M = np.linalg.det(M)
        total += c_k * det_M

    return total
```

**Determinant Derivatives:**

For efficient computation of gradients, use the identity:

$$
\frac{\partial}{\partial \theta} \det \mathbf{M} = \det \mathbf{M} \cdot \text{tr}\left(\mathbf{M}^{-1} \frac{\partial \mathbf{M}}{\partial \theta}\right)
$$

This can be computed efficiently without explicitly forming the inverse matrix:

```python
def determinant_derivative(M, dM_dtheta, det_M):
    """Compute d(det M)/dtheta efficiently."""
    # Solve M^T * x = dM_dtheta for x
    inv_M_T = np.linalg.solve(M.T, dM_dtheta)

    # Derivative = det M * trace(inv_M^T * dM_dtheta)
    d_det = det_M * np.trace(inv_M_T)
    return d_det
```

### 1.5 Jastrow Correlation Factor

The Jastrow factor captures dynamic electron correlation:

$$
\Psi_{\text{Jastrow}}(\mathbf{r}_1, \ldots, \mathbf{r}_N) = \exp\left(J(\mathbf{r}_1, \ldots, \mathbf{r}_N)\right)
$$

**Jastrow Function Form:**

The Jastrow function $J$ is typically decomposed into one- and two-body terms:

$$
J = J_1 + J_2
$$

**One-Body Term (Electron-Nucleus Correlation):**

$$
J_1 = \sum_{i=1}^N \sum_{a=1}^A f_{ia}(|\mathbf{r}_i - \mathbf{R}_a|)
$$

with function:
$$
f_{ia}(r) = \sum_{k=1}^{K} w_{iak} \cdot \phi_k(r)
$$

where $\phi_k(r)$ are radial basis functions (e.g., polynomial, exponential, or Gaussian).

**Two-Body Term (Electron-Electron Correlation):**

$$
J_2 = \sum_{i<j}^N g_{ij}(|\mathbf{r}_i - \mathbf{r}_j|)
$$

with function:
$$
g_{ij}(r) = \sum_{k=1}^{K} v_{ijk} \cdot \psi_k(r)
$$

**Cusp Conditions:**

The Jastrow factor must satisfy the electron-electron cusp condition:

$$
\left.\frac{\partial g_{ij}(r)}{\partial r}\right|_{r=0} = \frac{1}{2}
$$

and the electron-nucleus cusp condition:

$$
\left.\frac{\partial f_{ia}(r)}{\partial r}\right|_{r=0} = -Z_a
$$

where $Z_a$ is the nuclear charge.

For Gaussian basis functions, use cusp correction:

$$
g_{ij}^{\text{cusp}}(r) = \frac{r}{2(1 + \alpha r)}
$$

**Neural Network Jastrow:**

Alternatively, implement Jastrow as a neural network:

$$
J = \text{NN}_\theta\left(\{|\mathbf{r}_i - \mathbf{r}_j|\}_{i<j}, \{|\mathbf{r}_i - \mathbf{R}_a|\}_{i,a}\right)
$$

**Jastrow Gradient:**

$$
\frac{\partial \Psi_{\text{Jastrow}}}{\partial r_i} = \Psi_{\text{Jastrow}} \cdot \frac{\partial J}{\partial r_i}
$$

$$
\frac{\partial^2 \Psi_{\text{Jastrow}}}{\partial r_i^2} = \Psi_{\text{Jastrow}} \left[\left(\frac{\partial J}{\partial r_i}\right)^2 + \frac{\partial^2 J}{\partial r_i^2}\right]
$$

**Cusp Condition Enforcement:**

```python
def jastrow_with_cusp(r, params):
    """Jastrow function with enforced cusp condition."""
    # Neural network part
    j_nn = neural_network_jastrow(r, params)

    # Cusp correction (e.g., Pade-Jastrow)
    j_cusp = r / (2 * (1 + alpha * r))

    return j_nn + j_cusp
```

---

## Energy Formulas

### 2.1 Hamiltonian Operator

The non-relativistic electronic Hamiltonian in atomic units (Born-Oppenheimer approximation):

$$
\hat{H} = -\frac{1}{2}\sum_{i=1}^N \nabla_i^2 - \sum_{i=1}^N \sum_{a=1}^A \frac{Z_a}{|\mathbf{r}_i - \mathbf{R}_a|} + \sum_{i<j}^N \frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}
$$

where:
- $N$ is the number of electrons
- $A$ is the number of nuclei
- $Z_a$ is the charge of nucleus $a$
- $\mathbf{R}_a$ is the position of nucleus $a$
- $\mathbf{r}_i$ is the position of electron $i$

**Component Breakdown:**

1. **Kinetic Energy Operator:**
   $$\hat{T} = -\frac{1}{2}\sum_{i=1}^N \nabla_i^2$$

2. **External Potential (Nuclear Attraction):**
   $$\hat{V}_{\text{ne}} = -\sum_{i=1}^N \sum_{a=1}^A \frac{Z_a}{|\mathbf{r}_i - \mathbf{R}_a|}$$

3. **Electron Repulsion:**
   $$\hat{V}_{\text{ee}} = \sum_{i<j}^N \frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}$$

**Nuclear-Nuclear Repulsion (constant):**

$$
E_{\text{nn}} = \sum_{a<b}^A \frac{Z_a Z_b}{|\mathbf{R}_a - \mathbf{R}_b|}
$$

### 2.2 Local Energy Expression

The local energy is defined as the ratio of Hamiltonian applied to wave function to the wave function:

$$
E_L(\mathbf{R}) = \frac{\hat{H} \Psi_{\theta}(\mathbf{R})}{\Psi_{\theta}(\mathbf{R})} = \frac{\hat{T} \Psi}{\Psi} + \frac{\hat{V}_{\text{ne}} \Psi}{\Psi} + \frac{\hat{V}_{\text{ee}} \Psi}{\Psi}
$$

where $\mathbf{R} = (\mathbf{r}_1, \ldots, \mathbf{r}_N)$ represents all electron positions.

**Key Property:**

The local energy is used because the expectation value can be written as:

$$
\langle E \rangle = \frac{\int \Psi^*(\mathbf{R}) \hat{H} \Psi(\mathbf{R}) d\mathbf{R}}{\int |\Psi(\mathbf{R})|^2 d\mathbf{R}} = \int |\Psi(\mathbf{R})|^2 E_L(\mathbf{R}) d\mathbf{R}
$$

This allows Monte Carlo integration without explicit integration of the Hamiltonian.

### 2.3 Kinetic Energy (Gradient Formula)

For a product wave function $\Psi = \Psi_{\text{Pauli}} \Psi_{\text{Jastrow}}$, the kinetic energy is:

$$
\frac{\hat{T} \Psi}{\Psi} = -\frac{1}{2} \sum_{i=1}^N \left[\frac{\nabla_i^2 \Psi}{\Psi}\right]
$$

Using the logarithmic derivative:

$$
\frac{\nabla_i \Psi}{\Psi} = \frac{\nabla_i \Psi_{\text{Pauli}}}{\Psi_{\text{Pauli}}} + \frac{\nabla_i \Psi_{\text{Jastrow}}}{\Psi_{\text{Jastrow}}} = \nabla_i \ln \Psi_{\text{Pauli}} + \nabla_i \ln \Psi_{\text{Jastrow}}
$$

Define the **wave function gradient**:

$$
\mathbf{F}_i \equiv \nabla_i \ln \Psi = \nabla_i \ln \Psi_{\text{Pauli}} + \nabla_i \ln \Psi_{\text{Jastrow}}
$$

Then the kinetic energy per electron:

$$
\frac{\hat{T}_i \Psi}{\Psi} = -\frac{1}{2} \left[\nabla_i^2 \ln \Psi + (\nabla_i \ln \Psi)^2\right] = -\frac{1}{2} \left[\nabla_i \cdot \mathbf{F}_i + \|\mathbf{F}_i\|^2\right]
$$

**Total Kinetic Energy:**

$$
E_{\text{kin}} = -\frac{1}{2} \sum_{i=1}^N \left[\nabla_i \cdot \mathbf{F}_i + \|\mathbf{F}_i\|^2\right]
$$

**Computational Form:**

In Cartesian coordinates:

$$
\nabla_i \cdot \mathbf{F}_i = \sum_{\alpha = x,y,z} \frac{\partial^2}{\partial r_{i,\alpha}^2} \ln \Psi(\mathbf{R})
$$

$$
\|\mathbf{F}_i\|^2 = \sum_{\alpha = x,y,z} \left(\frac{\partial}{\partial r_{i,\alpha}} \ln \Psi(\mathbf{R})\right)^2
$$

**Pauli Part Gradients:**

For a single determinant $\det \mathbf{M}$:

$$
\frac{\nabla_i \det \mathbf{M}}{\det \mathbf{M}} = \text{row}_i(\mathbf{M}^{-1}) \cdot \nabla_i \text{row}_i(\mathbf{M})
$$

where $\text{row}_i(\mathbf{M})$ is the $i$-th row of matrix $\mathbf{M}$.

**Laplacian of Logarithm:**

$$
\nabla_i^2 \ln \Psi = \frac{\nabla_i^2 \Psi}{\Psi} - \left(\frac{\nabla_i \Psi}{\Psi}\right)^2
$$

**Algorithm for Computing Kinetic Energy:**

```python
def local_kinetic_energy(wavefunction, positions, theta):
    """Compute local kinetic energy at given electron positions."""
    grad_log_psi = wavefunction.grad_log_psi(positions, theta)  # Shape: (N, 3)
    laplacian_log_psi = wavefunction.laplacian_log_psi(positions, theta)  # Shape: (N,)

    E_kin = -0.5 * np.sum(laplacian_log_psi + np.sum(grad_log_psi**2, axis=1))
    return E_kin
```

### 2.4 Potential Energy Components

**Nuclear-Electron Potential Energy:**

$$
E_{\text{ne}} = -\sum_{i=1}^N \sum_{a=1}^A \frac{Z_a}{|\mathbf{r}_i - \mathbf{R}_a|}
$$

This is a simple pairwise sum over all electron-nucleus pairs.

**Soft-Core Nuclear-Electron Potential:**

To avoid singularities in numerical calculations:

$$
E_{\text{ne}}^{\text{soft}} = -\sum_{i=1}^N \sum_{a=1}^A \frac{Z_a}{\sqrt{|\mathbf{r}_i - \mathbf{R}_a|^2 + \epsilon^2}}
$$

where $\epsilon$ is a small soft-core parameter (e.g., $\epsilon = 10^{-6}$).

**Electron-Electron Potential Energy:**

$$
E_{\text{ee}} = \sum_{i<j}^N \frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}
$$

**Soft-Core Electron-Electron Potential:**

$$
E_{\text{ee}}^{\text{soft}} = \sum_{i<j}^N \frac{1{+}f_{\text{ee}}(|\mathbf{r}_i - \mathbf{r}_j|)}{\sqrt{|\mathbf{r}_i - \mathbf{r}_j|^2 + \epsilon^2}}
$$

where $f_{\text{ee}}$ is a smoothing function that approaches 1 at small distances.

**Cusp Condition Handling:**

For fermionic systems, the Jastrow factor ensures:

$$
\lim_{r_{ij} \to 0} \Psi \propto (1 + \frac{r_{ij}}{2}) \times \text{regular part}
$$

This regularizes the electron-electron singularity.

**Implementation:**

```python
def local_potential_energy(positions, nuclei, soft_core=True, epsilon=1e-6):
    """Compute local potential energy."""
    N = positions.shape[0]  # Number of electrons
    A = nuclei.shape[0]     # Number of nuclei

    E_ne = 0.0  # Nuclear-electron
    E_ee = 0.0  # Electron-electron

    # Nuclear-electron interaction
    for i in range(N):
        for a in range(A):
            r = np.linalg.norm(positions[i] - nuclei[a]['position'])
            if soft_core:
                E_ne -= nuclei[a]['charge'] / np.sqrt(r**2 + epsilon**2)
            else:
                E_ne -= nuclei[a]['charge'] / r

    # Electron-electron interaction
    for i in range(N):
        for j in range(i+1, N):
            r = np.linalg.norm(positions[i] - positions[j])
            if soft_core:
                E_ee += 1.0 / np.sqrt(r**2 + epsilon**2)
            else:
                E_ee += 1.0 / r

    return E_ne + E_ee
```

### 2.5 Total Local Energy

$$
E_L(\mathbf{R}) = E_{\text{kin}} + E_{\text{ne}} + E_{\text{ee}} + E_{\text{nn}}
$$

where $E_{\text{nn}}$ is the constant nuclear-nuclear repulsion.

---

## Variational Principle

### 3.1 Variational Monte Carlo Derivation

**Variational Principle:**

For any trial wave function $\Psi_\theta$, the expectation value of the Hamiltonian is an upper bound to the exact ground state energy $E_0$:

$$
E_\theta = \frac{\langle \Psi_\theta | \hat{H} | \Psi_\theta \rangle}{\langle \Psi_\theta | \Psi_\theta \rangle} \geq E_0
$$

**Proof Sketch:**

Expand $\Psi_\theta$ in the eigenbasis of $\hat{H}$:

$$
\Psi_\theta = \sum_n c_n \Phi_n
$$

where $\hat{H} \Phi_n = E_n \Phi_n$ and $E_0 \leq E_1 \leq E_2 \leq \ldots$

Then:

$$
E_\theta = \frac{\sum_n |c_n|^2 E_n}{\sum_n |c_n|^2} = \sum_n p_n E_n \geq E_0 \sum_n p_n = E_0
$$

where $p_n = |c_n|^2 / \sum_m |c_m|^2 \geq 0$ and $\sum_n p_n = 1$.

**Minimization Objective:**

The optimization problem is:

$$
\min_\theta E_\theta = \min_\theta \frac{\langle \Psi_\theta | \hat{H} | \Psi_\theta \rangle}{\langle \Psi_\theta | \Psi_\theta \rangle}
$$

### 3.2 Energy Expectation Value

**Integral Form:**

$$
E_\theta = \frac{\int \Psi_\theta^*(\mathbf{R}) \hat{H} \Psi_\theta(\mathbf{R}) d\mathbf{R}}{\int |\Psi_\theta(\mathbf{R})|^2 d\mathbf{R}}
$$

**Monte Carlo Representation:**

Define the probability distribution:

$$
p_\theta(\mathbf{R}) = \frac{|\Psi_\theta(\mathbf{R})|^2}{\int |\Psi_\theta(\mathbf{R})|^2 d\mathbf{R}}
$$

Then the energy becomes:

$$
E_\theta = \int p_\theta(\mathbf{R}) E_L(\mathbf{R}) d\mathbf{R} = \mathbb{E}_{\mathbf{R} \sim p_\theta}[E_L(\mathbf{R})]
$$

**Monte Carlo Estimator:**

Draw $M$ samples $\{\mathbf{R}^{(m)}\}_{m=1}^M$ from $p_\theta(\mathbf{R})$:

$$
\hat{E}_\theta = \frac{1}{M} \sum_{m=1}^M E_L(\mathbf{R}^{(m)})
$$

**Statistical Error:**

The standard error of the mean:

$$
\sigma_{\hat{E}} = \frac{\sigma_{E_L}}{\sqrt{M}}
$$

where $\sigma_{E_L}^2 = \mathbb{E}[E_L^2] - \mathbb{E}[E_L]^2$.

### 3.3 Gradient of Energy

**Objective:**

Compute $\nabla_\theta E_\theta$ for gradient-based optimization.

**Analytical Gradient:**

Using the local energy formulation:

$$
E_\theta = \int p_\theta(\mathbf{R}) E_L(\mathbf{R}) d\mathbf{R}
$$

Differentiate with respect to $\theta$:

$$
\nabla_\theta E_\theta = \int \left[\nabla_\theta p_\theta(\mathbf{R}) E_L(\mathbf{R}) + p_\theta(\mathbf{R}) \nabla_\theta E_L(\mathbf{R})\right] d\mathbf{R}
$$

Using $\nabla_\theta p_\theta(\mathbf{R}) = p_\theta(\mathbf{R}) \nabla_\theta \ln p_\theta(\mathbf{R}) = 2 p_\theta(\mathbf{R}) \nabla_\theta \ln |\Psi_\theta(\mathbf{R})|$:

$$
\nabla_\theta E_\theta = 2 \int p_\theta(\mathbf{R}) \left[\nabla_\theta \ln |\Psi_\theta(\mathbf{R})| \cdot E_L(\mathbf{R})\right] d\mathbf{R} + \int p_\theta(\mathbf{R}) \nabla_\theta E_L(\mathbf{R}) d\mathbf{R}
$$

**Simplification:**

The second term is usually negligible compared to the first term:

$$
\nabla_\theta E_\theta \approx 2 \mathbb{E}_{\mathbf{R} \sim p_\theta}\left[\nabla_\theta \ln |\Psi_\theta(\mathbf{R})| \cdot E_L(\mathbf{R})\right]
$$

**More Accurate Gradient:**

A better approximation includes both terms:

$$
\nabla_\theta E_\theta = 2 \left(\mathbb{E}[\nabla_\theta \ln |\Psi| \cdot E_L] - \mathbb{E}[\nabla_\theta \ln |\Psi|] \mathbb{E}[E_L]\right)
$$

**Monte Carlo Gradient Estimator:**

$$
\widehat{\nabla_\theta E_\theta} = \frac{2}{M} \sum_{m=1}^M \left[\nabla_\theta \ln |\Psi_\theta(\mathbf{R}^{(m)})| \cdot \left(E_L(\mathbf{R}^{(m)}) - \hat{E}_\theta\right)\right]
$$

This has the advantage that it reduces variance when $E_L$ and $\nabla_\theta \ln |\Psi|$ are correlated.

**Logarithmic Derivative:**

$$
\nabla_\theta \ln |\Psi_\theta(\mathbf{R})| = \frac{\nabla_\theta \Psi_\theta(\mathbf{R})}{\Psi_\theta(\mathbf{R})}
$$

For product wave function $\Psi = \Psi_{\text{Pauli}} \Psi_{\text{Jastrow}}$:

$$
\nabla_\theta \ln |\Psi| = \nabla_\theta \ln |\Psi_{\text{Pauli}}| + \nabla_\theta \ln |\Psi_{\text{Jastrow}}|
$$

### 3.4 Energy Minimization Algorithm

**Variational Monte Carlo (VMC) Algorithm:**

```
Algorithm: VMC Energy Minimization
Input: Initial parameters θ0, number of steps T
Output: Optimized parameters θ*

Initialize θ ← θ0
For t = 1 to T:
    1. Generate M samples {R^(m)} from p_θ(R) using MCMC
    2. Compute local energies E_L(R^(m)) for each sample
    3. Compute energy estimate E_θ = (1/M) Σ E_L(R^(m))
    4. Compute gradient estimate:
       ∇E_θ = (2/M) Σ [∇ln|Ψ_θ(R^(m))| · (E_L(R^(m)) - E_θ)]
    5. Update parameters: θ ← θ - α ∇E_θ (using optimizer)
    6. Optionally: adjust learning rate α

Return θ
```

**Key Considerations:**

1. **Burn-in:** Discard initial MCMC samples to reach equilibrium
2. **Blocking:** Use blocking analysis to estimate statistical errors
3. **Reblocking:** Combine consecutive samples to reduce autocorrelation
4. **Replay Memory:** Use a buffer of samples for gradient estimation

**Pseudocode Implementation:**

```python
def vmc_optimization(wavefunction, system, optimizer, n_steps, n_samples, batch_size):
    """Variational Monte Carlo optimization."""
    theta = wavefunction.initialize_parameters()
    positions = initialize_positions(system.n_electrons, system.box_size)

    for step in range(n_steps):
        # MCMC sampling (thermalization)
        for _ in range(batch_size // 10):
            positions = mcmc_step(positions, wavefunction, theta, system)

        # Collect samples
        samples = []
        local_energies = []
        log_grads = []

        for _ in range(n_samples):
            positions = mcmc_step(positions, wavefunction, theta, system)
            samples.append(positions.copy())

            # Compute local energy
            E_L = local_energy(positions, wavefunction, theta, system)
            local_energies.append(E_L)

            # Compute gradient of log wave function
            grad_log = wavefunction.grad_log_psi(positions, theta)
            log_grads.append(grad_log)

        # Compute energy and gradient estimates
        samples = np.array(samples)
        local_energies = np.array(local_energies)
        log_grads = np.array(log_grads)

        E_mean = np.mean(local_energies)
        E_centered = local_energies - E_mean

        grad_E = 2 * np.mean(log_grads * E_centered[:, None, None], axis=0)

        # Update parameters
        theta = optimizer.update(theta, grad_E)

        # Log progress
        if step % 100 == 0:
            print(f"Step {step}: E = {E_mean:.6f}")

    return theta
```

### 3.5 Variance Minimization

**Alternative Objective:**

Minimize the variance of the local energy instead of the energy:

$$
\sigma_\theta^2 = \mathbb{E}_{\mathbf{R} \sim p_\theta}[(E_L(\mathbf{R}) - \mathbb{E}[E_L])^2]
$$

**Advantages:**
- Less sensitive to normalization
- Converges to exact wave function when variance → 0
- Can stabilize optimization

**Variance Gradient:**

$$
\nabla_\theta \sigma_\theta^2 = 4 \mathbb{E}\left[\nabla_\theta \ln |\Psi| \cdot E_L \cdot (E_L - E_\theta)\right]
$$

---

## MCMC Algorithms

### 4.1 Langevin Dynamics Equations

**Overdamped Langevin Dynamics:**

Langevin dynamics samples from the distribution $p(\mathbf{R}) \propto |\Psi(\mathbf{R})|^2$:

$$
\frac{d\mathbf{R}}{dt} = \frac{1}{2} \nabla \ln |\Psi(\mathbf{R})|^2 + \boldsymbol{\xi}(t) = \nabla \ln |\Psi(\mathbf{R})| + \boldsymbol{\xi}(t)
$$

where:
- $\nabla \ln |\Psi(\mathbf{R})|$ is the **quantum force** (drift term)
- $\boldsymbol{\xi}(t)$ is Gaussian white noise with $\langle \boldsymbol{\xi}(t) \boldsymbol{\xi}(t')\rangle = \delta(t-t')$

**Discrete Time Stepping:**

Euler-Maruyama discretization with time step $\tau$:

$$
\mathbf{R}_{t+1} = \mathbf{R}_t + \tau \nabla \ln |\Psi(\mathbf{R}_t)| + \sqrt{\tau} \cdot \mathcal{N}(0, 1)
$$

where $\mathcal{N}(0, 1)$ is standard normal noise.

**Component-wise for Each Electron:**

For electron $i$:

$$
\mathbf{r}_i^{(t+1)} = \mathbf{r}_i^{(t)} + \tau \mathbf{F}_i^{(t)} + \sqrt{\tau} \cdot \mathbf{n}_i
$$

where:
- $\mathbf{F}_i^{(t)} = \nabla_i \ln |\Psi(\mathbf{R}^{(t)})|$ is the drift force on electron $i$
- $\mathbf{n}_i \sim \mathcal{N}(0, \mathbf{I}_3)$ is 3D Gaussian noise

**Drift Force Computation:**

$$
\mathbf{F}_i = \nabla_i \ln \Psi = \nabla_i \ln \Psi_{\text{Pauli}} + \nabla_i \ln \Psi_{\text{Jastrow}}
$$

**Algorithm:**

```python
def langevin_step(positions, wavefunction, theta, tau):
    """Perform one Langevin dynamics step."""
    # Compute drift forces
    drift = wavefunction.grad_log_psi(positions, theta)  # Shape: (N, 3)

    # Generate Gaussian noise
    noise = np.random.normal(0, 1, positions.shape) * np.sqrt(tau)

    # Update positions
    new_positions = positions + tau * drift + noise

    return new_positions
```

### 4.2 Metropolis-Hastings Acceptance Criterion

**Metropolis-Hastings with Langevin Proposal:**

1. **Propose:** $\mathbf{R}' = \mathbf{R} + \tau \nabla \ln |\Psi(\mathbf{R})| + \sqrt{\tau} \cdot \boldsymbol{\xi}$
2. **Accept with probability:**
   $$A = \min\left(1, \frac{|\Psi(\mathbf{R}')|^2 \cdot q(\mathbf{R}|\mathbf{R}')}{|\Psi(\mathbf{R})|^2 \cdot q(\mathbf{R}'|\mathbf{R})}\right)$$
   where $q$ is the proposal kernel.

**Simplified Acceptance Ratio:**

For Langevin with symmetric noise, the proposal is approximately symmetric, giving:

$$
A = \min\left(1, \frac{|\Psi(\mathbf{R}')|^2}{|\Psi(\mathbf{R})|^2}\right) = \min\left(1, \exp\left(2 \ln \frac{|\Psi(\mathbf{R}')|}{|\Psi(\mathbf{R})|}\right)\right)
$$

**Numerical Implementation:**

To avoid overflow/underflow, compute log acceptance ratio:

$$
\ln A = 2 \left[\ln |\Psi(\mathbf{R}')| - \ln |\Psi(\mathbf{R})|\right]
$$

Accept if $\ln(\text{uniform}(0, 1)) < \min(0, \ln A)$.

**Full MCMC Step:**

```python
def mcmc_metropolis_hastings(positions, wavefunction, theta, tau, system):
    """Metropolis-Hastings step with Langevin proposal."""
    # Current log probability
    log_psi_current = wavefunction.log_psi(positions, theta)

    # Propose new positions
    drift = wavefunction.grad_log_psi(positions, theta)
    noise = np.random.normal(0, 1, positions.shape) * np.sqrt(tau)
    positions_proposed = positions + tau * drift + noise

    # Enforce boundary conditions
    positions_proposed = apply_boundary_conditions(positions_proposed, system.box_size)

    # Proposed log probability
    log_psi_proposed = wavefunction.log_psi(positions_proposed, theta)

    # Acceptance ratio
    log_accept = 2 * (log_psi_proposed - log_psi_current)

    # Accept or reject
    if np.log(np.random.random()) < min(0, log_accept):
        return positions_proposed
    else:
        return positions
```

**Electron-by-Electron Updates:**

For better efficiency, update one electron at a time:

```python
def mcmc_electron_wise(positions, wavefunction, theta, tau, system):
    """Update one electron at a time."""
    positions_new = positions.copy()

    for i in range(len(positions)):
        # Compute drift for electron i only
        drift_i = wavefunction.grad_log_psi_single(positions_new, i, theta)
        noise_i = np.random.normal(0, 1, 3) * np.sqrt(tau)

        # Propose new position for electron i
        r_i_old = positions_new[i].copy()
        positions_new[i] = r_i_old + tau * drift_i + noise_i

        # Enforce boundary conditions
        positions_new[i] = apply_boundary_conditions(positions_new[i], system.box_size)

        # Compute log acceptance
        log_psi_old = wavefunction.log_psi(positions_new, theta)
        positions_new[i] = r_i_old  # Restore old position
        log_psi_old = wavefunction.log_psi(positions_new, theta)

        positions_new[i] = r_i_old + tau * drift_i + noise_i  # Back to proposed
        log_psi_new = wavefunction.log_psi(positions_new, theta)

        log_accept = 2 * (log_psi_new - log_psi_old)

        # Accept or reject
        if np.log(np.random.random()) < min(0, log_accept):
            pass  # Keep proposed position
        else:
            positions_new[i] = r_i_old  # Revert to old position

    return positions_new
```

### 4.3 Sampling Efficiency Analysis

**Acceptance Rate:**

The acceptance rate depends on the time step $\tau$:
- Too large $\tau$: Low acceptance (proposals too far from equilibrium)
- Too small $\tau$: High autocorrelation (slow mixing)

**Optimal Time Step:**

For Langevin dynamics, optimal $\tau$ typically satisfies:

$$
\tau_{\text{opt}} \approx \frac{1}{\sigma_F^2}
$$

where $\sigma_F^2$ is the variance of the drift force.

**Autocorrelation Time:**

The integrated autocorrelation time for an observable $O$:

$$
\tau_{\text{int}} = 1 + 2 \sum_{t=1}^\infty \rho(t)
$$

where $\rho(t)$ is the autocorrelation function:

$$
\rho(t) = \frac{\langle O_{t} O_{0}\rangle - \langle O\rangle^2}{\langle O^2\rangle - \langle O\rangle^2}
$$

**Effective Sample Size:**

For $M$ samples with autocorrelation time $\tau_{\text{int}}$:

$$
M_{\text{eff}} = \frac{M}{\tau_{\text{int}}}
$$

**Statistical Error:**

The standard error with autocorrelation:

$$
\sigma_{\bar{O}} = \frac{\sigma_O}{\sqrt{M_{\text{eff}}}} = \frac{\sigma_O \sqrt{\tau_{\text{int}}}}{\sqrt{M}}
$$

**Block Averaging:**

Compute errors using block averaging:

```python
def block_averaging(data, block_size):
    """Compute statistical error using block averaging."""
    n_blocks = len(data) // block_size
    blocks = np.array([np.mean(data[i*block_size:(i+1)*block_size])
                      for i in range(n_blocks)])

    mean = np.mean(data)
    error = np.std(blocks, ddof=1) / np.sqrt(n_blocks)

    return mean, error
```

**Adaptive Time Step:**

Adjust $\tau$ to maintain target acceptance rate $\alpha_{\text{target}}$ (typically 0.5-0.7):

```python
def adaptive_tau(tau, acceptance_rate, target=0.5, factor=0.1):
    """Adjust time step based on acceptance rate."""
    if acceptance_rate < target:
        tau *= (1 - factor)  # Decrease τ
    else:
        tau *= (1 + factor)  # Increase τ
    return min(max(tau, 1e-6), 1.0)  # Clamp to reasonable range
```

**Sampling Metrics:**

Key metrics to monitor:

1. **Acceptance Rate:** Target ~50-70%
2. **Autocorrelation Time:** Lower is better
3. **Effective Sample Size:** Higher is better
4. **Mixing Rate:** Rate of decorrelation
5. **Equilibration Time:** Burn-in period length

---

## Optimization Algorithms

### 5.1 Adam Optimizer Equations

**Adam (Adaptive Moment Estimation):**

Adam combines momentum with adaptive learning rates:

**Update Rules:**

Initialize:
- $\mathbf{m}_0 = \mathbf{0}$ (first moment vector)
- $\mathbf{v}_0 = \mathbf{0}$ (second moment vector)
- $t = 0$ (timestep)

For each gradient $\mathbf{g}_t = \nabla_\theta E_\theta$:

1. **Increment timestep:**
   $$t \leftarrow t + 1$$

2. **Update biased first moment estimate:**
   $$\mathbf{m}_t \leftarrow \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$$

3. **Update biased second moment estimate:**
   $$\mathbf{v}_t \leftarrow \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$

4. **Compute bias-corrected first moment:**
   $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}$$

5. **Compute bias-corrected second moment:**
   $$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

6. **Update parameters:**
   $$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \odot \hat{\mathbf{m}}_t$$

where:
- $\alpha$ is the learning rate (e.g., $10^{-3}$)
- $\beta_1$ is the exponential decay rate for first moment (e.g., $0.9$)
- $\beta_2$ is the exponential decay rate for second moment (e.g., $0.999$)
- $\epsilon$ is a small number for numerical stability (e.g., $10^{-8}$)
- $\odot$ denotes element-wise multiplication

**Hyperparameters:**

- **Learning rate** $\alpha$: Controls step size (typically $10^{-4}$ to $10^{-2}$)
- **$\beta_1$**: Momentum parameter (typically $0.9$)
- **$\beta_2$**: Adaptive learning rate parameter (typically $0.999$)
- **$\epsilon$**: Numerical stability (typically $10^{-8}$)

**Algorithm:**

```python
class AdamOptimizer:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = None  # First moment
        self.v = None  # Second moment

    def update(self, params, gradient):
        """Update parameters using Adam."""
        if self.m is None:
            # Initialize moments
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Update biased moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update parameters
        params = params - self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params
```

### 5.2 Gradient Clipping

**Purpose:**

Prevent exploding gradients that cause unstable optimization.

**Clipping Strategies:**

1. **Clipping by Norm:**
   $$\mathbf{g}_{\text{clipped}} = \mathbf{g} \cdot \min\left(1, \frac{c}{\|\mathbf{g}\|_2}\right)$$
   where $c$ is the maximum allowed norm.

2. **Clipping by Value:**
   $$g_{\text{clipped}}^{(i)} = \text{clip}(g^{(i)}, -c, c)$$
   where clip constrains each component to $[-c, c]$.

3. **Adaptive Clipping:**
   $$c = \text{percentile}(\|\mathbf{g}\|, p)$$
   Clip to the $p$-th percentile of recent gradient norms.

**Algorithm (Norm Clipping):**

```python
def clip_gradient_by_norm(gradient, max_norm=1.0):
    """Clip gradient by L2 norm."""
    grad_norm = np.linalg.norm(gradient)
    if grad_norm > max_norm:
        gradient = gradient * (max_norm / grad_norm)
    return gradient
```

**Algorithm (Value Clipping):**

```python
def clip_gradient_by_value(gradient, min_val=-1.0, max_val=1.0):
    """Clip gradient by value."""
    return np.clip(gradient, min_val, max_val)
```

**Combined with Adam:**

```python
def vmc_step(wavefunction, params, optimizer, clip_norm=1.0):
    """VMC step with gradient clipping."""
    # Generate samples and compute gradient
    gradient = compute_gradient(wavefunction, params)

    # Clip gradient
    gradient = clip_gradient_by_norm(gradient, clip_norm)

    # Update parameters
    params = optimizer.update(params, gradient)

    return params
```

### 5.3 Learning Rate Scheduling

**Purpose:**

Improve convergence by adjusting learning rate during training.

**Scheduling Strategies:**

1. **Exponential Decay:**
   $$\alpha_t = \alpha_0 \cdot \gamma^{t}$$
   where $\gamma \in (0, 1)$ is the decay factor.

2. **Step Decay:**
   $$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / T\rfloor}$$
   Decay by factor $\gamma$ every $T$ steps.

3. **Cosine Annealing:**
   $$\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$

4. **Cyclical Learning Rates:**
   $$\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})\left(1 + \sin\left(\frac{2\pi t}{T_{\text{cycle}}}\right)\right)$$

5. **Adaptive Based on Energy Change:**
   $$\alpha_t = \alpha_{t-1} \cdot \begin{cases}
   \gamma_{\text{decrease}} & \text{if } |E_t - E_{t-1}| < \epsilon \\
   \gamma_{\text{increase}} & \text{if } \Delta E \text{ is large}
   \end{cases}$$

**Implementation:**

```python
class LearningRateScheduler:
    def __init__(self, initial_lr, decay_type='exponential', **kwargs):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.decay_type = decay_type
        self.step = 0
        self.params = kwargs

    def get_lr(self):
        """Get current learning rate."""
        if self.decay_type == 'exponential':
            gamma = self.params.get('gamma', 0.995)
            lr = self.initial_lr * (gamma ** self.step)

        elif self.decay_type == 'step':
            gamma = self.params.get('gamma', 0.5)
            decay_steps = self.params.get('decay_steps', 1000)
            lr = self.initial_lr * (gamma ** (self.step // decay_steps))

        elif self.decay_type == 'cosine':
            t_max = self.params.get('t_max', 10000)
            lr_min = self.params.get('lr_min', 1e-5)
            lr = lr_min + 0.5 * (self.initial_lr - lr_min) * \
                 (1 + np.cos(np.pi * self.step / t_max))

        return lr

    def update(self):
        """Update learning rate."""
        self.current_lr = self.get_lr()
        self.step += 1
        return self.current_lr
```

**Usage with Adam:**

```python
# Setup
lr_scheduler = LearningRateScheduler(initial_lr=1e-3, decay_type='exponential', gamma=0.995)
optimizer = AdamOptimizer(learning_rate=lr_scheduler.current_lr)

# Training loop
for step in range(n_steps):
    # Compute gradient and update
    gradient = compute_gradient(wavefunction, params)
    params = optimizer.update(params, gradient)

    # Update learning rate
    lr_scheduler.update()
    optimizer.alpha = lr_scheduler.current_lr
```

### 5.4 Advanced Optimization Techniques

**Preconditioning:**

Use natural gradient or Fisher information matrix for optimization:

$$
\theta_{t+1} = \theta_t - \alpha \mathbf{F}^{-1} \nabla_\theta E_\theta
$$

where $\mathbf{F}$ is the Fisher information matrix:

$$
\mathbf{F}_{ij} = \mathbb{E}\left[\frac{\partial \ln |\Psi|}{\partial \theta_i} \frac{\partial \ln |\Psi|}{\partial \theta_j}\right]
$$

**Stochastic Reconfiguration:**

Approximate $\mathbf{F}^{-1} \nabla E$ using iterative methods:

$$
\theta_{t+1} = \theta_t - \alpha \left(\mathbf{S} + \lambda \mathbf{I}\right)^{-1} \nabla E
$$

where $\mathbf{S}$ is the overlap matrix and $\lambda$ is a regularization parameter.

**Mini-batch Gradient Estimation:**

Use mini-batches for more efficient gradient estimation:

$$
\widehat{\nabla E_\theta} = \frac{2}{B} \sum_{b=1}^B \left[\nabla_\theta \ln |\Psi_\theta(\mathbf{R}^{(b)})| \cdot \left(E_L(\mathbf{R}^{(b)}) - \hat{E}_\theta\right)\right]
$$

where $B$ is the batch size.

**Momentum (Nesterov Accelerated Gradient):**

```python
class NesterovOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, gradient):
        """Update parameters using Nesterov momentum."""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        # Lookahead position
        params_lookahead = params + self.momentum * self.velocity

        # Compute gradient at lookahead
        gradient_lookahead = compute_gradient_at(params_lookahead)

        # Update velocity
        self.velocity = self.momentum * self.velocity - self.lr * gradient_lookahead

        # Update parameters
        params = params + self.velocity

        return params
```

---

## Numerical Considerations

### 6.1 Stability Tricks

**Log-Domain Computations:**

To avoid numerical overflow/underflow:

**Wave Function Ratio:**

Instead of $\frac{\Psi(\mathbf{R}')}{\Psi(\mathbf{R})}$, compute:

$$
\ln \frac{\Psi(\mathbf{R}')}{\Psi(\mathbf{R})} = \ln |\Psi(\mathbf{R}')| - \ln |\Psi(\mathbf{R})|
$$

Then accept if $\ln(u) < 2 \ln \frac{\Psi(\mathbf{R}')}{\Psi(\mathbf{R})}$.

**Determinant Evaluation:**

For numerical stability, use logarithms of determinants:

$$
\ln |\det \mathbf{M}| = \ln \prod_i \lambda_i = \sum_i \ln |\lambda_i|
$$

where $\lambda_i$ are the eigenvalues (or use LU decomposition with log determinant).

**Soft-Core Potentials:**

Replace singular Coulomb potentials with soft-core versions:

**Original:**
$$\frac{1}{r}$$

**Soft-core:**
$$\frac{1}{\sqrt{r^2 + \epsilon^2}}$$

where $\epsilon \approx 10^{-6}$.

**Cusp Regularization:**

Add regularization to Jastrow derivative near singularities:

$$
f'(r) = \frac{\tilde{f}'(r) + \frac{1}{2}e^{-r/\epsilon}}{1 + e^{-r/\epsilon}}
$$

where $\tilde{f}$ is the unregularized derivative.

### 6.2 Soft-Core Potentials

**Electron-Electron Interaction:**

Standard Coulomb:
$$V_{\text{ee}}(r) = \frac{1}{r}$$

Soft-core versions:

1. **Gaussian Soft-Core:**
   $$V_{\text{ee}}^{\text{sc}}(r) = \frac{1}{\sqrt{r^2 + \epsilon^2}}$$

2. **Erf Soft-Core:**
   $$V_{\text{ee}}^{\text{sc}}(r) = \frac{\text{erf}(r/\epsilon)}{r}$$

3. **Modified Coulomb:**
   $$V_{\text{ee}}^{\text{sc}}(r) = \frac{1}{\epsilon} \left(1 - e^{-r^2/\epsilon^2}\right)$$

**Electron-Nucleus Interaction:**

$$V_{\ne}(r) = -\frac{Z}{\sqrt{r^2 + \epsilon^2}}$$

**Cusp Preservation:**

To maintain correct cusp conditions with soft-core potentials:

$$
V_{\text{ee}}^{\text{cusp}}(r) = \frac{1 + f_{\text{cusp}}(r)}{\sqrt{r^2 + \epsilon^2}}
$$

where $f_{\text{cusp}}(r)$ ensures the correct cusp:

$$
\lim_{r \to 0} \frac{\partial V_{\text{ee}}^{\text{cusp}}}{\partial r} = -\frac{1}{2}
$$

**Implementation:**

```python
def soft_coulomb(r, epsilon=1e-6):
    """Soft-core Coulomb potential."""
    return 1.0 / np.sqrt(r**2 + epsilon**2)

def soft_coulomb_derivative(r, epsilon=1e-6):
    """Derivative of soft-core Coulomb."""
    denom = (r**2 + epsilon**2)**(3/2)
    return -r / denom

def cusp_preserving_coulomb(r, epsilon=1e-6):
    """Soft-core Coulomb with cusp preservation."""
    return 0.5 / (r + epsilon)
```

### 6.3 Numerical Differentiation

**Central Difference for gradients:**

$$
\frac{\partial f}{\partial x} \approx \frac{f(x + h) - f(x - h)}{2h}
$$

where $h$ is a small step (e.g., $10^{-5}$).

**Second derivative (Laplacian):**

$$
\frac{\partial^2 f}{\partial x^2} \approx \frac{f(x + h) - 2f(x) + f(x - h)}{h^2}
$$

**Analytic vs. Numerical Gradients:**

Prefer analytic gradients for:
- Efficiency (O(1) vs O(n) where n is number of parameters)
- Accuracy (no discretization error)

Use numerical gradients for:
- Debugging and verification
- Validation of analytic gradients

**Gradient Verification:**

```python
def verify_gradients(wavefunction, positions, theta, h=1e-6, tol=1e-5):
    """Verify analytic gradients against numerical gradients."""
    # Analytic gradient
    grad_analytic = wavefunction.grad_log_psi(positions, theta)

    # Numerical gradient
    grad_numeric = np.zeros_like(grad_analytic)

    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += h
        log_psi_plus = wavefunction.log_psi(positions, theta_plus)

        theta_minus = theta.copy()
        theta_minus[i] -= h
        log_psi_minus = wavefunction.log_psi(positions, theta_minus)

        grad_numeric[i] = (log_psi_plus - log_psi_minus) / (2 * h)

    # Compare
    error = np.linalg.norm(grad_analytic - grad_numeric)
    assert error < tol, f"Gradient error {error} exceeds tolerance {tol}"

    return error
```

### 6.4 Numerical Integration

**Monte Carlo Integration:**

Estimate $\int f(\mathbf{x}) p(\mathbf{x}) d\mathbf{x}$ as:

$$\frac{1}{M} \sum_{m=1}^M f(\mathbf{x}^{(m)})$$

where $\mathbf{x}^{(m)} \sim p(\mathbf{x})$.

**Importance Sampling:**

For integrals $\int f(\mathbf{x}) d\mathbf{x}$, use proposal distribution $q(\mathbf{x})$:

$$\int f(\mathbf{x}) d\mathbf{x} = \int \frac{f(\mathbf{x})}{q(\mathbf{x})} q(\mathbf{x}) d\mathbf{x} \approx \frac{1}{M} \sum_{m=1}^M \frac{f(\mathbf{x}^{(m)})}{q(\mathbf{x}^{(m)})}$$

Choose $q(\mathbf{x}) \propto f(\mathbf{x})$ for minimum variance.

### 6.5 Parallel Tempering

**Purpose:**

Escape local minima by simulating at multiple temperatures.

**Algorithm:**

Simulate $K$ replicas at temperatures $T_1 < T_2 < \ldots < T_K$:

1. **MCMC steps at each temperature:**
   $$\mathbf{R}_k^{(t+1)} \sim \text{MCMC}\left(\mathbf{R}_k^{(t)}, |\Psi|^{2/T_k}\right)$$

2. **Exchange proposals:**
   Attempt to swap configurations between adjacent temperatures $k$ and $k+1$:
   $$A_{k,k+1} = \min\left(1, \exp\left[\left(\frac{1}{T_k} - \frac{1}{T_{k+1}}\right)(2 \ln |\Psi(\mathbf{R}_{k+1})| - 2 \ln |\Psi(\mathbf{R}_k)|)\right]\right)$$

**Temperature Schedule:**

Geometric spacing:
$$T_k = T_{\min} \left(\frac{T_{\max}}{T_{\min}}\right)^{k/(K-1)}$$

### 6.6 Convergence Diagnostics

**Energy Convergence:**

Monitor energy over time and check for plateau:

$$|E_{t+1} - E_t| < \epsilon_E$$

Variance convergence:

$$\sigma_{E,t} < \epsilon_{\sigma}$$

**Parameter Convergence:**

$$\|\theta_{t+1} - \theta_t\| < \epsilon_\theta$$

**Autocorrelation Check:**

Integrated autocorrelation time should be small relative to simulation length.

**Gelman-Rubin Diagnostic:**

For multiple chains, compute:

$$\hat{R} = \sqrt{\frac{\hat{V}}{W}}$$

where:
- $W$ is the within-chain variance
- $\hat{V}$ is an estimate of the marginal posterior variance

Convergence when $\hat{R} \approx 1$.

---

## Summary of Key Formulas

| Quantity | Formula |
|----------|---------|
| **FermiNet Ansatz** | $\Psi = \Psi_{\text{Pauli}} \cdot \Psi_{\text{Jastrow}}$ |
| **Slater Determinant** | $\det \mathbf{M}^{(k)} = \sum_{\sigma \in S_N} \text{sgn}(\sigma) \prod_i M_{i,\sigma(i)}^{(k)}$ |
| **Hamiltonian** | $\hat{H} = -\frac{1}{2}\sum_i \nabla_i^2 - \sum_{i,a} \frac{Z_a}{|\mathbf{r}_i-\mathbf{R}_a|} + \sum_{i<j} \frac{1}{|\mathbf{r}_i-\mathbf{r}_j|}$ |
| **Local Energy** | $E_L = \frac{\hat{H}\Psi}{\Psi}$ |
| **Kinetic Energy** | $E_{\text{kin}} = -\frac{1}{2}\sum_i \left[\nabla_i \cdot \mathbf{F}_i + \|\mathbf{F}_i\|^2\right]$ |
| **Langevin Update** | $\mathbf{r}_i^{(t+1)} = \mathbf{r}_i^{(t)} + \tau \mathbf{F}_i^{(t)} + \sqrt{\tau} \cdot \mathcal{N}(0, 1)$ |
| **Adam Update** | $\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \odot \hat{\mathbf{m}}_t$ |
| **Energy Gradient** | $\nabla E = 2\mathbb{E}[\nabla \ln |\Psi| \cdot (E_L - E)]$ |

---

## References

1. Pfau, D., Spencer, J. S., et al. (2020). "Ab-initio solution of the many-electron Schrödinger equation with deep neural networks." *Physical Review Research*.

2. Spencer, J. S., Pfau, D., et al. (2020). "Better, faster, more accurate: The FermiNet." *arXiv preprint*.

3. Ceperley, D. M., & Alder, B. J. (1980). "Ground state of the electron gas by a Monte Carlo method." *Physical Review Letters*.

4. Umrigar, C. J., et al. (2007). "Spin-adapted spatial wave functions for three-electron systems." *Physical Review A*.

5. Foulkes, W. M. C., et al. (2001). "Quantum Monte Carlo simulations of solids." *Reviews of Modern Physics*.

---

*End of Documentation*
