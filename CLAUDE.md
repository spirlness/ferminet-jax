# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
FermiNet is a JAX-based implementation of **Fermi Neural Network** (FermiNet) for solving the many-electron Schrödinger equation using Variational Monte Carlo (VMC).
- **Core Framework**: JAX (Functional programming, JIT compilation).
- **Current Status**: "Stage 2" development (Multi-determinant, residual connections).
- **Goal**: Achieve chemical accuracy for molecular energy calculations.

## Commands

### Environment
There is no package manager configuration. Dependencies are installed in the environment: `jax`, `jax.numpy`, `numpy`.
Note: The project root is `G:\FermiNet`. Scripts assume `src` is in the python path or handle it manually.

### Running Code
- **Training (Stage 1 - Stable)**: `python examples/train_stable.py`
- **Training (Stage 2 - Quick)**: `python examples/train_stage2_quick.py`
- **Training (Stage 2 - Full)**: `python examples/train_stage2.py`
- **Demo**: `python examples/demo.py`

### Testing
- **Stage 2 Components**: `python tests/test_stage2.py`
- **Network Stability**: `python tests/test_network_stability.py`
- **Energy Calculation**: `python tests/test_energy_quick.py`
- **Extended Debug**: `python tests/test_extended_debug.py`

## Architecture (`src/ferminet/`)
- **`network.py`**: Defines `ExtendedFermiNet` (inputs electron positions -> outputs log wavefunction). Uses `MultiDeterminantOrbitals` and `JastrowFactor`.
- **`physics.py`**: Implements Hamiltonian (Kinetic + Potential energy) and Local Energy. Uses `jax.grad` and `jax.vmap`.
- **`mcmc.py`**: Metropolis-Hastings sampler with Langevin dynamics.
- **`trainer.py`**: `VMCTrainer` and `ExtendedTrainer` for the optimization loop (gradient clipping, learning rate scheduling).

## Code Style & Guidelines
- **Paradigm**: **JAX Functional**. Avoid side effects in functions intended for JIT.
- **Immutability**: Prefer creating new objects over mutation.
- **Typing**: Use standard Python type hints (`typing`, `jax.numpy.ndarray`).
- **Docstrings**: Follow Google/DeepMind style.
- **Architecture**:
  - Keep logic in `src/`.
  - Configs in `configs/` (as Python dictionaries).
  - Scripts in `examples/`.
- **Note on PyTorch**: Ignore `src/ferminet/residual_layers.py` and `scheduler.py` if they contain PyTorch code; the project is JAX-based.

## Known Issues
- **Numerical Stability**: Stage 2 training can be unstable (NaNs, exploding gradients). Use `ExtendedTrainer` with gradient clipping and lower learning rates.
- **Performance**: Large batches (2048+) may require JIT compilation of critical paths.
