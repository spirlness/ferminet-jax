# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
FermiNet is a JAX-based implementation of **Fermi Neural Network** (FermiNet) for solving the many-electron Schrodinger equation using Variational Monte Carlo (VMC).

- **Core Framework**: JAX (functional + jit/vmap/pmap).
- **Code Location**: primary implementation lives in `ferminet/`.

## Commands

### Environment
This repository uses `uv` for dependency management.

### Running Code
- **Tests**: `uv run pytest`
- **Example (Helium)**: `uv run python examples/test_helium.py`
- **Training CLI**: `uv run python -m ferminet.main --config ferminet/configs/helium.py`

### Testing
- `uv run pytest`

## Architecture (`ferminet/`)
- **`networks.py`**: functional network factory `make_fermi_net` returning `(init, apply, orbitals)`.
- **`hamiltonian.py`**: local energy and kinetic/potential terms.
- **`mcmc.py`**: Metropolis-Hastings sampling and `make_mcmc_step`.
- **`loss.py`**: VMC loss with custom JVP for unbiased gradients.
- **`train.py`**: training loop (KFAC/Adam), logging, checkpoints.

## Code Style & Guidelines
- **Paradigm**: **JAX Functional**. Avoid side effects in functions intended for JIT.
- **Immutability**: Prefer creating new objects over mutation.
- **Typing**: Use standard Python type hints (`typing`, `jax.numpy.ndarray`).
- **Docstrings**: Follow Google/DeepMind style.
- **Architecture**:
  - Keep logic in `ferminet/`.
  - Configs in `ferminet/configs/`.
  - Scripts in `examples/`.

## Known Issues
- Numerical stability depends on configs / step sizes; prefer small test configs for CI.
