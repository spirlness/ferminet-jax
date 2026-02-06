# FERMINET-JAX KNOWLEDGE BASE

**Generated:** 2026-02-06  
**Commit:** 50de595  
**Branch:** master

## OVERVIEW

JAX-based FermiNet for solving many-electron Schrodinger equation via VMC. Functional `init/apply` pattern throughout.

## STRUCTURE

```
ferminet-jax/
├── src/ferminet/          # Core library (see AGENTS.md there)
│   ├── configs/           # System configurations (see AGENTS.md there)
│   └── utils/             # Statistics, system parsing
├── tests/                 # pytest suite (21 files)
├── examples/              # Runnable demo: test_helium.py
├── scripts/               # detect_accelerator.py
├── checkpoints/           # Training outputs (gitignore-worthy)
└── results/               # Experiment outputs
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Network architecture | `src/ferminet/networks.py` | `make_fermi_net` factory |
| Local energy / Hamiltonian | `src/ferminet/hamiltonian.py` | Kinetic + potential |
| MCMC sampling | `src/ferminet/mcmc.py` | `make_mcmc_step`, MH updates |
| VMC loss + gradients | `src/ferminet/loss.py` | Custom JVP for unbiased estimator |
| Training loop | `src/ferminet/train.py` | KFAC/Adam, checkpointing |
| Add new atom/molecule | `src/ferminet/configs/` | Copy helium.py, modify |
| Envelope functions | `src/ferminet/envelopes.py` | isotropic/diagonal/full |

## CONVENTIONS

- **Functional JAX**: No class-based state. Functions for JIT.
- **Immutability**: Create new objects, never mutate.
- **init/apply pattern**: Factories return `(init_fn, apply_fn, ...)`.
- **Type hints**: Standard Python typing, `jax.Array` for arrays.
- **Docstrings**: Google/DeepMind style.

## ANTI-PATTERNS

| Forbidden | Why |
|-----------|-----|
| Class-based network state | Breaks JIT, use functional pattern |
| `as any` type suppressions | Use proper typing |
| Dotted-path FieldReference | Use `cfg.get_ref()` explicitly |
| Zero electron distances | Produces NaN; use epsilon stabilization |
| Exact e-e collisions in tests | Causes NaN gradients |

## CRITICAL IMPLEMENTATION NOTES

- **KFAC always uses `multi_device=True`** per upstream implementation.
- **Custom JVP in loss.py** provides unbiased gradient estimator; don't replace with naive autodiff.
- **Envelope softplus stabilization** prevents NaN at zero distance.
- **MCMC width auto-adapts** to maintain pmove 0.5-0.6.

## COMMANDS

```bash
# Environment (uses uv)
uv sync --dev

# Run tests
uv run pytest

# Quick example
uv run python examples/test_helium.py

# Full training
uv run python -m ferminet.main --config src/ferminet/configs/helium.py

# GPU setup
uv pip install "jax[cuda12]"
```

## STABILITY NOTES

- NaN recovery: Training skips param update if energy is NaN.
- First KFAC compile: 10-20 min (normal).
- Reduce `batch_size` first if OOM.
- Stable params: `lr=0.005`, `damping=0.01`, `clip_local_energy=3.0`.
