# FERMINET CORE MODULE

## OVERVIEW

Core FermiNet implementation: networks, Hamiltonian, MCMC, loss, training.

## MODULE MAP

| File | Purpose | Key Functions |
|------|---------|---------------|
| `networks.py` | FermiNet architecture | `make_fermi_net` → (init, apply, orbitals) |
| `hamiltonian.py` | Local energy | `local_energy`, `local_kinetic_energy` |
| `mcmc.py` | Metropolis-Hastings | `make_mcmc_step`, `mh_update` |
| `loss.py` | VMC loss + custom JVP | `make_loss`, `clip_local_energy` |
| `train.py` | Training loop | `train()`, KFAC/Adam setup |
| `base_config.py` | Default ConfigDict | `default()`, `resolve()` |
| `envelopes.py` | Decay envelopes | `make_isotropic_envelope`, etc. |
| `jastrows.py` | Jastrow factors | `get_jastrow` |
| `network_blocks.py` | Linear layers | `init_linear_layer`, `linear_layer` |
| `checkpoint.py` | Save/restore | `save_checkpoint`, `restore_checkpoint` |
| `types.py` | Type definitions | `FermiNetData`, `ParamTree`, protocols |
| `constants.py` | pmap helpers | `pmap`, `pmean`, `all_gather` |
| `pretrain.py` | HF pretraining | Optional module |
| `main.py` | CLI entry | `python -m ferminet.main` |

## CONVENTIONS

- All network functions: pure, JIT-compatible.
- Factories return tuples: `(init_fn, apply_fn, ...)`.
- Data flows via `FermiNetData` NamedTuple.
- Config via `ml_collections.ConfigDict`.

## ANTI-PATTERNS

- Don't use class instances for network state.
- Don't call `jax.grad` on functions with side effects.
- Don't assume single-device; use `constants.pmap`.

## DEPENDENCIES

Internal only: `network_blocks`, `envelopes`, `types`, `constants`.  
External: `jax`, `optax`, `kfac_jax`, `ml_collections`, `chex`.
