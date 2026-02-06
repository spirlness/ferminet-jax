# CONFIGURATION FILES

## OVERVIEW

Python-based configs for different atomic/molecular systems.

## AVAILABLE CONFIGS

| Config | System | batch_size | GPU Memory |
|--------|--------|------------|------------|
| `helium_quick.py` | He | 256 | <2 GB |
| `helium.py` | He | 4096 | ~3 GB |
| `helium_max.py` | He | 4096 | ~4 GB |
| `helium_scaled.py` | He | 8192 | ~6 GB |
| `hydrogen.py` | H | 2048 | ~2 GB |
| `lithium.py` | Li | 4096 | ~4 GB |

## ADDING A NEW SYSTEM

```python
from ferminet import base_config

def get_config():
    cfg = base_config.default()
    
    cfg.system.molecule = [("Li", (0.0, 0.0, 0.0))]
    cfg.system.electrons = (2, 1)  # (n_up, n_down)
    cfg.system.charges = (3,)
    
    cfg.batch_size = 4096
    cfg.network.determinants = 16
    
    return cfg
```

## KEY PARAMETERS

| Parameter | Effect | Tune When |
|-----------|--------|-----------|
| `batch_size` | Memory, statistics | OOM or high variance |
| `network.determinants` | Expressivity | Poor convergence |
| `network.ferminet.hidden_dims` | Capacity | Larger systems |
| `optim.lr.rate` | Step size | Instability |
| `optim.kfac.damping` | Regularization | NaN issues |
| `mcmc.move_width` | Proposal scale | pmove out of range |

## CONVENTIONS

- Each config exports `get_config() -> ConfigDict`.
- Inherit from `base_config.default()`.
- Use FieldReference for linked params: `cfg.get_ref("batch_size")`.
