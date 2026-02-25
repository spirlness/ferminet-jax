"""Base configuration for FermiNet experiments."""

# pyright: reportMissingImports=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false
# pyright: reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
# pyright: reportOptionalMemberAccess=false, reportOptionalSubscript=false
# pyright: reportCallIssue=false, reportIndexIssue=false
# pyright: reportArgumentType=false

import enum

import ml_collections
from ml_collections import config_dict


class SystemType(enum.IntEnum):
    """Enumerates supported physical system types."""

    MOLECULE = enum.auto()


def default() -> ml_collections.ConfigDict:
    """Returns a ConfigDict with base FermiNet settings."""
    cfg = ml_collections.ConfigDict(
        {
            "batch_size": 4096,
            "config_module": __name__,
            "optim": {
                "objective": "vmc",
                "iterations": 1_000_000,
                "optimizer": "kfac",
                "laplacian": "default",
                "lr": {
                    "rate": 0.05,
                    "decay": 1.0,
                    "delay": 10000.0,
                },
                "clip_local_energy": 5.0,
                "clip_median": False,
                "center_at_clip": True,
                "reset_if_nan": False,
                "kfac": {
                    "damping": 0.001,
                    "momentum": 0.0,
                    "momentum_type": "regular",
                    "norm_constraint": 0.001,
                    "mean_center": True,
                    "l2_reg": 0.0,
                    "invert_every": 1,
                    "cov_update_every": 1,
                    "estimator": "fisher_gradients",
                    "register_only_generic": False,
                    "pmap_axis_name": "devices",
                    # Filled with a FieldReference after cfg construction.
                    "batch_size": 4096,
                },
                "adam": {
                    "b1": 0.9,
                    "b2": 0.999,
                    "eps": 1.0e-8,
                },
            },
            "log": {
                "stats_every": 100,
                "summary_every": 100,
                "checkpoint_every": 10000,
                "save_path": "./checkpoints",
                "restore": False,
                "restore_path": None,
                "save_all": False,
                "print_every": 10,
                "history_every": 10,
                "max_summary_length": 50,
            },
            "system": {
                "type": SystemType.MOLECULE.value,
                "molecule": config_dict.placeholder(list),
                "electrons": tuple(),
                "ndim": 3,
                "states": 0,
                "units": "bohr",
                "charges": tuple(),
                "spin_polarized": False,
                "use_pp": False,
                "pp_symbols": tuple(),
                "pp_type": "hf",
                "use_electronic_potential": False,
                "use_localization": False,
            },
            "mcmc": {
                "burn_in": 100,
                "steps": 10,
                "init_width": 1.0,
                "move_width": 0.02,
                "adapt_frequency": 100,
                "target_acceptance": 0.5,
                "use_langevin": True,
                "mass": 1.0,
            },
            "network": {
                "network_type": "ferminet",
                "determinants": 16,
                "complex": False,
                "envelope_type": "isotropic",
                "full_det": True,
                "use_slogdet": False,
                "ferminet": {
                    "hidden_dims": ((256, 32), (256, 32), (256, 32), (256, 32)),
                    # Filled with FieldReferences after cfg construction.
                    "determinants": 16,
                    "full_det": True,
                    "use_slogdet": False,
                    "envelope_type": "isotropic",
                    "bias_orbitals": True,
                    "hidden_activation": "tanh",
                    "use_last_layer": True,
                    "rescale_inputs": True,
                    "envelope_softplus": 1.0,
                },
                "envelope": {
                    "type": "isotropic",
                    "isotropic": {
                        "sigma": 1.0,
                    },
                    "full": {
                        "sigma": 1.0,
                    },
                },
                "jastrow": {
                    "use": False,
                    "type": "one_body",
                    "trainable": True,
                    "scalar_only": True,
                    "hidden_dims": (16, 16),
                },
            },
            "pretrain": {
                "method": "none",
                "iterations": 1000,
                # Filled with a FieldReference after cfg construction.
                "batch_size": 4096,
                "hf_steps": 20,
                "basis": "sto-3g",
                "atoms": config_dict.placeholder(list),
            },
            "debug": {
                "determinant_mode": False,
                "log_psi": False,
                "use_jit": True,
                "pmap_axis_name": "devices",
                "seed": 0,
                "print_every": 10,
                "dump_unstable": False,
                "check_nan": False,
                "debug_nans": False,
            },
        }
    )

    # ml_collections FieldReference expects a reference object created via get_ref.
    # Dotted-path strings (e.g. "network.determinants") do not resolve.
    cfg.optim.kfac["batch_size"] = cfg.get_ref("batch_size")
    cfg.pretrain["batch_size"] = cfg.get_ref("batch_size")

    cfg.network.ferminet["determinants"] = cfg.network.get_ref("determinants")
    cfg.network.ferminet["full_det"] = cfg.network.get_ref("full_det")
    cfg.network.ferminet["use_slogdet"] = cfg.network.get_ref("use_slogdet")
    cfg.network.ferminet["envelope_type"] = cfg.network.get_ref("envelope_type")
    return cfg


def resolve(cfg: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
    """Resolves FieldReferences in the provided configuration."""
    return cfg.copy_and_resolve_references()


# ── Valid values for key configuration fields ─────────────────────────────
_VALID_OPTIMIZERS = {"kfac", "adam"}
_VALID_LAPLACIAN_MODES = {"default", "scan", "forward_over_reverse"}
_VALID_ENVELOPE_TYPES = {"isotropic", "diagonal", "full"}


def validate_config(cfg: ml_collections.ConfigDict) -> None:
    """Validate core configuration fields before training.

    Raises ``ValueError`` for clearly invalid settings so that problems are
    caught at startup rather than deep inside a training loop.
    """
    from typing import Any, cast

    c = cast(Any, cfg)

    # ── batch_size ─────────────────────────────────────────────────────────
    batch_size = int(c.batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    # ── optimizer ──────────────────────────────────────────────────────────
    optimizer = str(c.optim.optimizer)
    if optimizer not in _VALID_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer '{optimizer}'. "
            f"Choose from: {sorted(_VALID_OPTIMIZERS)}"
        )

    # ── laplacian mode ─────────────────────────────────────────────────────
    laplacian = str(c.optim.get("laplacian", "default"))
    if laplacian not in _VALID_LAPLACIAN_MODES:
        raise ValueError(
            f"Unsupported laplacian mode '{laplacian}'. "
            f"Choose from: {sorted(_VALID_LAPLACIAN_MODES)}"
        )

    # ── hidden_dims ────────────────────────────────────────────────────────
    hidden_dims = c.network.ferminet.hidden_dims
    if not isinstance(hidden_dims, (list, tuple)):
        raise ValueError("network.ferminet.hidden_dims must be a list/tuple of pairs")
    for i, pair in enumerate(hidden_dims):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"hidden_dims[{i}] must be a (one_dim, two_dim) pair, got {pair}"
            )

    # ── envelope type ──────────────────────────────────────────────────────
    envelope = str(c.network.get("envelope_type", "isotropic")).lower()
    if envelope not in _VALID_ENVELOPE_TYPES:
        raise ValueError(
            f"Unsupported envelope_type '{envelope}'. "
            f"Choose from: {sorted(_VALID_ENVELOPE_TYPES)}"
        )

    # ── iterations ─────────────────────────────────────────────────────────
    iterations = int(c.optim.iterations)
    if iterations <= 0:
        raise ValueError(f"optim.iterations must be > 0, got {iterations}")
