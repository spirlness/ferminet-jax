# Copyright 2024 FermiNet Authors.
# Licensed under the Apache License, Version 2.0.

"""Training support utilities for FermiNet."""

from collections.abc import Callable
from typing import Any, Tuple, cast

import chex
import jax
import jax.numpy as jnp
import ml_collections

from ferminet import checkpoint, hamiltonian, networks, types
from ferminet.utils import system as system_utils

Array = jnp.ndarray
ParamTree = types.ParamTree


@chex.dataclass
class StepStats:
    """Statistics for a single training step."""

    values: Array

    @classmethod
    def from_scalars(
        cls, energy: Array, variance: Array, pmove: Array, learning_rate: Array
    ) -> "StepStats":
        """Create StepStats from scalars."""
        # Use jax.numpy to ensure we can handle both JAX arrays and numpy scalars
        # in a way compatible with JAX transformations.
        return cls(values=jnp.stack([energy, variance, pmove, learning_rate], axis=-1))

    @property
    def energy(self) -> Array:
        return self.values[..., 0]

    @property
    def variance(self) -> Array:
        return self.values[..., 1]

    @property
    def pmove(self) -> Array:
        return self.values[..., 2]

    @property
    def learning_rate(self) -> Array:
        return self.values[..., 3]


def build_network(
    cfg: ml_collections.ConfigDict,
    atoms: Array,
    charges: Array,
    spins: Tuple[int, int],
) -> tuple[types.InitFermiNet, types.LogFermiNetLike, dict[str, Any]]:
    """Create network init/apply functions."""
    factory_candidates = [
        "make_ferminet",
        "make_fermi_net",
        "make_network",
        "build_network",
    ]
    factory = None
    for name in factory_candidates:
        if hasattr(networks, name):
            factory = getattr(networks, name)
            break
    if factory is None:
        raise AttributeError("Could not find a network factory in ferminet.networks.")

    result = factory(atoms, charges, spins, cfg)
    if isinstance(result, tuple):
        init_fn = result[0]
        apply_fn = cast(Callable[..., Any], result[1])
        extras = {"factory": factory.__name__, "extras": result[2:]}
    else:
        raise TypeError("Network factory must return at least (init_fn, apply_fn)")

    def apply_log(
        params: ParamTree,
        positions: Array,
        spins_arr: Array,
        atoms_arr: Array,
        charges_arr: Array,
    ) -> Array:
        out = apply_fn(params, positions, spins_arr, atoms_arr, charges_arr)
        if isinstance(out, tuple) and len(out) == 2:
            return cast(Array, out[1])
        return cast(Array, out)

    return (
        cast(types.InitFermiNet, init_fn),
        cast(types.LogFermiNetLike, apply_log),
        extras,
    )


def make_local_energy_fn(
    apply_log: types.LogFermiNetLike,
    charges: Array,
    spins: Tuple[int, int],
    cfg: ml_collections.ConfigDict,
) -> Callable[[ParamTree, jax.Array, types.FermiNetData], Array]:
    """Create batched local energy function."""
    cfg_any = cast(Any, cfg)

    def apply_sign_log(
        params: ParamTree,
        positions: Array,
        spins_arr: Array,
        atoms_arr: Array,
        charges_arr: Array,
    ) -> tuple[Array, Array]:
        log_psi = apply_log(params, positions, spins_arr, atoms_arr, charges_arr)
        return jnp.ones_like(log_psi), log_psi

    single_local_energy = hamiltonian.local_energy(
        apply_sign_log,
        charges=charges,
        nspins=spins,
        use_scan=cfg_any.optim.get("laplacian", "default") == "scan",
        complex_output=cfg_any.network.get("complex", False),
    )

    def local_energy_fn(
        params: ParamTree,
        key: jax.Array,
        data: types.FermiNetData,
    ) -> Array:
        def per_config(pos: Array) -> Array:
            sample = types.FermiNetData(
                positions=pos,
                spins=data.spins,
                atoms=data.atoms,
                charges=data.charges,
            )
            energy, _ = single_local_energy(params, key, sample)
            return energy

        return jax.vmap(per_config)(data.positions)

    return local_energy_fn


def prepare_system(
    cfg: ml_collections.ConfigDict,
) -> tuple[Array, Array, Tuple[int, int], int]:
    """Extract system parameters from config."""
    cfg_any = cast(Any, cfg)
    system_cfg = cfg_any.system
    if system_cfg.molecule:
        atoms, charges = system_utils.parse_molecule(
            system_cfg.molecule,
            units=system_cfg.get("units", "bohr"),
        )
    else:
        raise ValueError("System configuration requires a molecule definition.")

    if system_cfg.electrons:
        spins = tuple(system_cfg.electrons)
    else:
        spins = system_utils.get_spin_config(
            charges,
            spin_polarized=system_cfg.get("spin_polarized", False),
        )
    ndim = int(system_cfg.get("ndim", 3))
    return atoms, charges, spins, ndim


def init_mcmc_data(
    key: jax.Array,
    atoms: Array,
    charges: Array,
    spins: Tuple[int, int],
    batch_size: int,
    ndim: int,
) -> types.FermiNetData:
    """Initialize MCMC walker positions around atoms."""
    n_electrons = sum(spins)
    key, subkey = jax.random.split(key)
    positions = jax.random.normal(subkey, (batch_size, n_electrons * ndim))
    if atoms.shape[0] > 0:
        for i in range(n_electrons):
            atom_idx = i % atoms.shape[0]
            positions = positions.at[:, i * ndim : (i + 1) * ndim].add(
                atoms[atom_idx : atom_idx + 1]
            )
    spins_arr = jnp.array([0] * spins[0] + [1] * spins[1])
    return types.FermiNetData(
        positions=positions,
        spins=spins_arr,
        atoms=atoms,
        charges=charges,
    )


def restore_checkpoint(
    cfg: ml_collections.ConfigDict,
    params: ParamTree,
    opt_state: Any,
    data: types.FermiNetData,
    step: int,
) -> tuple[ParamTree, Any, types.FermiNetData, int]:
    """Restore state from checkpoint if available."""
    cfg_any = cast(Any, cfg)
    restore_path = cfg_any.log.get("restore_path", None)
    if restore_path:
        ckpt = checkpoint.restore_checkpoint(restore_path)
        return ckpt.params, ckpt.opt_state, ckpt.mcmc_state, ckpt.step

    if cfg_any.log.get("restore", False):
        latest = checkpoint.find_latest_checkpoint(cfg_any.log.save_path)
        if latest is not None:
            _, ckpt = latest
            return ckpt.params, ckpt.opt_state, ckpt.mcmc_state, ckpt.step
    return params, opt_state, data, step


def log_stats(
    step: int,
    stats: StepStats,
    walltime: float,
    width: float | None = None,
) -> None:
    """Log training step statistics to console."""
    if width is not None:
        print(
            "Step {:>8d} | E {:>12.6f} | Var {:>10.6f} | pmove {:>6.3f} | "
            "width {:>6.3f} | lr {:>8.5f} | {:.2f} s".format(
                step,
                float(stats.energy),
                float(stats.variance),
                float(stats.pmove),
                width,
                float(stats.learning_rate),
                walltime,
            )
        )
    else:
        print(
            "Step {:>8d} | E {:>12.6f} | Var {:>10.6f} | pmove {:>6.3f} | "
            "lr {:>8.5f} | {:.2f} s".format(
                step,
                float(stats.energy),
                float(stats.variance),
                float(stats.pmove),
                float(stats.learning_rate),
                walltime,
            )
        )
