"""Training loop for FermiNet VMC with KFAC/Adam optimizers."""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional, Tuple, cast

import chex
import jax
import jax.numpy as jnp
import kfac_jax
import ml_collections
import optax

from ferminet import (
    base_config,
    checkpoint,
    constants,
    hamiltonian,
    loss,
    mcmc,
    networks,
    types,
)
from ferminet.utils import system as system_utils

Array = jnp.ndarray
ParamTree = types.ParamTree


@chex.dataclass
class TrainState:
    params: ParamTree
    opt_state: Any
    data: types.FermiNetData
    key: jax.Array
    step: jnp.ndarray


@chex.dataclass
class StepStats:
    energy: Array
    variance: Array
    pmove: Array
    learning_rate: Array


def make_schedule(cfg: ml_collections.ConfigDict) -> optax.Schedule:
    """Builds warmup + inverse decay schedule."""
    cfg_any = cast(Any, cfg)
    lr_cfg = cfg_any.optim.lr
    return optax.warmup_exponential_decay_schedule(
        init_value=0.0,
        peak_value=lr_cfg.rate,
        warmup_steps=int(lr_cfg.get("warmup_steps", 1000)),
        transition_steps=1,
        decay_rate=lr_cfg.decay,
        transition_begin=int(lr_cfg.delay),
    )


def _filter_kwargs(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(fn)
    allowed = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in allowed}


def _replicate_tree(tree: Any, devices: Sequence[jax.Device]) -> Any:
    del devices
    return kfac_jax.utils.replicate_all_local_devices(tree)


def _shard_array(arr: Array, device_count: int) -> Array:
    if arr.shape[0] % device_count != 0:
        raise ValueError(
            "Batch dimension must be divisible by device count."
            f" batch={arr.shape[0]} devices={device_count}"
        )
    per_device = arr.shape[0] // device_count
    new_shape = (device_count, per_device) + arr.shape[1:]
    reshaped = jnp.reshape(arr, new_shape)
    return kfac_jax.utils.broadcast_all_local_devices(reshaped)


def _p_split(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    return kfac_jax.utils.p_split(key)


def _shard_data(
    data: types.FermiNetData,
    device_count: int,
) -> types.FermiNetData:
    return types.FermiNetData(
        positions=_shard_array(jnp.asarray(data.positions), device_count),
        spins=_replicate_tree(jnp.asarray(data.spins), jax.devices()),
        atoms=_replicate_tree(jnp.asarray(data.atoms), jax.devices()),
        charges=_replicate_tree(jnp.asarray(data.charges), jax.devices()),
    )


def _build_network(
    cfg: ml_collections.ConfigDict,
    atoms: Array,
    charges: Array,
    spins: Tuple[int, int],
) -> tuple[types.InitFermiNet, types.LogFermiNetLike, dict[str, Any]]:
    """Create network init/apply functions from the networks module."""
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
        raise AttributeError(
            "Could not find a network factory in ferminet.networks. "
            "Expected one of: make_ferminet, make_fermi_net, make_network, build_network."
        )

    result = factory(atoms, charges, spins, cfg)
    if isinstance(result, tuple):
        init_fn = result[0]
        apply_fn = cast(Callable[..., Any], result[1])
        extras = {"factory": factory.__name__, "extras": result[2:]}
    else:
        raise TypeError("Network factory must return at least (init_fn, apply_fn)")

    def apply_sign_log(
        params: ParamTree,
        positions: Array,
        spins_arr: Array,
        atoms_arr: Array,
        charges_arr: Array,
    ) -> tuple[Array, Array]:
        out = apply_fn(params, positions, spins_arr, atoms_arr, charges_arr)
        if isinstance(out, tuple) and len(out) == 2:
            sign = cast(Array, out[0])
            log_psi = cast(Array, out[1])
            return sign, log_psi
        log_psi = cast(Array, out)
        return jnp.ones_like(log_psi), log_psi

    def apply_log(
        params: ParamTree,
        positions: Array,
        spins_arr: Array,
        atoms_arr: Array,
        charges_arr: Array,
    ) -> Array:
        return apply_sign_log(params, positions, spins_arr, atoms_arr, charges_arr)[1]

    return (
        cast(types.InitFermiNet, init_fn),
        cast(types.LogFermiNetLike, apply_log),
        extras,
    )


def _find_dense_params(
    params: Any,
    in_dim: int,
) -> list[tuple[Array, Optional[Array]]]:
    """Search for (w, b) pairs with matching input dimension."""
    dense_params: list[tuple[Array, Optional[Array]]] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            w = None
            b = None
            if "w" in node:
                w = node["w"]
            elif "weight" in node:
                w = node["weight"]
            if "b" in node:
                b = node["b"]
            elif "bias" in node:
                b = node["bias"]
            if w is not None and isinstance(w, jnp.ndarray) and w.ndim == 2:
                if w.shape[0] == in_dim:
                    dense_params.append((cast(Array, w), cast(Optional[Array], b)))
            for value in node.values():
                visit(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                visit(value)

    visit(params)
    return dense_params


def _register_kfac_dense(params: ParamTree, inputs: Array) -> bool:
    """Register a dense layer for KFAC using matching parameters."""
    dense_params = _find_dense_params(params, inputs.shape[-1])
    if not dense_params:
        return False
    w, b = dense_params[0]
    outputs = jnp.dot(inputs, w)
    if b is not None:
        outputs = outputs + b
    kfac_jax.register_dense(inputs, outputs, w, b)
    return True


def _make_local_energy_fn(
    apply_sign_log: Callable[..., tuple[Array, Array]],
    charges: Array,
    spins: Tuple[int, int],
    cfg: ml_collections.ConfigDict,
) -> Callable[[ParamTree, jax.Array, types.FermiNetData], Array]:
    cfg_any = cast(Any, cfg)
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


def make_kfac_optimizer(
    cfg: ml_collections.ConfigDict,
    loss_fn: Callable[[ParamTree, jax.Array, types.FermiNetData], tuple[Array, Any]],
) -> tuple[kfac_jax.Optimizer, Callable[..., Any], Callable[..., Any]]:
    """Create KFAC optimizer init and update functions.

    When multi_device=True (multiple devices), KFAC internally uses pmap and
    expects replicated inputs. When multi_device=False (single device), KFAC
    works on single-device inputs directly.

    Returns:
        Tuple of (optimizer, init_fn, update_fn) where optimizer is the KFAC
        optimizer instance for direct use in the training loop.
    """
    cfg_any = cast(Any, cfg)

    val_and_grad = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

    kfac_cfg = cfg_any.optim.kfac
    lr_cfg = cfg_any.optim.lr

    def learning_rate_schedule(t: jnp.ndarray) -> jnp.ndarray:
        return lr_cfg.rate * jnp.power((1.0 / (1.0 + (t / lr_cfg.delay))), lr_cfg.decay)

    # KFAC always uses multi_device=True as per official FermiNet implementation.
    # It handles internal pmap and expects replicated inputs.
    optimizer = kfac_jax.Optimizer(
        value_and_grad_func=val_and_grad,
        l2_reg=kfac_cfg.l2_reg,
        norm_constraint=kfac_cfg.norm_constraint,
        value_func_has_aux=True,
        value_func_has_rng=True,
        learning_rate_schedule=learning_rate_schedule,
        curvature_ema=kfac_cfg.get("cov_ema_decay", 0.95),
        inverse_update_period=kfac_cfg.invert_every,
        min_damping=kfac_cfg.get("min_damping", 1e-4),
        num_burnin_steps=0,
        register_only_generic=kfac_cfg.register_only_generic,
        estimation_mode="fisher_exact",
        multi_device=True,
        pmap_axis_name=constants.PMAP_AXIS_NAME,
    )

    def init_fn(params: ParamTree, key: jax.Array, data: types.FermiNetData) -> Any:
        return optimizer.init(params, key, data)

    def update_fn(
        params: ParamTree,
        opt_state: Any,
        key: jax.Array,
        data: types.FermiNetData,
        step: jnp.ndarray,
    ) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        del step
        shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
        shared_damping = kfac_jax.utils.replicate_all_local_devices(
            jnp.asarray(kfac_cfg.damping)
        )
        new_params, new_state, stats = optimizer.step(
            params=params,
            state=opt_state,
            rng=key,
            batch=data,
            momentum=shared_mom,
            damping=shared_damping,
        )
        loss_value = stats.get("loss", jnp.asarray(0.0))
        aux = stats.get("aux", None)
        return new_params, new_state, loss_value, aux, stats

    return optimizer, init_fn, update_fn


def make_adam_optimizer(
    cfg: ml_collections.ConfigDict,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Create Adam optimizer init and update functions."""
    cfg_any = cast(Any, cfg)
    schedule = make_schedule(cfg)
    optimizer = optax.adam(
        learning_rate=schedule,
        b1=cfg_any.optim.adam.b1,
        b2=cfg_any.optim.adam.b2,
        eps=cfg_any.optim.adam.eps,
    )

    def init_fn(params: ParamTree) -> Any:
        return optimizer.init(params)

    def update_fn(
        params: ParamTree,
        opt_state: Any,
        key: jax.Array,
        data: types.FermiNetData,
        step: jnp.ndarray,
        loss_fn: Callable[
            [ParamTree, jax.Array, types.FermiNetData], tuple[Array, Any]
        ],
    ) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        (loss_value, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, key, data
        )
        grads = constants.pmean(grads)
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        stats = {"grad_norm": optax.global_norm(grads)}
        return new_params, new_state, loss_value, aux, stats

    return init_fn, update_fn


def make_optimizer(
    cfg: ml_collections.ConfigDict,
    loss_fn: Callable[[ParamTree, jax.Array, types.FermiNetData], tuple[Array, Any]],
    params: ParamTree,
) -> tuple[Optional[kfac_jax.Optimizer], Callable[..., Any], Callable[..., Any]]:
    cfg_any = cast(Any, cfg)
    if cfg_any.optim.optimizer == "kfac":
        return make_kfac_optimizer(cfg, loss_fn)
    init_fn, update_fn = make_adam_optimizer(cfg)
    return None, init_fn, update_fn


def _prepare_system(
    cfg: ml_collections.ConfigDict,
) -> tuple[Array, Array, Tuple[int, int], int]:
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


def _init_mcmc_data(
    key: jax.Array,
    atoms: Array,
    charges: Array,
    spins: Tuple[int, int],
    batch_size: int,
    ndim: int,
) -> types.FermiNetData:
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


def _restore_checkpoint(
    cfg: ml_collections.ConfigDict,
    params: ParamTree,
    opt_state: Any,
    data: types.FermiNetData,
    step: int,
) -> tuple[ParamTree, Any, types.FermiNetData, int]:
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


def _log_stats(
    step: int,
    stats: StepStats,
    walltime: float,
    width: float | None = None,
) -> None:
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


def train(cfg: ml_collections.ConfigDict) -> Mapping[str, Any]:
    """Run VMC training with KFAC or Adam optimizer."""
    cfg = base_config.resolve(cfg)
    cfg_any = cast(Any, cfg)
    key = jax.random.PRNGKey(cfg_any.debug.get("seed", 0))

    atoms, charges, spins, ndim = _prepare_system(cfg)
    batch_size = int(cfg_any.batch_size)

    init_fn, apply_log, _ = _build_network(cfg, atoms, charges, spins)

    def apply_sign_log(
        params: ParamTree,
        positions: Array,
        spins_arr: Array,
        atoms_arr: Array,
        charges_arr: Array,
    ) -> tuple[Array, Array]:
        log_psi = apply_log(params, positions, spins_arr, atoms_arr, charges_arr)
        return jnp.ones_like(log_psi), log_psi

    local_energy_fn = _make_local_energy_fn(apply_sign_log, charges, spins, cfg)
    loss_fn = loss.make_loss(
        apply_log,
        local_energy_fn,
        clip_local_energy=cfg_any.optim.clip_local_energy,
    )

    key, init_key, data_key = jax.random.split(key, 3)
    params = init_fn(init_key)
    data = _init_mcmc_data(data_key, atoms, charges, spins, batch_size, ndim)

    devices = jax.devices()
    device_count = len(devices)
    if batch_size % device_count != 0:
        raise ValueError(
            f"Batch size {batch_size} must be divisible by device count {device_count}."
        )
    batch_per_device = batch_size // device_count

    data = _shard_data(data, device_count)
    params = _replicate_tree(params, devices)
    key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

    kfac_optimizer, init_opt_state, update_fn = make_optimizer(cfg, loss_fn, params)
    if cfg_any.optim.optimizer == "kfac":
        key, init_keys = _p_split(key)
        opt_state = init_opt_state(params, init_keys, data)
    else:
        opt_state = jax.pmap(init_opt_state)(params)

    params, opt_state, data, step = _restore_checkpoint(
        cfg, params, opt_state, data, step=0
    )
    step_array = jnp.full((device_count,), step, dtype=jnp.int32)

    mcmc_step = mcmc.make_mcmc_step(
        apply_log,
        batch_per_device=batch_per_device,
        steps=int(cfg_any.mcmc.steps),
        atoms=atoms if cfg_any.mcmc.use_langevin else None,
        ndim=ndim,
    )

    schedule = make_schedule(cfg)
    width = float(cfg_any.mcmc.move_width)
    pmoves = jnp.zeros(int(cfg_any.mcmc.adapt_frequency))

    def _loss_with_kfac(
        params: ParamTree,
        key: jax.Array,
        data: types.FermiNetData,
    ) -> tuple[Array, Any]:
        if cfg_any.optim.optimizer == "kfac":
            _register_kfac_dense(params, data.positions)
        return loss_fn(params, key, data)

    if cfg_any.optim.optimizer == "kfac":
        pmapped_mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
        shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
        shared_damping = kfac_jax.utils.replicate_all_local_devices(
            jnp.asarray(cfg_any.optim.kfac.damping)
        )

        def kfac_step_fn(
            params: ParamTree,
            opt_state: Any,
            data: types.FermiNetData,
            key: jax.Array,
            step: jnp.ndarray,
            mcmc_width: Any,
        ) -> tuple[Any, Any, Any, Any, StepStats]:
            mcmc_keys, loss_keys = _p_split(key)
            new_data, pmove = pmapped_mcmc_step(params, data, mcmc_keys, mcmc_width)

            new_params, new_opt_state, stats = kfac_optimizer.step(
                params=params,
                state=opt_state,
                rng=loss_keys,
                batch=new_data,
                momentum=shared_mom,
                damping=shared_damping,
            )
            loss_value = stats.get("loss", jnp.asarray(0.0))
            aux = stats.get("aux", None)

            energy = loss_value[0] if hasattr(loss_value, "__getitem__") else loss_value
            variance = (
                aux.variance[0]
                if hasattr(aux.variance, "__getitem__")
                else aux.variance
            )
            pmove_val = pmove[0] if hasattr(pmove, "__getitem__") else pmove
            step_val = step[0] if hasattr(step, "__getitem__") else step
            lr = jnp.asarray(schedule(step_val))
            step_stats = StepStats(
                energy=energy, variance=variance, pmove=pmove_val, learning_rate=lr
            )
            return new_params, new_opt_state, new_data, loss_keys, step_stats

        step_fn = kfac_step_fn
    else:

        @functools.partial(constants.pmap, donate_argnums=(0, 1, 2))
        def adam_step_fn(
            params: ParamTree,
            opt_state: Any,
            data: types.FermiNetData,
            key: jax.Array,
            step: jnp.ndarray,
            mcmc_width: Any,
        ) -> tuple[Any, Any, Any, Any, StepStats]:
            key, mcmc_key, loss_key = jax.random.split(key, 3)
            new_data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)

            new_params, new_opt_state, loss_value, aux, _ = update_fn(
                params, opt_state, loss_key, new_data, step, _loss_with_kfac
            )

            energy = constants.pmean(loss_value)
            variance = constants.pmean(aux.variance)
            pmove = constants.pmean(pmove)
            lr = jnp.asarray(schedule(step))
            stats = StepStats(
                energy=energy, variance=variance, pmove=pmove, learning_rate=lr
            )
            return new_params, new_opt_state, new_data, key, stats

        step_fn = adam_step_fn

    iterations = int(cfg_any.optim.iterations)
    print_every = int(cfg_any.log.print_every)
    checkpoint_every = int(cfg_any.log.checkpoint_every)
    save_path = cfg_any.log.save_path

    start = time.time()
    for i in range(step, iterations):
        width_array = jnp.full((device_count,), width)
        step_array = jnp.full((device_count,), i, dtype=jnp.int32)
        if cfg_any.optim.optimizer == "kfac":
            step_result = step_fn(params, opt_state, data, key, step_array, width_array)
        else:
            step_array = jnp.full((device_count,), i, dtype=jnp.int32)
            step_result = step_fn(params, opt_state, data, key, step_array, width_array)
        step_result = cast(tuple[Any, Any, Any, Any, Any], step_result)
        new_params = step_result[0]
        new_opt_state = step_result[1]
        data = step_result[2]
        key = step_result[3]
        stats = step_result[4]

        if cfg_any.optim.optimizer == "kfac":
            host_stats = jax.tree_util.tree_map(
                lambda x: float(x) if hasattr(x, "item") else x, stats
            )
        else:
            host_stats = jax.tree_util.tree_map(lambda x: jax.device_get(x)[0], stats)

        energy_val = float(host_stats.energy)
        if not jnp.isfinite(energy_val):
            width = float(cfg_any.mcmc.move_width)
            if (i + 1) % print_every == 0:
                wall = time.time() - start
                _log_stats(i + 1, host_stats, wall, width)
                start = time.time()
            continue

        params = new_params
        opt_state = new_opt_state

        if (i + 1) % print_every == 0:
            wall = time.time() - start
            _log_stats(i + 1, host_stats, wall, width)
            start = time.time()

        if (i + 1) % int(cfg_any.mcmc.adapt_frequency) == 0:
            pmove_value = float(host_stats.pmove)
            width, pmoves = mcmc.update_mcmc_width(
                i + 1,
                width,
                int(cfg_any.mcmc.adapt_frequency),
                pmove_value,
                pmoves,
                pmove_max=cfg_any.mcmc.get("pmove_max", 0.55),
                pmove_min=cfg_any.mcmc.get("pmove_min", 0.5),
            )

        if (i + 1) % checkpoint_every == 0:
            host_params = jax.tree_util.tree_map(lambda x: jax.device_get(x)[0], params)
            host_opt_state = jax.tree_util.tree_map(
                lambda x: jax.device_get(x)[0], opt_state
            )
            host_data = jax.tree_util.tree_map(lambda x: jax.device_get(x)[0], data)
            checkpoint.save_checkpoint(
                save_path, i + 1, host_params, host_opt_state, host_data
            )

    host_params = jax.tree_util.tree_map(lambda x: jax.device_get(x)[0], params)
    host_opt_state = jax.tree_util.tree_map(lambda x: jax.device_get(x)[0], opt_state)
    host_data = jax.tree_util.tree_map(lambda x: jax.device_get(x)[0], data)
    return {
        "params": host_params,
        "opt_state": host_opt_state,
        "data": host_data,
        "step": iterations,
    }
