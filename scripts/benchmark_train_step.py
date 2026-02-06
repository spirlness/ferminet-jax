"""Benchmark compile and steady-state train step latency.

This script provides a lightweight timing harness for the core training path:
- MCMC proposal/update
- Loss/gradient evaluation
- Optimizer update (Adam)

Example:
    uv run python scripts/benchmark_train_step.py --config helium_quick
"""

from __future__ import annotations

import argparse
import importlib
import statistics
import time
from typing import Any, cast

import jax
import jax.numpy as jnp

from ferminet import base_config, constants, loss, mcmc, optimizers, train_utils


def _load_config(name: str):
    mod = importlib.import_module(f"ferminet.configs.{name}")
    if not hasattr(mod, "get_config"):
        raise ValueError(f"Config module ferminet.configs.{name} lacks get_config()")
    cfg = mod.get_config()
    return base_config.resolve(cfg)


def _block_tree(tree: Any) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        jax.block_until_ready(leaf)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FermiNet train-step latency"
    )
    parser.add_argument("--config", default="helium_quick", help="Config module name")
    parser.add_argument(
        "--batch-size", type=int, default=0, help="Override cfg.batch_size"
    )
    parser.add_argument(
        "--mcmc-steps", type=int, default=0, help="Override cfg.mcmc.steps"
    )
    parser.add_argument(
        "--timed-steps", type=int, default=20, help="Number of timed steady-state steps"
    )
    parser.add_argument(
        "--seed", type=int, default=17, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    cfg_any = cast(Any, cfg)
    cfg_any.optim.optimizer = "adam"
    if args.batch_size > 0:
        cfg_any.batch_size = args.batch_size
    if args.mcmc_steps > 0:
        cfg_any.mcmc.steps = args.mcmc_steps

    atoms, charges, spins, ndim = train_utils.prepare_system(cfg)
    init_fn, apply_log, _ = train_utils.build_network(cfg, atoms, charges, spins)

    key = jax.random.PRNGKey(args.seed)
    key, pkey, dkey = jax.random.split(key, 3)
    params = init_fn(pkey)
    data = train_utils.init_mcmc_data(
        dkey,
        atoms,
        charges,
        spins,
        int(cfg_any.batch_size),
        int(ndim),
    )

    local_energy_fn = train_utils.make_local_energy_fn(apply_log, charges, spins, cfg)
    loss_fn = loss.make_loss(
        apply_log,
        local_energy_fn,
        clip_local_energy=float(cfg_any.optim.get("clip_local_energy", 5.0)),
    )

    # Optimizer update functions use constants.pmean, which requires pmap axis.
    # For single-device benchmark mode, use identity reduction.
    if jax.local_device_count() == 1:
        setattr(constants, "pmean", lambda values: values)

    _, init_opt_state, update_fn = optimizers.make_optimizer(cfg, loss_fn, params)
    opt_state = init_opt_state(params)

    mcmc_width = float(cfg_any.mcmc.move_width)
    mcmc_step = mcmc.make_mcmc_step(
        apply_log,
        int(cfg_any.batch_size),
        int(cfg_any.mcmc.steps),
        atoms,
        ndim=int(ndim),
    )

    @jax.jit
    def benchmark_step(params, opt_state, data, key, step):
        key, mcmc_key, loss_key = jax.random.split(key, 3)
        new_data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)
        new_params, new_opt_state, loss_value, aux, _ = update_fn(
            params, opt_state, loss_key, new_data, step
        )
        return new_params, new_opt_state, new_data, key, loss_value, aux.variance, pmove

    # Compile + first execute
    t0 = time.perf_counter()
    params, opt_state, data, key, loss_val, var_val, pmove = benchmark_step(
        params, opt_state, data, key, jnp.asarray(0, dtype=jnp.int32)
    )
    _block_tree((params, opt_state, data, loss_val, var_val, pmove))
    compile_plus_first = time.perf_counter() - t0

    # Steady-state timing
    step_times = []
    for i in range(args.timed_steps):
        t1 = time.perf_counter()
        params, opt_state, data, key, loss_val, var_val, pmove = benchmark_step(
            params, opt_state, data, key, jnp.asarray(i + 1, dtype=jnp.int32)
        )
        _block_tree((loss_val, var_val, pmove))
        step_times.append(time.perf_counter() - t1)

    avg_ms = 1000.0 * statistics.mean(step_times)
    p50_ms = 1000.0 * statistics.median(step_times)
    p95_ms = 1000.0 * sorted(step_times)[max(0, int(0.95 * len(step_times)) - 1)]

    print("=== FermiNet Train-Step Benchmark (Adam) ===")
    print(f"Config: {args.config}")
    print(f"Batch size: {cfg_any.batch_size}")
    print(f"MCMC steps: {cfg_any.mcmc.steps}")
    print(f"Timed steps: {args.timed_steps}")
    print(f"Compile+first step: {compile_plus_first:.3f} s")
    print(f"Steady step avg: {avg_ms:.2f} ms")
    print(f"Steady step p50: {p50_ms:.2f} ms")
    print(f"Steady step p95: {p95_ms:.2f} ms")
    print(
        "Last stats: "
        f"loss={float(jax.device_get(loss_val)):.6f}, "
        f"variance={float(jax.device_get(var_val)):.6f}, "
        f"pmove={float(jax.device_get(pmove)):.4f}"
    )


if __name__ == "__main__":
    main()
