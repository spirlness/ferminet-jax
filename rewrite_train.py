import re

with open('src/ferminet/train.py', 'r') as f:
    content = f.read()

# Replace jnp.stack with StepStats
content = re.sub(
    r'step_stats = jnp\.stack\(\[energy, variance, pmove_val, lr\]\)',
    r'step_stats = train_utils.StepStats(energy=energy, variance=variance, pmove=pmove_val, learning_rate=lr)',
    content
)
content = re.sub(
    r'stats = jnp\.stack\(\[energy, variance, pmove, lr\]\)',
    r'stats = train_utils.StepStats(energy=energy, variance=variance, pmove=pmove, learning_rate=lr)',
    content
)

# Rename _to_float to _to_host_scalar
content = content.replace('def _to_float(arr: Any) -> float:', 'def _to_host_scalar(arr: Any) -> float:')

# Update the training loop logic for stats handling
old_loop_logic = """        if (i + 1) % print_every == 0:
            stats_host = jax.device_get(stats)
            # Handle sharded stats array (e.g. from pmap)
            if stats_host.ndim == 2:
                stats_host = stats_host[0]

            energy_val = float(stats_host[ENERGY])
            variance_val = float(stats_host[VARIANCE])
            pmove_val = float(stats_host[PMOVE])
            lr_val = float(stats_host[LEARNING_RATE])

            if not jnp.isfinite(energy_val):
                width = float(cfg_any.mcmc.move_width)
                log_stats = train_utils.StepStats(
                    energy=energy_val,
                    variance=variance_val,
                    pmove=pmove_val,
                    learning_rate=lr_val,
                )
                wall = time.time() - start
                train_utils.log_stats(i + 1, log_stats, wall, width)
                start = time.time()
                continue

            log_stats = train_utils.StepStats(
                energy=energy_val,
                variance=variance_val,
                pmove=pmove_val,
                learning_rate=lr_val,
            )
            wall = time.time() - start
            train_utils.log_stats(i + 1, log_stats, wall, width)
            start = time.time()

        # Handle potential sharded stats array
        if stats.ndim == 2:
            pmove_ref = stats[0, PMOVE]
        else:
            pmove_ref = stats[PMOVE]"""

new_loop_logic = """        if (i + 1) % print_every == 0:
            energy_val = _to_host_scalar(jax.device_get(stats.energy))

            if not jnp.isfinite(energy_val):
                width = float(cfg_any.mcmc.move_width)

                variance_host, pmove_host, lr_host = jax.device_get(
                    (stats.variance, stats.pmove, stats.learning_rate)
                )
                log_stats = train_utils.StepStats(
                    energy=energy_val,
                    variance=_to_host_scalar(variance_host),
                    pmove=_to_host_scalar(pmove_host),
                    learning_rate=_to_host_scalar(lr_host),
                )

                wall = time.time() - start
                train_utils.log_stats(i + 1, log_stats, wall, width)
                start = time.time()
                continue

            variance_host, pmove_host, lr_host = jax.device_get(
                (stats.variance, stats.pmove, stats.learning_rate)
            )
            log_stats = train_utils.StepStats(
                energy=energy_val,
                variance=_to_host_scalar(variance_host),
                pmove=_to_host_scalar(pmove_host),
                learning_rate=_to_host_scalar(lr_host),
            )
            wall = time.time() - start
            train_utils.log_stats(i + 1, log_stats, wall, width)
            start = time.time()

        pmove_ref = stats.pmove
        if hasattr(pmove_ref, "ndim") and pmove_ref.ndim > 0:
            pmove_ref = pmove_ref[0]"""

content = content.replace(old_loop_logic, new_loop_logic)

with open('src/ferminet/train.py', 'w') as f:
    f.write(content)
print("Done")
