1. **Refactor stats gathering in `train.py`**:
   - Instead of packing `energy`, `variance`, `pmove`, and `lr` into a single array using `jnp.stack`, group them into a standard Python tuple `(energy, variance, pmove, lr)`.
   - Update both `kfac_step_fn` and `adam_step_fn` to return this tuple.
   - Also update the type hints in `adam_step_fn` from `tuple[Any, Any, Any, Any, jax.Array]` to `tuple[Any, Any, Any, Any, tuple[Any, Any, Any, Any]]`.

2. **Refactor stats unpacking in the main training loop**:
   - Unpack `stats_host = jax.device_get(stats)` into `energy_host, variance_host, pmove_host, lr_host`.
   - Process each value with the existing `_to_float` helper, eliminating the need to manually handle `.ndim == 2` sharding for a stacked array.
   - For `pmove_ref` passed to `mcmc.update_mcmc_width()`, unpack the device tuple: `pmove_device = stats[2]`. If `pmove_device.ndim == 1`, take `pmove_device[0]`, otherwise `pmove_device`.
   - Remove unused `ENERGY`, `VARIANCE`, `PMOVE`, and `LEARNING_RATE` constants.

3. **Verify functionality and performance**:
   - Run `uv run pytest tests/`
   - Run `uv run ruff check` and `uv run ruff format`
   - Run the benchmark script `uv run python scripts/benchmark_train_step.py --timed-steps 20` to verify a performance improvement.

4. **Complete pre-commit steps**:
   - Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.

5. **Commit and create PR**:
   - Use the `submit` tool with a descriptive commit message explaining the change and its performance benefits.
