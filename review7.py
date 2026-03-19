import jax
import jax.numpy as jnp
import time

# What if we move update_mcmc_width INSIDE step_fn?
# NO, the reviewer said "only fetch pmove and energy every N steps (less frequently)."
# This is explicitly saying we should ONLY CALL `update_mcmc_width` or only fetch `pmove` less frequently!

# "only fetch pmove and energy every N steps (less frequently)"
# WAIT! The current `mcmc.update_mcmc_width` computes the mean of the past `adapt_frequency` pmoves.
# What if we just average the `pmove` across `adapt_frequency` steps INSIDE the `step_fn`, or we just use `pmove` fetched every `adapt_frequency` steps to update width?
# But `mcmc.update_mcmc_width` uses `pmoves` as a ring buffer: `pmoves.at[idx].set(pmove)`.
# IF we only call it every `adapt_frequency` steps, we would miss the intermediate `pmove` values!
# OR we can change `update_mcmc_width`? NO, I am only supposed to modify `src/ferminet/train.py`.
# Wait, can I modify `src/ferminet/mcmc.py`?
# "File: src/ferminet/train.py:196 Issue: Blocking Host-Device Synchronization in Training Loop"
# The task description ONLY mentions `src/ferminet/train.py`.

# BUT WAIT! I ALREADY MODIFIED `src/ferminet/train.py`.
# What if the user just meant: "don't call `update_mcmc_width` every step"?
# NO, if we don't call it, the ring buffer isn't updated.

# What if `pmoves` in `train.py` is updated as a JAX array inside the JIT function?
# The reviewer said: "The agent needed to either move the width updates/checks inside the JIT-compiled step function (handling it on-device) OR only fetch pmove and energy every N steps (less frequently)."

# SO I CAN DO EITHER!
# "move the width updates/checks inside the JIT-compiled step function (handling it on-device)"
# OR
# "only fetch pmove and energy every N steps (less frequently)"

# Wait, if I choose "only fetch pmove every N steps", how do I update the ring buffer `pmoves`?
# In `train.py`, `pmoves = jnp.zeros(int(cfg_any.mcmc.adapt_frequency))`.
# If I don't fetch `pmove` every step, I can't populate the host-side ring buffer `pmoves` correctly!
# BUT what if I don't use a ring buffer? What if `update_mcmc_width` handles it?
# Let's look at `update_mcmc_width` again.
# Wait! In `train.py`, `pmoves = jnp.zeros(int(cfg_any.mcmc.adapt_frequency))`
# `width, pmoves = mcmc.update_mcmc_width(i + 1, width, adapt_frequency, pmove_value, pmoves, ...)`
# `pmove_value` is `_to_host(pmove_ref)`.
# IF I PASS `pmove_ref` directly to `mcmc.update_mcmc_width`, DOES IT FETCH `pmove` TO HOST?
# No, it stays a JAX device array inside the `pmoves` array!
# `jnp.asarray(pmove)` on a device array is a no-op!
# AND `pmoves.at[idx].set(...)` operates on the device array!
# It ONLY fetches to host when `mean_pmove = float(jnp.mean(pmoves))` happens, WHICH IS ONLY EVERY `adapt_frequency` STEPS!
# THIS IS EXACTLY "handling it on-device / only fetch pmove every N steps"!!

# Why did the reviewer say my previous patch failed?
# Because my previous patch DID NOT pass `pmove_ref` to `update_mcmc_width`!
# My previous patch did:
# `stats_host = jax.device_get(stats)` UNCONDITIONALLY EVERY STEP, and passed `float(stats_host[PMOVE])`!
# THAT IS A PER-STEP FETCH!
# My patch ADDED an unconditional `jax.device_get(stats)` every step!
# I NEED TO JUST PASS `pmove_ref` directly to `update_mcmc_width` WITHOUT fetching it to the host every step!
