import re

with open("src/ferminet/train.py", "r") as f:
    content = f.read()

# I am completely lost because I can't find `File: src/ferminet/train.py:196` Issue: `Checking isfinite on the host every step forces a blocking synchronization`
# IN THE CURRENT FILE AT LINE 196:
# It's returning from `kfac_step_fn`.
# But `is_finite = jnp.isfinite(energy)` is ON DEVICE (line 189).
# Then at line 290, `if not jnp.isfinite(energy_val):` is ON HOST but only every `print_every` steps.

# WAIT, the memory says:
# "To avoid redundant host-device synchronizations, the required `pmove` scalar is extracted from the unconditionally fetched `stats_host` array rather than via a separate `_to_host(pmove_ref)` call."

# "To minimize synchronization overhead for small data structures (like statistics arrays), fetch the entire structure to the host once using `jax.device_get` and extract fields there, rather than slicing on the device and fetching individual elements."

# "The `_to_host` utility function in `src/ferminet/train.py` has been removed. Device-to-host conversions should now be handled by extracting values from unconditionally fetched grouped arrays (like `stats_host`) to avoid sequential synchronization overhead."

# Is the reviewer complaining because I moved `jax.device_get(stats)` outside the `print_every` block?
# If I move it outside, it fetches every step unconditionally. Is that bad?
# The reviewer said: "The patch extracts `jax.device_get(stats)` outside of the `if (i + 1) % print_every == 0:` block, forcing a blocking synchronization to happen unconditionally on every single step."
# But wait... If the codebase previously had `_to_host(pmove_ref)` on EVERY step, then it WAS synchronizing unconditionally on every step!
# By extracting `jax.device_get(stats)` unconditionally on every step, I just changed ONE sync (`_to_host`) to ANOTHER sync (`device_get`).
# If I don't extract `jax.device_get(stats)` every step, then I don't sync every step.
# BUT I NEED `pmove_val` every step for `mcmc.update_mcmc_width`?
# NO I DON'T HAVE TO SYNC IT! I can just pass `pmove_ref` (which is a device array) to `mcmc.update_mcmc_width`!
# Oh, wait... But if I pass `pmove_ref` to `mcmc.update_mcmc_width`, then what about `width = float(...)` inside `mcmc.update_mcmc_width`? That will sync every step anyway!
# Oh, the memory actually says: "MCMC width adaptation is performed on the host using `mcmc.update_mcmc_width`. To avoid redundant host-device synchronizations, the required `pmove` scalar is extracted from the unconditionally fetched `stats_host` array rather than via a separate `_to_host(pmove_ref)` call."
# This literal sentence says: "extracted from the unconditionally fetched `stats_host` array".
# "unconditionally fetched `stats_host` array".
# UNCONDITIONALLY FETCHED.
# That means I *MUST* fetch `stats_host` unconditionally every step.

# Let me re-read the reviewer's feedback. "The agent needed to either move the width updates/checks inside the JIT-compiled step function (handling it on-device) or only fetch pmove and energy every N steps (less frequently)."
# Is there a way to use `jax.experimental.host_callback`? Or is `update_mcmc_width` supposed to be called ONLY every `adapt_frequency` steps?
# WAIT! `update_mcmc_width` says: "Adapts MCMC step width based on acceptance rate."
# If I look at the usage:
# `width, pmoves = mcmc.update_mcmc_width(..., pmove_value, ...)`
# `pmoves` is a ring buffer of size `adapt_frequency` that holds `pmove` for each step.
# Then every `adapt_frequency` steps, it computes the mean of `pmoves` and updates `width`.
# This fundamentally requires `pmove` to be collected every step.
# But it can collect it in a DEVICE array `pmoves` WITHOUT host synchronization!
# IF `pmove` is passed as a DEVICE array `pmove_ref`, then `pmoves.at[idx].set(pmove_ref)` is done entirely on the device asynchronously!
# THEN, only when `t > 0 and idx == 0` (every `adapt_frequency` steps), it computes `float(jnp.mean(pmoves))` which syncs with the host.
# BUT wait, the return value `width` is a python float. The `width = float(jnp.clip(jnp.asarray(width), ...))` at the end of `update_mcmc_width` will cause a sync every step if it compiles `jnp.clip` and pulls it to host.
# Wait, `jnp.clip(jnp.asarray(width), ...)` takes `width` (a python float), pushes to device, clips, and pulls back. That happens every step. But since `width` is a host scalar, `jnp.asarray(width)` is just putting it on device and pulling it right back. That's a tiny sync, but it's a sync.
# Wait! Can we avoid `jax.device_get(stats)` every step?
# If we DO NOT fetch `stats` every step, we avoid the large D2H sync of `stats` every step.

# BUT THE MEMORY SAYS:
# "To minimize synchronization overhead for small data structures (like statistics arrays), fetch the entire structure to the host once using `jax.device_get` and extract fields there, rather than slicing on the device and fetching individual elements."
# "The required `pmove` scalar is extracted from the unconditionally fetched `stats_host` array rather than via a separate `_to_host(pmove_ref)` call."

# Okay, what if I fetch `stats` to host ONLY inside the `step_fn`? NO, `step_fn` returns JAX arrays.

# Let's think about the reviewer's comment again:
# "The agent's patch extracts `jax.device_get(stats)` *outside* of the `if (i + 1) % print_every == 0:` block, forcing a blocking synchronization to happen unconditionally on every single step. While the agent successfully deleted a previous per-step block (`_to_host(pmove_ref)`), replacing it with another per-step block (`jax.device_get(stats)`) completely defeats the purpose of the issue. The training loop remains blocked every step."

# OKAY, so the reviewer is explicitly forbidding an unconditional `jax.device_get(stats)` every step!
# How do I fix the blocking sync of `_to_host(pmove_ref)` every step?
# Wait! In the JIT-compiled `step_fn`, `stats` is a stacked array of `[energy, variance, pmove, lr]`.
# Instead of `_to_host(pmove_ref)` every step, what if I use `jax.experimental.callbacks`?
# NO, the reviewer says: "The agent needed to either move the width updates/checks inside the JIT-compiled step function (handling it on-device) or only fetch pmove and energy every N steps (less frequently)."

# If I move the width update inside the `step_fn`?
# BUT `mcmc.update_mcmc_width` is a python function in `mcmc.py`.
# Let's look at `mcmc.py`. Is it possible to JIT compile `mcmc.update_mcmc_width`?
# It takes `t`, `width`, `adapt_frequency`, `pmove`, `pmoves`.
# Wait, `adapt_frequency` is static.
# If I do NOT update width inside `step_fn`, can I just update it less frequently?
# How do I collect `pmove` every step without syncing?
# What if I use `jax.device_get` ONLY every N steps? But `update_mcmc_width` is called EVERY step in the loop!
# Wait! What if I pass `jax.device_get(stats)[PMOVE]`? That syncs every step.
# What if I pass `stats[PMOVE]` to `update_mcmc_width`?
# Let's see what happens if I pass `stats[PMOVE]` to `update_mcmc_width`.
