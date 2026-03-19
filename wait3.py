# The task description might have been written BEFORE some refactoring, OR I should look at `train.py` again.
# The user's goal is to optimize the training loop by removing a blocking host-device synchronization.
# And the reviewer said:
# "The agent's patch extracts `jax.device_get(stats)` *outside* of the `if (i + 1) % print_every == 0:` block, forcing a blocking synchronization to happen unconditionally on every single step. While the agent successfully deleted a previous per-step block (`_to_host(pmove_ref)`), replacing it with another per-step block (`jax.device_get(stats)`) completely defeats the purpose of the issue. The training loop remains blocked every step."

# OK! So `_to_host(pmove_ref)` is the problem!
# "The training loop remains blocked every step" because `_to_host(pmove_ref)` causes a host-device sync every step!
# How do I get `pmove` to `update_mcmc_width` without blocking every step?
# The reviewer said:
# "The agent needed to either move the width updates/checks inside the JIT-compiled step function (handling it on-device) or only fetch pmove and energy every N steps (less frequently)."

# So I MUST implement one of these two things!
# "only fetch pmove and energy every N steps (less frequently)."

# Option 1: fetch `pmove` every `adapt_frequency` steps.
# BUT `update_mcmc_width` expects `pmove` to be passed every step to append to `pmoves`.
# IF I don't pass it every step, how does it append to `pmoves`?
# Can I pass `pmove_ref` directly to `update_mcmc_width` as a JAX array?
# If I pass `pmove_ref` directly, does it sync every step?
# We tested this in `review4.py` and it DID NOT sync every step... Wait, my test in `review4.py` took 1.3 seconds for 1000 steps. That's 1.3ms per step. Is that syncing?
# Let's test again if it syncs!
