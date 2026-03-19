import re

with open("src/ferminet/train.py", "r") as f:
    content = f.read()

# Make math import
if "import math" not in content:
    content = content.replace("import inspect\n", "import inspect\nimport math\n")

# Replace jnp.isfinite with math.isfinite
content = content.replace("if not jnp.isfinite(energy_val):", "if not math.isfinite(energy_val):")

# Remove _to_host
content = re.sub(r'def _to_host\(tree: Any\) -> Any:\n    """Convert a PyTree of device arrays to a PyTree of host scalars."""\n(?:    .*\n)*?\n', '', content)

# Fix the unconditional per-step _to_host
# Only fetch pmove_value and update width when t % adapt_frequency == 0 or we just log it?
# Wait! mcmc.update_mcmc_width appends pmove EVERY STEP.
# Wait, if we just pass `pmove_ref` directly to update_mcmc_width, it will be a JAX array.
# Let's see if update_mcmc_width can take a JAX array:
# `pmoves.at[idx].set(jnp.asarray(pmove, dtype=pmoves.dtype))`
# It DOES use jnp.asarray(pmove). So we can just pass the JAX array `pmove_ref`!
# BUT update_mcmc_width currently returns a python float for `width`!
# Because it does: `width = float(jnp.clip(jnp.asarray(width), width_min, width_max))`
# This would trigger a device-to-host sync inside update_mcmc_width every step if we evaluate it.
# Wait, if `width` is a python float, the only sync is the float() cast.
# But does `update_mcmc_width` sync? YES!
# `idx = int(jnp.clip(...))` -> sync!
# `mean_pmove = float(jnp.mean(pmoves))` -> sync!
# `width = float(...)` -> sync!
