import re

with open("src/ferminet/train.py", "r") as f:
    content = f.read()

# Make math import
if "import math" not in content:
    content = content.replace("import inspect\n", "import inspect\nimport math\n")

# Replace jnp.isfinite with math.isfinite
content = content.replace("if not jnp.isfinite(energy_val):", "if not math.isfinite(energy_val):")

# Remove _to_host function
content = re.sub(r'def _to_host\(tree: Any\) -> Any:\n    """Convert a PyTree of device arrays to a PyTree of host scalars."""\n(?:    .*\n)*?\n', '', content)

# Remove the use of _to_host in the loop
loop_search = """        # Handle potential sharded stats array
        if stats.ndim == 2:
            pmove_ref = stats[0, PMOVE]
        else:
            pmove_ref = stats[PMOVE]
        pmove_value = _to_host(pmove_ref)
        width, pmoves = mcmc.update_mcmc_width(
            i + 1,
            width,
            adapt_frequency,
            pmove_value,"""

loop_replace = """        # Handle potential sharded stats array
        if stats.ndim == 2:
            pmove_ref = stats[0, PMOVE]
        else:
            pmove_ref = stats[PMOVE]

        # Pass the JAX device array directly to update_mcmc_width to avoid
        # a blocking synchronization every step. The float cast and sync
        # only happens inside when t % adapt_frequency == 0.
        width, pmoves = mcmc.update_mcmc_width(
            i + 1,
            width,
            adapt_frequency,
            pmove_ref,"""

content = content.replace(loop_search, loop_replace)

with open("src/ferminet/train.py", "w") as f:
    f.write(content)
