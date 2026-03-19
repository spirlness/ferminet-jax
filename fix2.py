import re

with open("src/ferminet/train.py", "r") as f:
    content = f.read()

# Let's inspect the `update_mcmc_width` logic.
# If `idx = t % adapt_frequency` is only evaluated in Python!
# `t` is `i+1`, a Python int.
# So `idx` is a python int.
# `idx = int(jnp.clip(jnp.asarray(idx), 0, pmoves.size - 1))`
# Since `idx` and `pmoves.size` are python ints, `jnp.asarray(idx)` creates a small array and `int()` pulls it back.

print("Done")
