import jax.numpy as jnp
from ferminet import mcmc
w, pmoves = mcmc.update_mcmc_width(1, 0.5, 10, jnp.array(0.55), jnp.zeros(10))
print(w, pmoves)
