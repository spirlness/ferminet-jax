import jax
import jax.numpy as jnp
import time
import math

@jax.jit
def get_pmove():
    return jnp.ones(10).mean()

def update_mcmc_width(
    t: int,
    width: float,
    adapt_frequency: int,
    pmove: float,
    pmoves: jnp.ndarray,
) -> tuple[float, jnp.ndarray]:
    if adapt_frequency <= 0 or pmoves.size == 0:
        return width, pmoves

    idx = t % adapt_frequency
    idx = int(jnp.clip(jnp.asarray(idx), 0, pmoves.size - 1))
    pmoves = pmoves.at[idx].set(jnp.asarray(pmove, dtype=pmoves.dtype))

    if t > 0 and idx == 0:
        mean_pmove = float(jnp.mean(pmoves))
        if mean_pmove > 0.55:
            width *= 1.1
        elif mean_pmove < 0.5:
            width /= 1.1

    width = float(jnp.clip(jnp.asarray(width), 0.001, 10.0))
    return width, pmoves

pmoves = jnp.zeros(10)
t0 = time.time()
width = 1.0
for i in range(1, 1000):
    pmove = get_pmove()

    # If we cast to float every step, this is what `_to_host(pmove_ref)` does!
    pmove = float(pmove)

    width, pmoves = update_mcmc_width(i, width, 10, pmove, pmoves)
print(f"Time with float(): {time.time() - t0}")
