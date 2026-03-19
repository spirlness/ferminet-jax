import jax
import jax.numpy as jnp
import time

def update_mcmc_width(
    t: int,
    width: float,
    adapt_frequency: int,
    pmove: float,
    pmoves: jnp.ndarray,
    pmove_max: float = 0.55,
    pmove_min: float = 0.5,
    width_min: float = 0.001,
    width_max: float = 10.0,
) -> tuple[float, jnp.ndarray]:
    if adapt_frequency <= 0 or pmoves.size == 0:
        return width, pmoves

    idx = t % adapt_frequency
    idx = int(jnp.clip(jnp.asarray(idx), 0, pmoves.size - 1))
    pmoves = pmoves.at[idx].set(jnp.asarray(pmove, dtype=pmoves.dtype))

    if t > 0 and idx == 0:
        mean_pmove = float(jnp.mean(pmoves))
        if mean_pmove > pmove_max:
            width *= 1.1
        elif mean_pmove < pmove_min:
            width /= 1.1

    width = float(jnp.clip(jnp.asarray(width), width_min, width_max))
    return width, pmoves

pmoves = jnp.zeros(10)
pmove = jax.device_put(0.5)

# Benchmark sync time
@jax.jit
def dummy():
    return jnp.ones(10)

dummy().block_until_ready()

t0 = time.time()
for i in range(1, 1000):
    w, pmoves = update_mcmc_width(i, 1.0, 10, pmove, pmoves)
print(f"Time: {time.time() - t0}")
