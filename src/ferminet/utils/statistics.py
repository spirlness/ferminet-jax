import jax.numpy as jnp


def exponential_moving_average(
    current: jnp.ndarray,
    new_value: jnp.ndarray,
    decay: float = 0.99,
) -> jnp.ndarray:
    return decay * current + (1.0 - decay) * new_value


def welford_update(
    count: int,
    mean: jnp.ndarray,
    m2: jnp.ndarray,
    new_value: jnp.ndarray,
) -> tuple[int, jnp.ndarray, jnp.ndarray]:
    count += 1
    delta = new_value - mean
    mean = mean + delta / count
    delta2 = new_value - mean
    m2 = m2 + delta * delta2
    return count, mean, m2


def welford_finalize(count: int, m2: jnp.ndarray) -> jnp.ndarray:
    if count < 2:
        return jnp.zeros_like(m2)
    return m2 / (count - 1)


def block_average(
    data: jnp.ndarray, block_size: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    n_blocks = len(data) // block_size
    trimmed = data[: n_blocks * block_size]
    blocks = trimmed.reshape(n_blocks, block_size)
    block_means = jnp.mean(blocks, axis=1)

    mean = jnp.mean(block_means)
    std_err = jnp.std(block_means) / jnp.sqrt(n_blocks)

    return mean, std_err
