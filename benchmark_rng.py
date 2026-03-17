import jax
import jax.numpy as jnp
from jax import random
import time

def mh_accept_with_split(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    lp_1: jnp.ndarray,
    lp_2: jnp.ndarray,
    ratio: jnp.ndarray,
    key: jax.Array,
    num_accepts: jnp.ndarray,
):
    key, subkey = jax.random.split(key)
    rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
    finite_proposal = jnp.isfinite(lp_2) & jnp.isfinite(ratio)
    cond = (ratio > rnd) & finite_proposal
    x_new = jnp.where(cond[..., None], x2, x1)
    lp_new = jnp.where(cond, lp_2, lp_1)
    num_accepts += jnp.sum(cond)
    return x_new, lp_new, num_accepts, key

def mh_accept_without_split(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    lp_1: jnp.ndarray,
    lp_2: jnp.ndarray,
    ratio: jnp.ndarray,
    subkey: jax.Array,
    num_accepts: jnp.ndarray,
):
    rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
    finite_proposal = jnp.isfinite(lp_2) & jnp.isfinite(ratio)
    cond = (ratio > rnd) & finite_proposal
    x_new = jnp.where(cond[..., None], x2, x1)
    lp_new = jnp.where(cond, lp_2, lp_1)
    num_accepts += jnp.sum(cond)
    return x_new, lp_new, num_accepts

x1 = jnp.ones((100, 3))
x2 = jnp.ones((100, 3))
lp_1 = jnp.ones((100,))
lp_2 = jnp.ones((100,))
ratio = jnp.ones((100,))
key = random.PRNGKey(0)
num_accepts = jnp.array(0)

# Compile
mh_accept_with_split_jit = jax.jit(mh_accept_with_split)
mh_accept_without_split_jit = jax.jit(mh_accept_without_split)

_ = mh_accept_with_split_jit(x1, x2, lp_1, lp_2, ratio, key, num_accepts)
_ = mh_accept_without_split_jit(x1, x2, lp_1, lp_2, ratio, key, num_accepts)

n_iters = 10000

t0 = time.time()
for _ in range(n_iters):
    _ = mh_accept_with_split_jit(x1, x2, lp_1, lp_2, ratio, key, num_accepts)
    # Don't block inside loop
jax.block_until_ready(_)
t1 = time.time()

t2 = time.time()
for _ in range(n_iters):
    _ = mh_accept_without_split_jit(x1, x2, lp_1, lp_2, ratio, key, num_accepts)
jax.block_until_ready(_)
t3 = time.time()

print(f"With split: {t1 - t0:.4f}s")
print(f"Without split: {t3 - t2:.4f}s")
