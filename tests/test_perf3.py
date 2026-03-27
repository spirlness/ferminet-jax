import jax
import jax.numpy as jnp
import time

@jax.jit
def step_fn_stk(x):
    return jnp.stack([x+1, x+2, x+3, x+4])

@jax.jit
def step_fn_tup(x):
    return x+1, x+2, x+3, x+4

x = jnp.array(1.0)
stk = step_fn_stk(x)
tup = step_fn_tup(x)

def bench_stk():
    res = step_fn_stk(x)
    return jax.device_get(res)

def bench_tup():
    res = step_fn_tup(x)
    return jax.device_get(res)

bench_stk()
bench_tup()

n = 1000
t0 = time.time()
for _ in range(n):
    bench_stk()
t1 = time.time()
for _ in range(n):
    bench_tup()
t2 = time.time()

print(f"Stacked: {(t1 - t0) * 1000 / n:.3f} ms")
print(f"Tuple: {(t2 - t1) * 1000 / n:.3f} ms")
