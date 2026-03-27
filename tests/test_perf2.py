import jax
import jax.numpy as jnp
import time

key = jax.random.PRNGKey(0)

# simulate a PyTree of device arrays
class MockData:
    def __init__(self, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.energy = jax.random.normal(k1, ())
        self.variance = jax.random.normal(k2, ())
        self.pmove = jax.random.normal(k3, ())
        self.learning_rate = jax.random.normal(k4, ())

data = MockData(key)

def bench_seq():
    e = jax.device_get(data.energy)
    v = jax.device_get(data.variance)
    p = jax.device_get(data.pmove)
    l = jax.device_get(data.learning_rate)
    return e, v, p, l

def bench_tup():
    return jax.device_get((data.energy, data.variance, data.pmove, data.learning_rate))

def bench_stk():
    arr = jnp.stack([data.energy, data.variance, data.pmove, data.learning_rate])
    return jax.device_get(arr)

# warmup
bench_seq()
bench_tup()
bench_stk()

n = 1000
t0 = time.time()
for _ in range(n):
    bench_seq()
t1 = time.time()
for _ in range(n):
    bench_tup()
t2 = time.time()
for _ in range(n):
    bench_stk()
t3 = time.time()

print(f"Sequential: {(t1 - t0) * 1000 / n:.3f} ms")
print(f"Tuple: {(t2 - t1) * 1000 / n:.3f} ms")
print(f"Stacked: {(t3 - t2) * 1000 / n:.3f} ms")
