import jax
import jax.numpy as jnp
from ferminet import train
import time

def main():
    tree = {
        'energy': jnp.ones((100,)),
        'variance': jnp.ones((100,)),
        'pmove': jnp.ones((100,)),
        'learning_rate': jnp.ones((100,))
    }

    t0 = time.time()
    for _ in range(100):
        # packed single struct
        arr = jnp.stack([tree['energy'], tree['variance'], tree['pmove'], tree['learning_rate']])
        arr_host = jax.device_get(arr)

        val1 = float(arr_host[0, 0])
        val2 = float(arr_host[1, 0])
        val3 = float(arr_host[2, 0])
        val4 = float(arr_host[3, 0])

    print(f"Time: {time.time() - t0}")

if __name__ == '__main__':
    main()
