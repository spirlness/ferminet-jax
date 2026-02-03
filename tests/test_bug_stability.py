import jax
import jax.numpy as jnp
from ferminet.network import ExtendedFermiNet
from configs.h2_stage2_config import get_stage2_config

def test_collision_stability():
    config = get_stage2_config('default')
    network = ExtendedFermiNet(
        n_electrons=config['n_electrons'],
        n_up=config['n_up'],
        nuclei_config=config['nuclei'],
        network_config=config['network']
    )

    # All electrons at the same position (collision)
    x_collision = jnp.zeros((1, config['n_electrons'], 3))
    log_psi = network.apply(network.params, x_collision)

    print(f"Log psi at collision: {log_psi}")
    assert jnp.all(jnp.isfinite(log_psi)), "Log psi should be finite even at electron collision"

if __name__ == "__main__":
    try:
        test_collision_stability()
        print("Success: Log psi is finite at collision.")
    except AssertionError as e:
        print(f"Failure: {e}")
    except Exception as e:
        print(f"Error: {e}")
