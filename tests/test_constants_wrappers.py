import jax
import jax.numpy as jnp

from ferminet import constants


def test_constants_wrappers_execute_basic_collectives():
    @constants.pmap
    def mapped_fn(x):
        mean = constants.pmean(x)
        gathered = constants.all_gather(x)
        return mean, gathered

    inputs = jnp.arange(jax.local_device_count())
    mean, gathered = mapped_fn(inputs)
    assert jnp.allclose(mean, jnp.mean(inputs))
    assert gathered.shape[0] == jax.local_device_count()
