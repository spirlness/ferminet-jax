import jax.numpy as jnp

from ferminet import envelopes


def _sample_inputs():
    ae = jnp.ones((2, 2, 3))
    r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
    output_dims = (2, 1)
    return ae, r_ae, output_dims


def test_isotropic_envelope_applies_with_positive_parameters():
    env = envelopes.make_isotropic_envelope()
    params = env.init(2, (2, 1))
    ae, r_ae, output_dims = _sample_inputs()
    outputs = env.apply(params, ae, r_ae, output_dims)

    assert outputs[0] is not None and outputs[0].shape == (2, 2)
    assert outputs[1] is not None and outputs[1].shape == (2, 1)
    assert jnp.all(outputs[0] > 0)


def test_diagonal_envelope_uses_per_dimension_decay():
    env = envelopes.make_diagonal_envelope()
    params = env.init(2, (1, 1), ndim=3)
    ae, r_ae, _ = _sample_inputs()
    outputs = env.apply(params, ae, r_ae, (1, 1))

    assert outputs[0].shape == (2, 1)
    assert outputs[1] is not None and outputs[1].shape == (2, 1)
    assert jnp.all(outputs[0] > 0)


def test_full_envelope_handles_learnable_directions():
    env = envelopes.make_full_envelope()
    params = env.init(2, (1,))
    ae, r_ae, _ = _sample_inputs()
    outputs = env.apply(params, ae, r_ae, (1,))

    assert len(outputs) == 1
    assert outputs[0].shape == (2, 1)
    assert jnp.all(jnp.isfinite(outputs[0]))
