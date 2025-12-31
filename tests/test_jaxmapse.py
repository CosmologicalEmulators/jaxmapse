import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from jaxace import FlaxEmulator

from jaxmapse import LinearPkEmulator, NonLinearBoostPkEmulator, PkEmulator

# Configuration
jax.config.update("jax_enable_x64", True)


class SimpleMLP(nn.Module):
    out_features: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.out_features)(x)


@pytest.fixture
def mock_linear_emu():
    n_in = 6  # 5 params + 1 z
    n_out = 40  # k-grid size
    model = SimpleMLP(out_features=n_out)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones((n_in,)))

    flax_emu = FlaxEmulator(
        model=model,
        parameters=params,
        description={"emulator_description": {"name": "test_lin"}},
    )

    k_grid = jnp.linspace(0.01, 1.0, n_out)
    in_minmax = jnp.tile(jnp.array([0.0, 1.0]), (n_in, 1))
    out_minmax = jnp.tile(jnp.array([0.0, 1.0]), (n_out, 1))

    def preprocessing(x):
        return x

    def postprocessing(p, out, D, emu):
        return out * D

    return LinearPkEmulator(
        trained_emulator=flax_emu,
        k_grid=k_grid,
        in_minmax=in_minmax,
        out_minmax=out_minmax,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
    )


@pytest.fixture
def mock_boost_emu():
    n_in = 6  # 5 params + 1 z
    n_out = 40
    model = SimpleMLP(out_features=n_out)
    key = jax.random.PRNGKey(1)
    params = model.init(key, jnp.ones((n_in,)))

    flax_emu = FlaxEmulator(
        model=model,
        parameters=params,
        description={"emulator_description": {"name": "test_boost"}},
    )

    k_grid = jnp.linspace(0.01, 1.0, n_out)
    in_minmax = jnp.tile(jnp.array([0.0, 1.0]), (n_in, 1))
    out_minmax = jnp.tile(jnp.array([0.0, 1.0]), (n_out, 1))

    def postprocessing(p, out, emu):
        return out

    return NonLinearBoostPkEmulator(
        trained_emulator=flax_emu,
        k_grid=k_grid,
        in_minmax=in_minmax,
        out_minmax=out_minmax,
        postprocessing=postprocessing,
    )


def test_linear_pk_shapes(mock_linear_emu):
    params = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    z_scalar = 1.0
    D_scalar = 0.8

    # Test scalar
    pk = mock_linear_emu.get_Pk(params, z_scalar, D_scalar)
    assert pk.shape == (40,)

    # Test vector
    z_vec = jnp.array([0.0, 1.0, 2.0])
    D_vec = jnp.array([1.0, 0.8, 0.6])
    pk_vec = mock_linear_emu.get_Pk(params, z_vec, D_vec)
    assert pk_vec.shape == (3, 40)


def test_boost_pk_shapes(mock_boost_emu):
    params = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    z_scalar = 1.0

    pk = mock_boost_emu.get_Pk(params, z_scalar)
    assert pk.shape == (40,)

    z_vec = jnp.array([0.0, 1.0, 2.0])
    pk_vec = mock_boost_emu.get_Pk(params, z_vec)
    assert pk_vec.shape == (3, 40)


def test_pk_emulator_composite(mock_linear_emu, mock_boost_emu):
    full_emu = PkEmulator(
        linear_pmm=mock_linear_emu, linear_pkcb=mock_linear_emu, boost=mock_boost_emu
    )

    params = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    z = 1.0
    D = 0.8

    pk_total = full_emu.get_Pk(params, z, D)
    pk_lin = full_emu.get_linear_pmm(params, z, D)
    pk_boost = mock_boost_emu.get_Pk(params, z)

    assert pk_total.shape == (40,)
    assert jnp.allclose(pk_total, pk_lin * pk_boost)


def test_differentiability(mock_linear_emu):
    params = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    z = 1.0
    D = 0.8

    def loss(p):
        return jnp.sum(mock_linear_emu.get_Pk(p, z, D))

    grad = jax.grad(loss)(params)
    assert grad.shape == (5,)
    assert jnp.all(jnp.isfinite(grad))


def test_vmap_consistency(mock_linear_emu):
    params = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    z_vec = jnp.array([0.5, 1.5])
    D_vec = jnp.array([0.9, 0.7])

    # Batch call
    pk_batch = mock_linear_emu.get_Pk(params, z_vec, D_vec)

    # Manual loop
    pk_single_1 = mock_linear_emu.get_Pk(params, z_vec[0], D_vec[0])
    pk_single_2 = mock_linear_emu.get_Pk(params, z_vec[1], D_vec[1])

    assert jnp.allclose(pk_batch[0], pk_single_1)
    assert jnp.allclose(pk_batch[1], pk_single_2)
