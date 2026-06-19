from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 compatibility
    import tomli as tomllib

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from jaxace import FlaxEmulator

from jaxmapse import (
    DEFAULT_EMULATOR_ARTIFACT,
    LinearPkEmulator,
    NonLinearBoostPkEmulator,
    PkEmulator,
)

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

    def preprocessing(x):
        return x

    def postprocessing(p, out, D, emu):
        return out

    return NonLinearBoostPkEmulator(
        trained_emulator=flax_emu,
        k_grid=k_grid,
        in_minmax=in_minmax,
        out_minmax=out_minmax,
        preprocessing=preprocessing,
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
    D_scalar = 1.0

    pk = mock_boost_emu.get_Pk(params, z_scalar, D_scalar)
    assert pk.shape == (40,)

    z_vec = jnp.array([0.0, 1.0, 2.0])
    D_vec = jnp.array([1.0, 0.8, 0.6])
    pk_vec = mock_boost_emu.get_Pk(params, z_vec, D_vec)
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
    pk_boost = mock_boost_emu.get_Pk(params, z, D)

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


def test_pca_output_reconstruction(mock_linear_emu):
    mock_linear_emu.pca_mean = jnp.array([10.0, 20.0, 30.0])
    mock_linear_emu.pca_projection = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    decoded = mock_linear_emu._decode_output(jnp.array([2.0, 3.0]))

    assert jnp.allclose(decoded, jnp.array([12.0, 23.0, 35.0]))


def test_default_artifact_metadata():
    artifacts_toml = Path(__file__).resolve().parents[1] / "Artifacts.toml"
    data = tomllib.loads(artifacts_toml.read_text())

    assert DEFAULT_EMULATOR_ARTIFACT == "mnuw0wacdm_class"
    artifact = data[DEFAULT_EMULATOR_ARTIFACT]

    assert artifact["git-tree-sha1"] == "c1a93f08faafd81f6c62ac3ee97bb9fe37f8cf2e"
    assert artifact["download"][0]["url"] == (
        "https://zenodo.org/records/20646263/files/"
        "trained_mapse_mnuw0wacdm_sym_ratio_pca_1em6_250000.tar.xz?download=1"
    )
    assert artifact["download"][0]["sha256"] == (
        "1624999b2ae943a8820927cac1eafede033f6b77b3c166ce88a6cf109361c594"
    )


class StaticComponent:
    def __init__(self, k_grid, values):
        self.k_grid = jnp.asarray(k_grid)
        self.values = jnp.asarray(values)

    def get_Pk(self, input_params, z, D):
        z_arr = jnp.asarray(z)
        if z_arr.ndim == 0:
            return self.values
        return jnp.repeat(self.values[None, :], z_arr.shape[0], axis=0)


def test_pk_emulator_interpolates_linear_pmm_to_boost_grid():
    linear_k = jnp.array([0.0, 1.0, 2.0, 3.0])
    boost_k = jnp.array([0.5, 1.5, 2.5])
    linear_values = jnp.array([1.0, 3.0, 7.0, 13.0])
    boost_values = jnp.array([2.0, 4.0, 8.0])
    emu = PkEmulator(
        linear_pmm=StaticComponent(linear_k, linear_values),
        linear_pkcb=StaticComponent(linear_k, linear_values),
        boost=StaticComponent(boost_k, boost_values),
    )

    pk = emu.get_Pk(jnp.ones(5), 0.0, 1.0)
    expected = jnp.interp(boost_k, linear_k, linear_values) * boost_values

    assert pk.shape == boost_k.shape
    assert jnp.allclose(pk, expected)

    z = jnp.array([0.0, 1.0])
    pk_z = emu.get_Pk(jnp.ones(5), z, jnp.ones_like(z))
    assert pk_z.shape == (2, len(boost_k))
    assert jnp.allclose(pk_z[0], expected)


def test_packaged_artifacts_toml_is_discoverable():
    from importlib.resources import files

    registry = files("jaxmapse") / "Artifacts.toml"
    data = tomllib.loads(registry.read_text())

    assert DEFAULT_EMULATOR_ARTIFACT in data
