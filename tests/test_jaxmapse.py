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
    BUILTIN_POSTPROCESSING,
    BUILTIN_PREPROCESSING,
    DEFAULT_EMULATOR_ARTIFACT,
    load_trained_emulators,
    LOAD_PRESETS,
    TransferFunctionEmulator,
    postprocessing_identity,
    postprocessing_linear_pk_mnuw0wacdm_sym_ratio,
    preprocessing_identity,
    preprocessing_linear_pk_mnuw0wacdm,
)
from jaxmapse import jaxmapse as core

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

    return TransferFunctionEmulator(
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


def test_builtin_preprocessing_and_postprocessing_registries():
    params = jnp.arange(8.0)

    assert BUILTIN_PREPROCESSING["identity"] is preprocessing_identity
    assert BUILTIN_PREPROCESSING["linear_pk_mnuw0wacdm"] is preprocessing_linear_pk_mnuw0wacdm
    assert BUILTIN_POSTPROCESSING["identity"] is postprocessing_identity
    assert (
        BUILTIN_POSTPROCESSING["linear_pk_mnuw0wacdm_sym_ratio"]
        is postprocessing_linear_pk_mnuw0wacdm_sym_ratio
    )
    assert LOAD_PRESETS["mnuw0wacdm_linear"]["preprocessing_name"] == "linear_pk_mnuw0wacdm"

    assert jnp.allclose(preprocessing_identity(params), params)
    assert jnp.allclose(preprocessing_linear_pk_mnuw0wacdm(params), params[2:])
    assert jnp.allclose(postprocessing_identity(params, params + 1.0, None, None), params + 1.0)


def test_load_component_function_uses_named_builtins_and_legacy_files(tmp_path):
    assert (
        core._load_component_function(
            str(tmp_path),
            {"preprocessing_name": "identity"},
            "preprocessing_name",
            "preprocessing.py",
            BUILTIN_PREPROCESSING,
            "preprocessing",
        )
        is preprocessing_identity
    )
    assert (
        core._load_component_function(
            str(tmp_path),
            {"emulator_description": {"postprocessing_name": "identity"}},
            "postprocessing_name",
            "postprocessing.py",
            BUILTIN_POSTPROCESSING,
            "postprocessing",
        )
        is postprocessing_identity
    )
    assert (
        core._load_component_function(
            str(tmp_path),
            {"preprocessing_name": "identity"},
            "preprocessing_name",
            "preprocessing.py",
            BUILTIN_PREPROCESSING,
            "preprocessing",
            explicit_name="linear_pk_mnuw0wacdm",
        )
        is preprocessing_linear_pk_mnuw0wacdm
    )

    with pytest.raises(ValueError, match="not registered"):
        core._load_component_function(
            str(tmp_path),
            {"preprocessing_name": "not_a_builtin"},
            "preprocessing_name",
            "preprocessing.py",
            BUILTIN_PREPROCESSING,
            "preprocessing",
        )

    legacy_file = tmp_path / "preprocessing.py"
    legacy_file.write_text("def preprocessing(params):\n    return params + 1\n")
    legacy = core._load_component_function(
        str(tmp_path),
        {},
        "preprocessing_name",
        "preprocessing.py",
        BUILTIN_PREPROCESSING,
        "preprocessing",
    )
    assert jnp.allclose(legacy(jnp.array([1.0, 2.0])), jnp.array([2.0, 3.0]))


def test_load_emulator_strict_composite_validation(tmp_path):
    # Case 1: directory exists but has no nn_setup.json and subfolders like Pk_lin_mm or Boost
    (tmp_path / "Pk_lin_mm").mkdir()
    with pytest.raises(ValueError, match="appears to be a composite emulator bundle"):
        core.load_emulator(str(tmp_path))

    # Case 2: directory exists but has no nn_setup.json and no subfolders
    (tmp_path / "Pk_lin_mm").rmdir()
    with pytest.raises(ValueError, match="found in"):
        core.load_emulator(str(tmp_path))



def test_load_trained_emulators_returns_cached_component_dict(monkeypatch):
    calls = []

    class DummyEmu:
        def __init__(self, name):
            self.name = name
            self.k_grid = jnp.array([0.1, 0.2])

    def fake_artifact_path(name):
        calls.append(("artifact_path", name))
        return Path("/tmp/fake-artifact-root")

    def fake_load_emulator(path, preset=None, **kwargs):
        calls.append(("load_emulator", Path(path).name, preset))
        return DummyEmu(Path(path).name)

    monkeypatch.setattr(core, "artifact_path", fake_artifact_path)
    monkeypatch.setattr(core, "load_emulator", fake_load_emulator)
    monkeypatch.setattr(core, "_TRAINED_EMULATORS_CACHE", None)

    trained = load_trained_emulators(force_reload=True)
    assert DEFAULT_EMULATOR_ARTIFACT in trained
    assert set(trained[DEFAULT_EMULATOR_ARTIFACT]) == {"pmm", "pcb"}
    assert trained[DEFAULT_EMULATOR_ARTIFACT]["pmm"].name == "Pk_lin_mm"
    assert trained[DEFAULT_EMULATOR_ARTIFACT]["pcb"].name == "Pk_lin_cb"
    assert calls == [
        ("artifact_path", DEFAULT_EMULATOR_ARTIFACT),
        ("load_emulator", "Pk_lin_mm", "mnuw0wacdm_linear"),
        ("load_emulator", "Pk_lin_cb", "mnuw0wacdm_linear"),
    ]

    # Cached path: no additional loads
    calls.clear()
    cached = load_trained_emulators()
    assert cached is trained
    assert calls == []

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


def test_packaged_artifacts_toml_is_discoverable():
    from importlib.resources import files

    registry = files("jaxmapse") / "Artifacts.toml"
    data = tomllib.loads(registry.read_text())

    assert DEFAULT_EMULATOR_ARTIFACT in data


def test_interp_to_grid_validation():
    # Valid interpolation
    source_k = jnp.array([0.1, 0.5, 1.0])
    values = jnp.array([2.0, 4.0, 6.0])
    target_k = jnp.array([0.2, 0.8])
    res = core._interp_to_grid(source_k, values, target_k)
    assert res.shape == (2,)

    # Non-monotonic grids
    with pytest.raises(ValueError, match="source_k must be monotonically increasing"):
        core._interp_to_grid(jnp.array([1.0, 0.5]), values[:2], target_k)

    with pytest.raises(ValueError, match="target_k must be monotonically increasing"):
        core._interp_to_grid(source_k, values, jnp.array([0.8, 0.2]))

    # Target out of bounds
    with pytest.raises(ValueError, match="Target grid out of bounds"):
        core._interp_to_grid(source_k, values, jnp.array([0.05, 0.8]))

    with pytest.raises(ValueError, match="Target grid out of bounds"):
        core._interp_to_grid(source_k, values, jnp.array([0.2, 1.5]))
