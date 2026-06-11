import importlib.util
import json
import os
from functools import partial
from typing import Callable, Optional, Type, Union

import jax
import jax.numpy as jnp

# Import jaxace components
from jaxace import FlaxEmulator, init_emulator, inv_maximin, maximin
from jaxtyping import Array

# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)

DEFAULT_EMULATOR_ARTIFACT = "mnuw0wacdm_class"


class LinearPkEmulator:
    """
    Linear power spectrum emulator mirroring Mapse.jl's LinearPkEmulator.
    """

    def __init__(
        self,
        trained_emulator: FlaxEmulator,
        k_grid: Array,
        in_minmax: Array,
        out_minmax: Array,
        preprocessing: Callable,
        postprocessing: Callable,
        pca_mean: Optional[Array] = None,
        pca_projection: Optional[Array] = None,
    ):
        self.trained_emulator = trained_emulator
        self.k_grid = jnp.asarray(k_grid)
        self.in_minmax = jnp.asarray(in_minmax)
        self.out_minmax = jnp.asarray(out_minmax)
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.pca_mean = None if pca_mean is None else jnp.asarray(pca_mean)
        self.pca_projection = (
            None if pca_projection is None else jnp.asarray(pca_projection)
        )

    def _decode_output(self, output: Array) -> Array:
        """Map NN output coefficients back to the emulator k-grid if PCA is used."""
        if self.pca_mean is None or self.pca_projection is None:
            return output
        return self.pca_mean + self.pca_projection @ output

    def _get_Pk_single(self, input_params: Array, z: float, D: float) -> Array:
        """Core implementation for a single parameter set and single redshift."""
        preprocessed_input = self.preprocessing(input_params)
        nn_input = jnp.insert(preprocessed_input, 0, z)
        norm_input = maximin(nn_input, self.in_minmax)
        norm_output = self.trained_emulator.run_emulator(norm_input)
        output = inv_maximin(norm_output, self.out_minmax)
        output = self._decode_output(output)
        return self.postprocessing(input_params, output, D, self)

    def get_Pk(
        self, input_params: Array, z: Union[float, Array], D: Union[float, Array]
    ) -> Array:
        """
        Compute linear power spectrum. Handles scalar or vector z/D via automatic vmap.
        """
        if not hasattr(self, "_jit_get_Pk"):

            @partial(jax.jit, static_argnums=(0,))
            def _jit_get_Pk(self, params, z, D):
                if jnp.ndim(z) == 0:
                    return self._get_Pk_single(params, z, D)
                else:
                    return jax.vmap(self._get_Pk_single, in_axes=(None, 0, 0))(
                        params, z, D
                    )

            self._jit_get_Pk = _jit_get_Pk

        return self._jit_get_Pk(self, input_params, z, D)


class NonLinearBoostPkEmulator:
    """
    Non-linear boost emulator mirroring Mapse.jl's NonLinearBoostPkEmulator.
    """

    def __init__(
        self,
        trained_emulator: FlaxEmulator,
        k_grid: Array,
        in_minmax: Array,
        out_minmax: Array,
        preprocessing: Callable,
        postprocessing: Callable,
        pca_mean: Optional[Array] = None,
        pca_projection: Optional[Array] = None,
    ):
        self.trained_emulator = trained_emulator
        self.k_grid = jnp.asarray(k_grid)
        self.in_minmax = jnp.asarray(in_minmax)
        self.out_minmax = jnp.asarray(out_minmax)
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.pca_mean = None if pca_mean is None else jnp.asarray(pca_mean)
        self.pca_projection = (
            None if pca_projection is None else jnp.asarray(pca_projection)
        )

    def _decode_output(self, output: Array) -> Array:
        """Map NN output coefficients back to the emulator k-grid if PCA is used."""
        if self.pca_mean is None or self.pca_projection is None:
            return output
        return self.pca_mean + self.pca_projection @ output

    def _get_Pk_single(self, input_params: Array, z: float, D: float) -> Array:
        """Core implementation for a single parameter set and single redshift."""
        preprocessed_input = self.preprocessing(input_params)
        nn_input = jnp.insert(preprocessed_input, 0, z)
        norm_input = maximin(nn_input, self.in_minmax)
        norm_output = self.trained_emulator.run_emulator(norm_input)
        output = inv_maximin(norm_output, self.out_minmax)
        output = self._decode_output(output)
        return self.postprocessing(input_params, output, D, self)

    def get_Pk(self, input_params: Array, z: Union[float, Array], D: Union[float, Array]) -> Array:
        """Compute boost factor. Handles scalar or vector z via automatic vmap."""
        if D is None:
            raise ValueError("Growth factor D must be provided to get_Pk.")

        if not hasattr(self, "_jit_get_Pk"):

            @partial(jax.jit, static_argnums=(0,))
            def _jit_get_Pk(self, params, z, D):
                if jnp.ndim(z) == 0:
                    return self._get_Pk_single(params, z, D)
                else:
                    return jax.vmap(self._get_Pk_single, in_axes=(None, 0, 0))(
                        params, z, D
                    )

            self._jit_get_Pk = _jit_get_Pk

        return self._jit_get_Pk(self, input_params, z, D)


class PkEmulator:
    """
    Composite emulator combining linear and non-linear components.
    """

    def __init__(
        self,
        linear_pmm: LinearPkEmulator,
        linear_pkcb: LinearPkEmulator,
        boost: NonLinearBoostPkEmulator,
    ):
        self.linear_pmm = linear_pmm
        self.linear_pkcb = linear_pkcb
        self.boost = boost

    def get_Pk(
        self, input_params: Array, z: Union[float, Array], D: Union[float, Array]
    ) -> Array:
        """Returns P_mm,lin * Boost."""
        if not hasattr(self, "_jit_get_Pk"):

            @partial(jax.jit, static_argnums=(0,))
            def _jit_get_Pk(self, params, z, D):
                lin = self.linear_pmm.get_Pk(params, z, D)
                bst = self.boost.get_Pk(params, z, D)
                return lin * bst

            self._jit_get_Pk = _jit_get_Pk
        return self._jit_get_Pk(self, input_params, z, D)

    def get_linear_pmm(
        self, input_params: Array, z: Union[float, Array], D: Union[float, Array]
    ) -> Array:
        """Returns linear matter power spectrum."""
        return self.linear_pmm.get_Pk(input_params, z, D)

    def get_linear_pkcb(
        self, input_params: Array, z: Union[float, Array], D: Union[float, Array]
    ) -> Array:
        """Returns linear c+b power spectrum."""
        return self.linear_pkcb.get_Pk(input_params, z, D)


def _load_function(filepath: str, func_name: str) -> Callable:
    """Helper to load a function from a python file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Python file not found: {filepath}")
    spec = importlib.util.spec_from_file_location("module.name", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, func_name):
        raise ValueError(f"File {filepath} must define a '{func_name}' function")
    return getattr(module, func_name)


def load_emulator(
    path: str,
    structure: Union[
        Type[LinearPkEmulator], Type[NonLinearBoostPkEmulator]
    ] = LinearPkEmulator,
    **kwargs,
) -> Union[LinearPkEmulator, NonLinearBoostPkEmulator]:
    """
    Load an emulator from disk mirroring the Julia load_emulator logic.
    """
    with open(
        os.path.join(path, kwargs.get("nn_setup_file", "nn_setup.json")), "r"
    ) as f:
        nn_dict = json.load(f)

    weights = jnp.load(os.path.join(path, kwargs.get("weights_file", "weights.npy")))
    k_grid = jnp.load(os.path.join(path, kwargs.get("k_file", "k.npy")))
    in_minmax = jnp.load(
        os.path.join(path, kwargs.get("inminmax_file", "inminmax.npy"))
    )
    out_minmax = jnp.load(
        os.path.join(path, kwargs.get("outminmax_file", "outminmax.npy"))
    )

    pca_mean_file = os.path.join(path, kwargs.get("pca_mean_file", "pca_mean.npy"))
    pca_projection_name = kwargs.get(
        "pca_projection_file", kwargs.get("pca_basis_file", "pca_projection.npy")
    )
    pca_projection_file = os.path.join(path, pca_projection_name)
    has_pca = os.path.exists(pca_mean_file) or os.path.exists(pca_projection_file)
    if has_pca and not (os.path.exists(pca_mean_file) and os.path.exists(pca_projection_file)):
        raise FileNotFoundError(
            "PCA emulator output requires both pca_mean.npy and pca_projection.npy"
        )
    pca_mean = jnp.load(pca_mean_file) if has_pca else None
    pca_projection = jnp.load(pca_projection_file) if has_pca else None

    trained_emu = init_emulator(nn_dict, weights)

    preprocessing = _load_function(
        os.path.join(path, kwargs.get("preprocessing_file", "preprocessing.py")),
        "preprocessing",
    )
    postprocessing = _load_function(
        os.path.join(path, kwargs.get("postprocessing_file", "postprocessing.py")),
        "postprocessing",
    )

    return structure(
        trained_emulator=trained_emu,
        k_grid=k_grid,
        in_minmax=in_minmax,
        out_minmax=out_minmax,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        pca_mean=pca_mean,
        pca_projection=pca_projection,
    )


def load_pk_emulator(
    path: str,
    pmm_folder: str = "Pk_lin_mm",
    pcb_folder: str = "Pk_lin_cb",
    boost_folder: str = "Boost",
    **kwargs,
) -> PkEmulator:
    """
    Load the complete PkEmulator suite from a directory containing component subfolders.
    """
    pmm = load_emulator(
        os.path.join(path, pmm_folder), structure=LinearPkEmulator, **kwargs
    )
    pcb = load_emulator(
        os.path.join(path, pcb_folder), structure=LinearPkEmulator, **kwargs
    )
    boost = load_emulator(
        os.path.join(path, boost_folder), structure=NonLinearBoostPkEmulator, **kwargs
    )

    return PkEmulator(linear_pmm=pmm, linear_pkcb=pcb, boost=boost)


def load_emulator_from_artifact(
    artifact_name: str,
    structure: Union[
        Type[LinearPkEmulator], Type[NonLinearBoostPkEmulator]
    ] = LinearPkEmulator,
    artifacts_toml: Optional[str] = None,
    **kwargs,
) -> Union[LinearPkEmulator, NonLinearBoostPkEmulator]:
    """
    Load a trained emulator from an artifact defined in Artifacts.toml.
    """
    from pathlib import Path

    from fetch_artifacts import artifact

    if artifacts_toml is None:
        artifacts_toml = Path(__file__).parent.parent / "Artifacts.toml"

    emulator_path = artifact(artifact_name, toml_path=str(artifacts_toml))
    emulator_path = Path(emulator_path)

    if emulator_path.is_dir():
        if not (emulator_path / "nn_setup.json").exists():
            subdirs = [d for d in emulator_path.iterdir() if d.is_dir()]
            if len(subdirs) == 1:
                emulator_path = subdirs[0]

    return load_emulator(str(emulator_path), structure=structure, **kwargs)


def load_pk_emulator_from_artifact(
    artifact_name: str,
    artifacts_toml: Optional[str] = None,
    **kwargs,
) -> PkEmulator:
    """
    Load a complete PkEmulator suite from an artifact.
    """
    from pathlib import Path
    from fetch_artifacts import artifact

    if artifacts_toml is None:
        artifacts_toml = Path(__file__).parent.parent / "Artifacts.toml"

    emulator_path = artifact(artifact_name, toml_path=str(artifacts_toml))
    emulator_path = Path(emulator_path)

    # Handle case where tarball contains a single top-level directory
    if emulator_path.is_dir():
        # Check if expected subfolders exist directly
        has_subfolders = (emulator_path / "Pk_lin_mm").exists() or \
                         (emulator_path / "Boost").exists()
        
        if not has_subfolders:
            # Check if there is a single subdirectory containing them
            subdirs = [d for d in emulator_path.iterdir() if d.is_dir()]
            if len(subdirs) == 1:
                emulator_path = subdirs[0]

    return load_pk_emulator(str(emulator_path), **kwargs)


def compute_pca(data: Array, n_components: int):
    """
    Computes PCA on the training targets.
    Returns: mean vector, basis matrix, and PCA coefficients.
    """
    mu = jnp.mean(data, axis=1, keepdims=True)
    centered_data = data - mu
    u, s, vh = jnp.linalg.svd(centered_data, full_matrices=False)
    basis = u[:, :n_components]
    coefficients = jnp.dot(basis.T, centered_data)
    return jnp.squeeze(mu), basis, coefficients


def save_pca_metadata(path: str, mu: Array, basis: Array):
    """
    Saves PCA metadata needed for reconstruction.
    """
    jnp.save(os.path.join(path, "pca_mean.npy"), mu)
    jnp.save(os.path.join(path, "pca_projection.npy"), basis)
