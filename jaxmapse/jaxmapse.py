import jax
import jax.numpy as jnp
from typing import Union, Callable, Type
import os
import json
import importlib.util
from jaxtyping import Array
from functools import partial

# Import jaxace components
from jaxace import (
    init_emulator,
    FlaxEmulator,
    maximin,
    inv_maximin
)

# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)

class LinearPkEmulator:
    """
    Linear power spectrum emulator mirroring Mapse.jl's LinearPkEmulator.
    """
    def __init__(self, 
                 trained_emulator: FlaxEmulator, 
                 k_grid: Array, 
                 in_minmax: Array, 
                 out_minmax: Array, 
                 preprocessing: Callable, 
                 postprocessing: Callable):
        self.trained_emulator = trained_emulator
        self.k_grid = jnp.asarray(k_grid)
        self.in_minmax = jnp.asarray(in_minmax)
        self.out_minmax = jnp.asarray(out_minmax)
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def _get_Pk_single(self, input_params: Array, z: float, D: float) -> Array:
        """Core implementation for a single parameter set and single redshift."""
        preprocessed_input = self.preprocessing(input_params)
        nn_input = jnp.append(preprocessed_input, z)
        norm_input = maximin(nn_input, self.in_minmax)
        norm_output = self.trained_emulator.run_emulator(norm_input)
        output = inv_maximin(norm_output, self.out_minmax)
        return self.postprocessing(input_params, output, D, self)

    def get_Pk(self, 
               input_params: Array, 
               z: Union[float, Array], 
               D: Union[float, Array]) -> Array:
        """
        Compute linear power spectrum. Handles scalar or vector z/D via automatic vmap.
        """
        if not hasattr(self, '_jit_get_Pk'):
            @partial(jax.jit, static_argnums=(0,))
            def _jit_get_Pk(self, params, z, D):
                if jnp.ndim(z) == 0:
                    return self._get_Pk_single(params, z, D)
                else:
                    return jax.vmap(self._get_Pk_single, in_axes=(None, 0, 0))(params, z, D)
            self._jit_get_Pk = _jit_get_Pk
            
        return self._jit_get_Pk(self, input_params, z, D)

class NonLinearBoostPkEmulator:
    """
    Non-linear boost emulator mirroring Mapse.jl's NonLinearBoostPkEmulator.
    """
    def __init__(self, 
                 trained_emulator: FlaxEmulator, 
                 k_grid: Array, 
                 in_minmax: Array, 
                 out_minmax: Array, 
                 postprocessing: Callable):
        self.trained_emulator = trained_emulator
        self.k_grid = jnp.asarray(k_grid)
        self.in_minmax = jnp.asarray(in_minmax)
        self.out_minmax = jnp.asarray(out_minmax)
        self.postprocessing = postprocessing

    def _get_Pk_single(self, input_params: Array, z: float) -> Array:
        """Core implementation for a single parameter set and single redshift."""
        nn_input = jnp.append(input_params, z)
        norm_input = maximin(nn_input, self.in_minmax)
        norm_output = self.trained_emulator.run_emulator(norm_input)
        output = inv_maximin(norm_output, self.out_minmax)
        return self.postprocessing(input_params, output, self)

    def get_Pk(self, 
               input_params: Array, 
               z: Union[float, Array]) -> Array:
        """Compute boost factor. Handles scalar or vector z via automatic vmap."""
        if not hasattr(self, '_jit_get_Pk'):
            @partial(jax.jit, static_argnums=(0,))
            def _jit_get_Pk(self, params, z):
                if jnp.ndim(z) == 0:
                    return self._get_Pk_single(params, z)
                else:
                    return jax.vmap(self._get_Pk_single, in_axes=(None, 0))(params, z)
            self._jit_get_Pk = _jit_get_Pk
            
        return self._jit_get_Pk(self, input_params, z)

class PkEmulator:
    """
    Composite emulator combining linear and non-linear components.
    """
    def __init__(self, 
                 linear_pmm: LinearPkEmulator, 
                 linear_pkcb: LinearPkEmulator, 
                 boost: NonLinearBoostPkEmulator):
        self.linear_pmm = linear_pmm
        self.linear_pkcb = linear_pkcb
        self.boost = boost

    def get_Pk(self, input_params: Array, z: Union[float, Array], D: Union[float, Array]) -> Array:
        """Returns P_mm,lin * Boost."""
        if not hasattr(self, '_jit_get_Pk'):
            @partial(jax.jit, static_argnums=(0,))
            def _jit_get_Pk(self, params, z, D):
                lin = self.linear_pmm.get_Pk(params, z, D)
                bst = self.boost.get_Pk(params, z)
                return lin * bst
            self._jit_get_Pk = _jit_get_Pk
        return self._jit_get_Pk(self, input_params, z, D)

    def get_linear_pmm(self, input_params: Array, z: Union[float, Array], D: Union[float, Array]) -> Array:
        """Returns linear matter power spectrum."""
        return self.linear_pmm.get_Pk(input_params, z, D)

    def get_linear_pkcb(self, input_params: Array, z: Union[float, Array], D: Union[float, Array]) -> Array:
        """Returns linear c+b power spectrum."""
        return self.linear_pkcb.get_Pk(input_params, z, D)

def _load_function(filepath: str, func_name: str) -> Callable:
    """Helper to load a function from a python file."""
    spec = importlib.util.spec_from_file_location("module.name", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, func_name):
        raise ValueError(f"File {filepath} must define a '{func_name}' function")
    return getattr(module, func_name)

def load_emulator(path: str, 
                  structure: Union[Type[LinearPkEmulator], Type[NonLinearBoostPkEmulator]] = LinearPkEmulator,
                  **kwargs) -> Union[LinearPkEmulator, NonLinearBoostPkEmulator]:
    """
    Load an emulator from disk mirroring the Julia load_emulator logic.
    """
    with open(os.path.join(path, kwargs.get("nn_setup_file", "nn_setup.json")), 'r') as f:
        nn_dict = json.load(f)
    
    weights = jnp.load(os.path.join(path, kwargs.get("weights_file", "weights.npy")))
    k_grid = jnp.load(os.path.join(path, kwargs.get("k_file", "k.npy")))
    in_minmax = jnp.load(os.path.join(path, kwargs.get("inminmax_file", "inminmax.npy")))
    out_minmax = jnp.load(os.path.join(path, kwargs.get("outminmax_file", "outminmax.npy")))
    
    trained_emu = init_emulator(nn_dict, weights)
    
    postprocessing = _load_function(
        os.path.join(path, kwargs.get("postprocessing_file", "postprocessing.py")), 
        "postprocessing"
    )
    
    if structure == LinearPkEmulator:
        preprocessing = _load_function(
            os.path.join(path, kwargs.get("preprocessing_file", "preprocessing.py")), 
            "preprocessing"
        )
        return LinearPkEmulator(
            trained_emulator=trained_emu,
            k_grid=k_grid,
            in_minmax=in_minmax,
            out_minmax=out_minmax,
            preprocessing=preprocessing,
            postprocessing=postprocessing
        )
    elif structure == NonLinearBoostPkEmulator:
        return NonLinearBoostPkEmulator(
            trained_emulator=trained_emu,
            k_grid=k_grid,
            in_minmax=in_minmax,
            out_minmax=out_minmax,
            postprocessing=postprocessing
        )
    else:
        raise ValueError(f"Unknown structure: {structure}")

def load_emulator_from_artifact(
    artifact_name: str,
    structure: Union[Type[LinearPkEmulator], Type[NonLinearBoostPkEmulator]] = LinearPkEmulator,
    artifacts_toml: Optional[str] = None,
    **kwargs
) -> Union[LinearPkEmulator, NonLinearBoostPkEmulator]:
    """
    Load a trained emulator from an artifact defined in Artifacts.toml.
    """
    from fetch_artifacts import artifact
    from pathlib import Path

    if artifacts_toml is None:
        artifacts_toml = Path(__file__).parent / "Artifacts.toml"

    emulator_path = artifact(artifact_name, toml_path=str(artifacts_toml))
    emulator_path = Path(emulator_path)

    if emulator_path.is_dir():
        if not (emulator_path / "nn_setup.json").exists():
            subdirs = [d for d in emulator_path.iterdir() if d.is_dir()]
            if len(subdirs) == 1:
                emulator_path = subdirs[0]

    return load_emulator(str(emulator_path), structure=structure, **kwargs)
