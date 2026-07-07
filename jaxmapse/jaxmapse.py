import importlib.util
import json
import os
from functools import partial
from pathlib import Path
from typing import Callable, Mapping, Optional, Type, Union

import jax
import jax.numpy as jnp

# Import jaxace components
from jaxace import FlaxEmulator, init_emulator, inv_maximin, maximin
from jaxtyping import Array

from .builtins import BUILTIN_POSTPROCESSING, BUILTIN_PREPROCESSING, LOAD_PRESETS

# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)

DEFAULT_EMULATOR_ARTIFACT = "mnuw0wacdm_class"


def default_artifacts_toml() -> Path:
    """Return the packaged artifact registry path."""
    try:
        from importlib.resources import files

        return Path(str(files("jaxmapse") / "Artifacts.toml"))
    except Exception:
        return Path(__file__).parent / "Artifacts.toml"


def _interp_to_grid(source_k: Array, values: Array, target_k: Array) -> Array:
    """Interpolate one or more spectra from ``source_k`` onto ``target_k``."""
    source_k = jnp.asarray(source_k)
    target_k = jnp.asarray(target_k)
    values = jnp.asarray(values)

    # Perform validation if the grids are concrete (not JAX tracers)
    try:
        if len(source_k) > 0 and len(target_k) > 0:
            # Test if the inputs are concrete. This raises TracerBoolConversionError if tracing.
            bool(source_k[0] < target_k[0])

            if len(source_k) > 1 and source_k[1] < source_k[0]:
                raise ValueError("source_k must be monotonically increasing.")
            if len(target_k) > 1 and target_k[1] < target_k[0]:
                raise ValueError("target_k must be monotonically increasing.")
            if target_k[0] < source_k[0] or target_k[-1] > source_k[-1]:
                raise ValueError(
                    f"Target grid out of bounds: [{target_k[0]}, {target_k[-1]}] "
                    f"is outside source grid range [{source_k[0]}, {source_k[-1]}]."
                )
    except jax.errors.TracerBoolConversionError:
        pass

    if values.ndim == 1:
        return jnp.interp(target_k, source_k, values)
    return jax.vmap(lambda row: jnp.interp(target_k, source_k, row))(values)


class TransferFunctionEmulator:
    """
    TransferFunctionEmulator class representing a single cosmological transfer function or linear P(k) component.
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
        self.metadata = getattr(trained_emulator, "description", {})
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        self.name = self.metadata.get("emulator_description", {}).get("name")
        self.quantity = self.metadata.get("emulator_description", {}).get("quantity")

    def _decode_output(self, output: Array) -> Array:
        """Map NN output coefficients back to the emulator k-grid if PCA is used."""
        if self.pca_mean is None or self.pca_projection is None:
            return output
        return self.pca_mean + self.pca_projection @ output

    def _predict_single(self, input_params: Array, z: float, D: Optional[float] = None) -> Array:
        """Core implementation for a single parameter set and single redshift."""
        preprocessed_input = self.preprocessing(input_params)
        nn_input = jnp.insert(preprocessed_input, 0, z)
        norm_input = maximin(nn_input, self.in_minmax)
        norm_output = self.trained_emulator.run_emulator(norm_input)
        output = inv_maximin(norm_output, self.out_minmax)
        output = self._decode_output(output)
        return self.postprocessing(input_params, output, D, self)

    def predict(
        self, input_params: Array, z: Union[float, Array], D: Optional[Union[float, Array]] = None
    ) -> Array:
        """
        Compute prediction. Handles scalar or vector z/D via automatic vmap.
        """
        if not hasattr(self, "_jit_predict"):
            @partial(jax.jit, static_argnums=(0,))
            def _jit_predict(self, params, z, D):
                if jnp.ndim(z) == 0:
                    return self._predict_single(params, z, D)
                else:
                    return jax.vmap(self._predict_single, in_axes=(None, 0, 0 if D is not None else None))(
                        params, z, D
                    )
            self._jit_predict = _jit_predict

        return self._jit_predict(self, input_params, z, D)

    def __call__(
        self, input_params: Array, z: Union[float, Array], D: Optional[Union[float, Array]] = None
    ) -> Array:
        return self.predict(input_params, z, D)

    def get_Pk(
        self, input_params: Array, z: Union[float, Array], D: Optional[Union[float, Array]] = None
    ) -> Array:
        return self.predict(input_params, z, D)



def _evaluate_emu(emu, params, z, D):
    if hasattr(emu, "get_Pk"):
        return emu.get_Pk(params, z, D)
    return emu(params, z, D)


def halofit_pmm_from_emulator(
    input_params: Array,
    z: Union[float, Array],
    D: Optional[Union[float, Array]] = None,
    linear_pmm_emu: TransferFunctionEmulator = None,
    omega_m_z: Optional[Union[float, Array]] = None,
    omega_v_z: Optional[Union[float, Array]] = None,
) -> tuple[Array, Array]:
    """
    Evaluate the linear matter power spectrum from a single linear emulator,
    then apply Takahashi/Bird Halofit.
    """
    from jaxace.background import w0waCDMCosmology

    from .halofit import halofit_background, halofit_cosmology, halofit_pmm

    params = jnp.asarray(input_params)
    if params.ndim != 1 or params.shape[0] != 8:
        raise ValueError(
            "Halofit helpers expect flat mnuw0wacdm parameters in order "
            "[ln10As, ns, H0, omega_b, omega_c, Mnu, w0, wa]."
        )
    z_arr = jnp.asarray(z)

    if D is None:
        h = jnp.where(params[2] > 10.0, params[2] / 100.0, params[2])
        growth_cosmology = w0waCDMCosmology(
            ln10As=params[0],
            ns=params[1],
            h=h,
            omega_b=params[3],
            omega_c=params[4],
            m_nu=params[5],
            w0=params[6],
            wa=params[7],
        )
        D = growth_cosmology.D_z(z_arr)

    if linear_pmm_emu is None:
        raise ValueError("linear_pmm_emu must be provided.")

    pk_lin_mm = _evaluate_emu(linear_pmm_emu, params, z_arr, D)
    halofit_cpar = halofit_cosmology(params)

    if (omega_m_z is None) != (omega_v_z is None):
        raise ValueError("omega_m_z and omega_v_z must be provided together.")
    if omega_m_z is None:
        omega_m_z, omega_v_z = halofit_background(halofit_cpar, z_arr)

    pk_nl = halofit_pmm(
        halofit_cpar,
        z_arr,
        linear_pmm_emu.k_grid,
        pk_lin_mm,
        omega_m_z,
        omega_v_z,
    )
    return linear_pmm_emu.k_grid, pk_nl


get_halofit_pmm = halofit_pmm_from_emulator


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


def _function_name(name):
    if name is None:
        return None
    if isinstance(name, str):
        return name
    return str(name)


def _metadata_name(nn_dict: Mapping, key: str, explicit_name=None) -> Optional[str]:
    if explicit_name is not None:
        return _function_name(explicit_name)

    name = nn_dict.get(key)
    if name is None:
        description = nn_dict.get("emulator_description", {})
        if isinstance(description, Mapping):
            name = description.get(key)
    return _function_name(name)


def _load_component_function(
    path: str,
    nn_dict: Mapping,
    key: str,
    file_name: str,
    registry: Mapping[str, Callable],
    role: str,
    explicit_name=None,
) -> Callable:
    """Load a named builtin function or fall back to an artifact-local file."""
    name = _metadata_name(nn_dict, key, explicit_name)
    if name is not None:
        if name in registry:
            return registry[name]
        raise ValueError(
            f"{role} function {name!r} was requested in "
            f"{os.path.join(path, 'nn_setup.json')}, but it is not registered. "
            f"Register it in the corresponding jaxmapse BUILTIN_* dictionary, "
            f"or remove the metadata entry and provide {file_name}."
        )

    return _load_function(os.path.join(path, file_name), role)


def _load_preset(preset):
    if preset is None:
        return {}
    name = _function_name(preset)
    if name in LOAD_PRESETS:
        return LOAD_PRESETS[name]
    available = ", ".join(sorted(LOAD_PRESETS))
    raise ValueError(f"Unknown jaxmapse load preset {name!r}. Available presets: {available}")


def load_emulator(
    path: str,
    structure: Type[TransferFunctionEmulator] = TransferFunctionEmulator,
    preset: Optional[str] = None,
    **kwargs,
) -> TransferFunctionEmulator:
    """
    Load an emulator from disk mirroring the Julia load_emulator logic.
    """
    path_obj = Path(path)
    nn_setup_file = kwargs.get("nn_setup_file", "nn_setup.json")
    has_subfolders = (path_obj / "Pk_lin_mm").is_dir() or (path_obj / "Boost").is_dir()
    has_single_setup = (path_obj / nn_setup_file).is_file()

    if has_subfolders and not has_single_setup:
        raise ValueError(
            "load_emulator expects a single component directory containing nn_setup.json. "
            "The provided path appears to be a composite emulator bundle. "
            "Pass root / 'Pk_lin_mm' or root / 'Pk_lin_cb' instead."
        )

    if not has_single_setup:
        raise ValueError(
            f"No {nn_setup_file!r} found in {path}. Pass a single component emulator directory."
        )

    with open(
        os.path.join(path, nn_setup_file), "r"
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
    if has_pca and not (
        os.path.exists(pca_mean_file) and os.path.exists(pca_projection_file)
    ):
        raise FileNotFoundError(
            "PCA emulator output requires both pca_mean.npy and pca_projection.npy"
        )
    pca_mean = jnp.load(pca_mean_file) if has_pca else None
    pca_projection = jnp.load(pca_projection_file) if has_pca else None

    trained_emu = init_emulator(nn_dict, weights)

    load_preset = _load_preset(preset)
    preprocessing_name = kwargs.get(
        "preprocessing_name", load_preset.get("preprocessing_name")
    )
    postprocessing_name = kwargs.get(
        "postprocessing_name", load_preset.get("postprocessing_name")
    )

    preprocessing = _load_component_function(
        path,
        nn_dict,
        "preprocessing_name",
        kwargs.get("preprocessing_file", "preprocessing.py"),
        BUILTIN_PREPROCESSING,
        "preprocessing",
        preprocessing_name,
    )
    postprocessing = _load_component_function(
        path,
        nn_dict,
        "postprocessing_name",
        kwargs.get("postprocessing_file", "postprocessing.py"),
        BUILTIN_POSTPROCESSING,
        "postprocessing",
        postprocessing_name,
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


def artifact_path(
    name: str = DEFAULT_EMULATOR_ARTIFACT,
    artifacts_toml: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolve, download, and install a trained-emulator artifact, returning its local path.
    """
    from fetch_artifacts import artifact

    if artifacts_toml is None:
        artifacts_toml = default_artifacts_toml()

    emulator_path = Path(artifact(name, toml_path=str(artifacts_toml)))
    if emulator_path.is_dir() and not (emulator_path / "nn_setup.json").is_file():
        subdirs = [d for d in emulator_path.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            return subdirs[0]
    return emulator_path


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


_TRAINED_EMULATORS_CACHE = None


def load_trained_emulators(force_reload: bool = False):
    """Load and cache the built-in trained emulators as a small dictionary.

    The dictionary is keyed by artifact name and contains the linear Pmm and Pcb
    components loaded with the explicit single-component API.
    """
    global _TRAINED_EMULATORS_CACHE
    if _TRAINED_EMULATORS_CACHE is not None and not force_reload:
        return _TRAINED_EMULATORS_CACHE

    root = artifact_path(DEFAULT_EMULATOR_ARTIFACT)
    _TRAINED_EMULATORS_CACHE = {
        DEFAULT_EMULATOR_ARTIFACT: {
            "pmm": load_emulator(str(root / "Pk_lin_mm"), preset="mnuw0wacdm_linear"),
            "pcb": load_emulator(str(root / "Pk_lin_cb"), preset="mnuw0wacdm_linear"),
        }
    }
    return _TRAINED_EMULATORS_CACHE


def save_pca_metadata(path: str, mu: Array, basis: Array):
    """
    Saves PCA metadata needed for reconstruction.
    """
    jnp.save(os.path.join(path, "pca_mean.npy"), mu)
    jnp.save(os.path.join(path, "pca_projection.npy"), basis)


def _parse_params(params, kwargs):
    p_dict = {}
    if isinstance(params, dict):
        p_dict.update(params)
    elif params is not None:
        try:
            if hasattr(params, 'shape'):
                if params.ndim == 1 and params.shape[0] == 8:
                    return params
            elif hasattr(params, '__len__') and len(params) == 8:
                return jnp.asarray(params)
        except Exception:
            pass
        if hasattr(params, '_asdict'):
            p_dict.update(params._asdict())
        elif hasattr(params, '__dict__'):
            p_dict.update(params.__dict__)
            
    p_dict.update(kwargs)
    
    ln10As = p_dict.get('ln10As', p_dict.get('logA', p_dict.get('A_s', None)))
    if ln10As is None:
        raise ValueError('Missing parameter ln10As')
    ns = p_dict.get('ns', p_dict.get('n_s', None))
    if ns is None:
        raise ValueError('Missing parameter ns')
    H0 = p_dict.get('H0', None)
    h = p_dict.get('h', None)
    if H0 is None:
        if h is not None:
            H0 = h * 100.0
        else:
            raise ValueError('Missing parameter H0 or h')
    omega_b = p_dict.get('omega_b', p_dict.get('omega_b_h2', p_dict.get('ombh2', None)))
    if omega_b is None:
        raise ValueError('Missing parameter omega_b')
    omega_c = p_dict.get('omega_c', p_dict.get('omega_cdm', p_dict.get('omch2', None)))
    if omega_c is None:
        raise ValueError('Missing parameter omega_c')
    Mnu = p_dict.get('Mnu', p_dict.get('m_nu', p_dict.get('mnu', 0.0)))
    w0 = p_dict.get('w0', -1.0)
    wa = p_dict.get('wa', 0.0)
    
    return jnp.stack([ln10As, ns, H0, omega_b, omega_c, Mnu, w0, wa])


def hmcode_pmm_from_emulator(
    input_params: Optional[Union[Array, dict]] = None,
    z: Optional[Union[float, Array]] = None,
    *,
    D: Optional[Union[float, Array]] = None,
    linear_pmm_emu: Optional[TransferFunctionEmulator] = None,
    linear_pcb_emu: Optional[TransferFunctionEmulator] = None,
    T_AGN: Optional[float] = None,
    accuracy: float = 1.0,
    nM: Optional[int] = None,
    k_out: Optional[Array] = None,
    **kwargs,
) -> tuple[Array, Array]:
    """
    Evaluate the linear matter power spectrum (and optional cb spectrum) from emulators,
    compute sigma_8 natively, and apply HMCode2020 non-linear correction.

    Parameters:
    -----------
    input_params: Array or dict, optional
        Cosmological parameters. Can be a flat array in order [ln10As, ns, H0, omega_b, omega_c, Mnu, w0, wa],
        or a dictionary with keys (e.g. ln10As, ns, h, ombh2, omch2, mnu, w0, wa).
    z: float or Array, optional
        Redshift(s) at which to evaluate the power spectrum.
    D: float or Array, optional
        Linear growth factor. If None, it is calculated natively in JAX from the input parameters.
    linear_pmm_emu: TransferFunctionEmulator, optional
        The linear matter power spectrum emulator. If None, loads the default built-in emulator.
    linear_pcb_emu: TransferFunctionEmulator, optional
        The linear cb (baryons + CDM) power spectrum emulator. If None, loads the default built-in emulator.
    T_AGN: float, optional
        Baryon feedback temperature log10(T_AGN/K). If None, uses Dark Matter Only (DMO).
    accuracy: float, optional
        HMCode integration accuracy (default: 1.0).
    nM: int, optional
        Number of mass integration steps (default: 256 * accuracy).
    k_out: Array, optional
        Custom physical wavenumbers (Mpc^-1) for output. If None, uses the emulator's k grid.
    kwargs:
        Cosmological parameters can also be passed directly as keyword arguments.

    Returns:
    --------
    k: Array
        Wavenumbers in physical units (Mpc^-1).
    pk_nl: Array
        Non-linear matter power spectrum in physical units (Mpc^3).
    """
    from jaxace.background import w0waCDMCosmology
    from .hmcode import HMCodeCosmology, hmcode_pmm_jax, _sigma_grid_jax

    # Determine if the first argument was actually the redshift z
    is_first_arg_z = False
    if input_params is not None:
        if isinstance(input_params, (float, int)):
            is_first_arg_z = True
        elif not isinstance(input_params, dict) and hasattr(input_params, "__len__"):
            if len(input_params) != 8:
                is_first_arg_z = True

    if is_first_arg_z:
        if z is not None:
            raise ValueError("z passed both positionally and as keyword/second argument.")
        z = input_params
        input_params = None

    if z is None:
        raise ValueError("Missing required parameter 'z'.")

    params = _parse_params(input_params, kwargs)
    z_arr = jnp.atleast_1d(z)
    h = jnp.where(params[2] > 10.0, params[2] / 100.0, params[2])

    if D is None:
        growth_cosmology = w0waCDMCosmology(
            ln10As=params[0],
            ns=params[1],
            h=h,
            omega_b=params[3],
            omega_c=params[4],
            m_nu=params[5],
            w0=params[6],
            wa=params[7],
        )
        D = growth_cosmology.D_z(z_arr)

    # Load emulators if not provided
    if linear_pmm_emu is None or linear_pcb_emu is None:
        emus = load_trained_emulators()[DEFAULT_EMULATOR_ARTIFACT]
        if linear_pmm_emu is None:
            linear_pmm_emu = emus["pmm"]
        if linear_pcb_emu is None:
            linear_pcb_emu = emus["pcb"]

    k_support = linear_pmm_emu.k_grid
    if k_out is None:
        k = k_support
    else:
        k = jnp.asarray(k_out)

    # Predict linear spectra (in physical units)
    pk_lin_mm = _evaluate_emu(linear_pmm_emu, params, z_arr, D)
    pk_lin_cb = _evaluate_emu(linear_pcb_emu, params, z_arr, D)

    # Convert linear spectra to h-units for jaxmapse input
    k_h = k / h
    k_support_h = k_support / h
    Pmm_lin_h = pk_lin_mm * (h**3)
    Pcb_lin_h = pk_lin_cb * (h**3)

    # Predict at z=0 to calculate sigma_8 natively
    pk_lin_mm_z0 = _evaluate_emu(linear_pmm_emu, params, 0.0, 1.0)
    Pmm_lin_h_z0 = pk_lin_mm_z0 * (h**3)
    sigma_8_jax = _sigma_grid_jax(k_support_h, Pmm_lin_h_z0[None, :], jnp.array([8.0]))[0, 0]

    # Setup jaxmapse cosmology
    omega_nu = (params[5] / 93.14) / h**2
    omega_m = (params[3] + params[4]) / h**2 + omega_nu
    omega_b_h2 = params[3] / h**2
    hmcode_cosmo = HMCodeCosmology(
        Omega_m=omega_m,
        Omega_b=omega_b_h2,
        h=h,
        n_s=params[1],
        sigma_8=sigma_8_jax,
        w0=params[6],
        wa=params[7],
        Omega_nu=omega_nu,
        Omega_k=0.0,
    )

    # Ensure 2D arrays for JAX-native solver
    import math
    Pmm_lin_h_2d = jnp.atleast_2d(Pmm_lin_h)
    Pcb_lin_h_2d = jnp.atleast_2d(Pcb_lin_h)
    if nM is None:
        nM = int(math.ceil(256 * accuracy))

    # Solve non-linear HMCode2020 natively in JAX (JIT friendly)
    Pmm_jax_h = hmcode_pmm_jax(
        hmcode_cosmo,
        z_arr,
        k_h,
        k_support_h,
        Pmm_lin_h_2d,
        Pcb_lin_h_2d,
        T_AGN=10.0**7.8 if T_AGN is None else T_AGN,
        Mmin=1.0,
        Mmax=1.0e18,
        nM=nM,
        include_feedback=T_AGN is not None,
    )

    # Convert output back to physical units and match input dimension
    pk_nl_2d = Pmm_jax_h / (h**3)
    pk_nl = pk_nl_2d[0] if jnp.ndim(z) == 0 else pk_nl_2d

    return k, pk_nl


get_hmcode_pmm = hmcode_pmm_from_emulator

