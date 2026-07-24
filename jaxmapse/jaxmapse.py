import importlib.util
import json
import os
from functools import partial
from pathlib import Path
from typing import Callable, Mapping, Optional, Type, Union

import jax
import jax.numpy as jnp
import numpy as np

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

    # Validate only concrete host grids. The traced kernel assumes immutable,
    # previously validated emulator grids.
    try:
        source_host = np.asarray(source_k)
        target_host = np.asarray(target_k)
        if source_host.ndim != 1 or target_host.ndim != 1:
            raise ValueError("source_k and target_k must be one-dimensional.")
        if source_host.size == 0 or target_host.size == 0:
            raise ValueError("source_k and target_k must be non-empty.")
        if np.any(np.diff(source_host) <= 0.0):
            raise ValueError("source_k must be strictly increasing.")
        if np.any(np.diff(target_host) <= 0.0):
            raise ValueError("target_k must be strictly increasing.")
        if target_host[0] < source_host[0] or target_host[-1] > source_host[-1]:
            raise ValueError(
                f"Target grid out of bounds: [{target_host[0]}, {target_host[-1]}] "
                f"is outside source grid range [{source_host[0]}, {source_host[-1]}]."
            )
    except jax.errors.TracerArrayConversionError:
        pass

    if values.ndim == 1:
        return jnp.interp(target_k, source_k, values)
    return jax.vmap(lambda row: jnp.interp(target_k, source_k, row))(values)


class TransferFunctionEmulator:
    """
    Single cosmological transfer-function or linear-P(k) component.

    Instances are immutable after their first prediction. ``predict`` caches a
    JIT executable with ``self`` static, so changing model, normalization, PCA,
    or function attributes after that call can serve stale compiled state.
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

    def _predict_single(
        self, input_params: Array, z: float, D: Optional[float] = None
    ) -> Array:
        """Core implementation for a single parameter set and single redshift."""
        preprocessed_input = self.preprocessing(input_params)
        nn_input = jnp.insert(preprocessed_input, 0, z)
        norm_input = maximin(nn_input, self.in_minmax)
        norm_output = self.trained_emulator.run_emulator(norm_input)
        output = inv_maximin(norm_output, self.out_minmax)
        output = self._decode_output(output)
        return self.postprocessing(input_params, output, D, self)

    def predict(
        self,
        input_params: Array,
        z: Union[float, Array],
        D: Optional[Union[float, Array]] = None,
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
                    return jax.vmap(
                        self._predict_single,
                        in_axes=(None, 0, 0 if D is not None else None),
                    )(params, z, D)

            self._jit_predict = _jit_predict

        return self._jit_predict(self, input_params, z, D)

    def __call__(
        self,
        input_params: Array,
        z: Union[float, Array],
        D: Optional[Union[float, Array]] = None,
    ) -> Array:
        return self.predict(input_params, z, D)

    def get_Pk(
        self,
        input_params: Array,
        z: Union[float, Array],
        D: Optional[Union[float, Array]] = None,
    ) -> Array:
        return self.predict(input_params, z, D)


def _evaluate_emu(emu, params, z, D):
    if hasattr(emu, "get_Pk"):
        return emu.get_Pk(params, z, D)
    return emu(params, z, D)


def halofit_pmm_from_emulator(
    input_params: Array,
    z: Union[float, Array],
    *,
    linear_pmm_emu: TransferFunctionEmulator,
    D: Optional[Union[float, Array]] = None,
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


def _load_function(filepath: str, func_name: str) -> Callable:
    """Load a legacy artifact-local hook from trusted executable Python code."""
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
    raise ValueError(
        f"Unknown jaxmapse load preset {name!r}. Available presets: {available}"
    )


def _validate_component_shapes(
    path: str,
    k_grid: Array,
    in_minmax: Array,
    out_minmax: Array,
    pca_mean: Optional[Array],
    pca_projection: Optional[Array],
    nn_dict: Mapping,
) -> None:
    """Validate artifact arrays before constructing an emulator.

    This mirrors Mapse.jl's loader checks so malformed artifacts fail at the
    boundary with an actionable error, rather than later inside normalization or
    PCA reconstruction under JIT.
    """
    try:
        n_input = int(nn_dict["n_input_features"])
        n_output = int(nn_dict["n_output_features"])
    except KeyError as exc:
        raise ValueError(f"{path}: nn_setup.json is missing {exc.args[0]!r}.") from exc

    if tuple(in_minmax.shape) != (n_input, 2):
        raise ValueError(
            f"{path}: inminmax.npy has shape {tuple(in_minmax.shape)}; "
            f"expected ({n_input}, 2)."
        )
    if tuple(out_minmax.shape) != (n_output, 2):
        raise ValueError(
            f"{path}: outminmax.npy has shape {tuple(out_minmax.shape)}; "
            f"expected ({n_output}, 2)."
        )
    if len(k_grid.shape) != 1:
        raise ValueError(f"{path}: k.npy must be one-dimensional.")

    if pca_mean is None:
        if pca_projection is not None:
            raise ValueError(f"{path}: PCA projection requires pca_mean.npy.")
        if k_grid.shape[0] != n_output:
            raise ValueError(
                f"{path}: k-grid has length {k_grid.shape[0]} but uncompressed "
                f"NN output has {n_output} features."
            )
        return

    if pca_projection is None:
        raise ValueError(f"{path}: pca_mean.npy requires PCA projection metadata.")
    if pca_mean.ndim != 1 or pca_mean.shape[0] != k_grid.shape[0]:
        raise ValueError(
            f"{path}: pca_mean.npy has shape {tuple(pca_mean.shape)}; expected "
            f"({k_grid.shape[0]},)."
        )
    if tuple(pca_projection.shape) != (k_grid.shape[0], n_output):
        raise ValueError(
            f"{path}: PCA projection has shape {tuple(pca_projection.shape)}; "
            f"expected ({k_grid.shape[0]}, {n_output})."
        )


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

    with open(os.path.join(path, nn_setup_file), "r") as f:
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

    _validate_component_shapes(
        path,
        k_grid,
        in_minmax,
        out_minmax,
        pca_mean,
        pca_projection,
        nn_dict,
    )

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
            "pmm": load_emulator(str(root / "Pk_lin_mm")),
            "pcb": load_emulator(str(root / "Pk_lin_cb")),
        }
    }
    return _TRAINED_EMULATORS_CACHE


class _LazyTrainedEmulators(dict):
    """Dict-compatible lazy registry for the official trained emulators."""

    def _load(self):
        if not self:
            loaded = load_trained_emulators()
            super().update(
                {
                    name: _TrainedEmulatorBundle(components)
                    for name, components in loaded.items()
                }
            )
        return self

    def get(self, key, default=None):
        return dict.get(self._load(), key, default)

    def __getitem__(self, key):
        return dict.__getitem__(self._load(), key)

    def __contains__(self, key):
        return key in self._load()


class _TrainedEmulatorBundle:
    """Compatibility view of the official Pmm/Pcb emulator pair."""

    def __init__(self, components):
        self.linear_pmm = components["pmm"]
        self.linear_pcb = components["pcb"]

    def __getitem__(self, key):
        return {"pmm": self.linear_pmm, "pcb": self.linear_pcb}[key]

    def get_linear_pmm(self, input_params, z, D=None):
        return self.linear_pmm.get_Pk(input_params, z, D)

    def get_linear_pkcb(self, input_params, z, D=None):
        return self.linear_pcb.get_Pk(input_params, z, D)


# Public registry retained for notebook and user workflows. Loading remains
# lazy, so importing jaxmapse does not download artifacts unexpectedly.
trained_emulators = _LazyTrainedEmulators()


def _parse_params(params, kwargs):
    p_dict = {}
    if isinstance(params, dict):
        p_dict.update(params)
    elif params is not None:
        try:
            if hasattr(params, "shape"):
                if params.ndim == 1 and params.shape[0] == 8:
                    return params
            elif hasattr(params, "__len__") and len(params) == 8:
                return jnp.asarray(params)
        except Exception:
            pass
        if hasattr(params, "_asdict"):
            p_dict.update(params._asdict())
        elif hasattr(params, "__dict__"):
            p_dict.update(params.__dict__)

    p_dict.update(kwargs)

    if p_dict.get("ln10As") is not None:
        ln10As = p_dict["ln10As"]
    elif p_dict.get("A_s") is not None:
        ln10As = jnp.log(1.0e10 * p_dict["A_s"])
    elif p_dict.get("logA") is not None:
        raise ValueError("logA is ambiguous; pass ln10As or A_s explicitly.")
    else:
        raise ValueError("Missing parameter ln10As or A_s.")
    ns = p_dict.get("ns", p_dict.get("n_s", None))
    if ns is None:
        raise ValueError("Missing parameter ns")
    H0 = p_dict.get("H0", None)
    h = p_dict.get("h", None)
    if H0 is None:
        if h is not None:
            H0 = h * 100.0
        else:
            raise ValueError("Missing parameter H0 or h")
    omega_b = p_dict.get("omega_b", p_dict.get("omega_b_h2", p_dict.get("ombh2", None)))
    if omega_b is None:
        raise ValueError("Missing parameter omega_b")
    omega_c = p_dict.get("omega_c", p_dict.get("omega_cdm", p_dict.get("omch2", None)))
    if omega_c is None:
        raise ValueError("Missing parameter omega_c")
    Mnu = p_dict.get("Mnu", p_dict.get("m_nu", p_dict.get("mnu", 0.0)))
    w0 = p_dict.get("w0", -1.0)
    wa = p_dict.get("wa", 0.0)

    return jnp.stack([ln10As, ns, H0, omega_b, omega_c, Mnu, w0, wa])


def hmcode_pmm_from_emulator(
    input_params: Optional[Union[Array, dict]] = None,
    z: Optional[Union[float, Array]] = None,
    *,
    linear_pmm_emu: TransferFunctionEmulator,
    linear_pcb_emu: TransferFunctionEmulator,
    D: Optional[Union[float, Array]] = None,
    T_AGN: Optional[float] = None,
    nM: int = 128,
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
    linear_pmm_emu: TransferFunctionEmulator, required
        The linear total-matter power spectrum emulator.
    linear_pcb_emu: TransferFunctionEmulator, required
        The linear cold+baryon (cb) power spectrum emulator.
    T_AGN: float, optional
        Baryon feedback temperature in Kelvin. The conventional feedback value
        is ``10.0**7.8`` K. If None, uses Dark Matter Only (DMO).
    nM: int, optional
        Number of mass integration steps (default: 128).
    k_out: Array, optional
        Custom output wavenumbers in **physical** units (Mpc^-1), not h-units.
        If None, uses the emulator's native k grid, which is also physical.
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

    from .hmcode import HMCodeCosmology, hmcode_pmm_jax

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
            raise ValueError(
                "z passed both positionally and as keyword/second argument."
            )
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

    k_support = linear_pmm_emu.k_grid
    if k_out is None:
        k = k_support
    else:
        k = jnp.asarray(k_out)

    # Predict linear spectra (in physical units)
    pk_lin_mm = _evaluate_emu(linear_pmm_emu, params, z_arr, D)
    pk_lin_cb = _evaluate_emu(linear_pcb_emu, params, z_arr, D)

    # Setup jaxmapse cosmology
    omega_nu = (params[5] / 93.14) / h**2
    omega_m = (params[3] + params[4]) / h**2 + omega_nu
    omega_b_h2 = params[3] / h**2
    hmcode_cosmo = HMCodeCosmology(
        Omega_m=omega_m,
        Omega_b=omega_b_h2,
        h=h,
        n_s=params[1],
        # HMCode keeps this legacy field for API compatibility but does not use it.
        sigma_8=0.0,
        w0=params[6],
        wa=params[7],
        Omega_nu=omega_nu,
        Omega_k=0.0,
    )

    pk_lin_mm_2d = jnp.atleast_2d(pk_lin_mm)
    pk_lin_cb_2d = jnp.atleast_2d(pk_lin_cb)
    if nM is None:
        nM = 128

    # Solve non-linear HMCode2020 natively in JAX (JIT friendly)
    pk_nl_2d = hmcode_pmm_jax(
        hmcode_cosmo,
        z_arr,
        k,
        k_support,
        pk_lin_mm_2d,
        pk_lin_cb_2d,
        T_AGN=10.0**7.8 if T_AGN is None else T_AGN,
        Mmin=1.0,
        Mmax=1.0e18,
        nM=nM,
        include_feedback=T_AGN is not None,
    )

    # Match the input redshift dimension.
    pk_nl = pk_nl_2d[0] if jnp.ndim(z) == 0 else pk_nl_2d

    return k, pk_nl


def hmcode_pmm_from_emulator_fast(
    input_params: Optional[Union[Array, dict]] = None,
    z: Optional[Union[float, Array]] = None,
    z_coarse: Optional[Array] = None,
    z_fine: Optional[Union[float, Array]] = None,
    N_z_coarse: int = 50,
    *,
    linear_pmm_emu: TransferFunctionEmulator,
    linear_pcb_emu: TransferFunctionEmulator,
    D: Optional[Union[float, Array]] = None,
    T_AGN: Optional[float] = None,
    nM: int = 128,
    k_out: Optional[Array] = None,
    piecewise_z_feature: Optional[float] = None,
    piecewise_split_index: Optional[int] = None,
    **kwargs,
) -> tuple[Array, Array]:
    """
    Fast end-to-end evaluation of the non-linear matter power spectrum on redshift grid `z`
    (or target grid `z_fine`) by running the core pipeline on a coarse grid (either custom
    `z_coarse` or auto-generated with `N_z_coarse` nodes) and interpolating the non-linear
    results via Akima splines.

    Warning:
        This fast API is an approximation. Instead of evaluating the full non-linear
        HMCode equations on the high-fidelity `z` (or `z_fine`) grid, it solves HMCode
        on the coarse grid (either `z_coarse` or `N_z_coarse` linearly spaced nodes) and
        uses Akima splines to reconstruct the results.

        - Typical Errors: Redshift interpolation errors are generally small but largest
          at high k, high redshift, and in regions where the nonlinear boost factor
          evolves rapidly.
        - Recommended Coarse Grid Size:
            - `N_z_coarse = 10` is NOT precision-safe and can introduce percent-level artifacts.
            - `N_z_coarse = 50` is a reasonable compromise between speed and accuracy.
            - `N_z_coarse = 100` is safer, typically guaranteeing sub-percent worst-case
              accuracy compared to full direct evaluation in studied regimes.
        - Validation: Users are advised to validate the accuracy of this fast
          path against the direct `hmcode_pmm_from_emulator` function for their specific
          redshift and k ranges.

    Preferred Production Pattern:
        Using this smart/coarse-grid API is the preferred production pattern:
        - Provide custom `z_coarse` and `z_fine` manually.
        - This allows linear emulators and HMCode to execute exclusively on the coarse
          redshift grid, and only the final non-linear spectrum is reconstructed on `z_fine`.
        - This avoids running the linear/transfer emulators on the dense fine redshift grid.
    """
    from jaxace.utils import akima_interpolation

    from .hmcode import _piecewise_akima_interpolation

    if N_z_coarse < 5:
        raise ValueError("N_z_coarse must be at least 5 for Akima interpolation.")
    if piecewise_z_feature is not None and piecewise_split_index is None:
        raise ValueError("piecewise_split_index is required with piecewise_z_feature.")

    if z_coarse is None:
        if z is None:
            raise ValueError("Either z or z_coarse must be provided.")
        z_arr = jnp.atleast_1d(z)

        # If the redshift grid is small, evaluate directly without interpolation
        if z_arr.shape[0] <= N_z_coarse or z_arr.shape[0] < 5:
            return hmcode_pmm_from_emulator(
                input_params=input_params,
                z=z,
                D=D,
                linear_pmm_emu=linear_pmm_emu,
                linear_pcb_emu=linear_pcb_emu,
                T_AGN=T_AGN,
                nM=nM,
                k_out=k_out,
                **kwargs,
            )

        if D is not None:
            raise ValueError(
                "Growth factor D is not supported on the fast/interpolated path. Please use direct evaluation or pass D=None."
            )

        # Generate coarse redshift grid
        z_min, z_max = jnp.min(z_arr), jnp.max(z_arr)
        _z_coarse = jnp.linspace(z_min, z_max, N_z_coarse)
        _z_fine = z_arr
        is_scalar = jnp.ndim(z) == 0
    else:
        if z_fine is None:
            raise ValueError("z_fine must be provided if z_coarse is specified.")
        if D is not None:
            raise ValueError(
                "Growth factor D is not supported on the fast/interpolated path. Please use direct evaluation or pass D=None."
            )
        _z_coarse = z_coarse
        _z_fine = jnp.atleast_1d(z_fine)
        is_scalar = jnp.ndim(z_fine) == 0

    # Redshift validation checks for concrete inputs
    if not isinstance(_z_coarse, jax.core.Tracer):
        z_c_np = np.asarray(_z_coarse)
        if len(z_c_np) < 5:
            raise ValueError(
                "z_coarse must have at least 5 points for Akima interpolation."
            )
        if np.any(np.diff(z_c_np) <= 0.0):
            raise ValueError("z_coarse must be strictly increasing.")

    if not isinstance(_z_fine, jax.core.Tracer):
        z_f_np = np.asarray(_z_fine)
        if len(z_f_np) > 1 and np.any(np.diff(z_f_np) <= 0.0):
            raise ValueError("z_fine must be strictly increasing.")
        if not isinstance(_z_coarse, jax.core.Tracer):
            z_c_np = np.asarray(_z_coarse)
            if np.any(z_f_np < z_c_np[0]) or np.any(z_f_np > z_c_np[-1]):
                raise ValueError("z_fine must lie within the range of z_coarse.")

    # Evaluate coarse non-linear power spectrum
    # (Note: growth factor D is automatically re-evaluated internally on _z_coarse)
    k, pk_nl_coarse = hmcode_pmm_from_emulator(
        input_params=input_params,
        z=_z_coarse,
        D=None,
        linear_pmm_emu=linear_pmm_emu,
        linear_pcb_emu=linear_pcb_emu,
        T_AGN=T_AGN,
        nM=nM,
        k_out=k_out,
        **kwargs,
    )

    # Optionally split the interpolation at a known baryonic feature.
    if piecewise_z_feature is None:
        pk_nl_fine = akima_interpolation(pk_nl_coarse, _z_coarse, _z_fine)
    else:
        pk_nl_fine = _piecewise_akima_interpolation(
            pk_nl_coarse,
            _z_coarse,
            _z_fine,
            piecewise_z_feature,
            split_index=piecewise_split_index,
        )

    # Restore original scalar/vector shape
    pk_nl = pk_nl_fine[0] if is_scalar else pk_nl_fine

    return k, pk_nl


def predict_baryonic_discontinuity(
    input_params: Optional[Union[Array, dict]] = None,
    T_AGN: float = 10.0**7.8,
    **kwargs,
) -> float:
    """Predict the baryonic feedback threshold/feature redshift z_discontinuity."""
    params = _parse_params(input_params, kwargs)
    h = jnp.where(params[2] > 10.0, params[2] / 100.0, params[2])
    logT_AGN = jnp.log10(T_AGN)
    sbar = -0.0030 * (logT_AGN - 7.8) + 0.0201
    sbarz = 0.0224 * (logT_AGN - 7.8) + 0.409
    omega_b = params[3] / h**2
    omega_c = params[4] / h**2
    omega_nu = (params[5] / 93.14) / h**2
    omega_m = omega_b + omega_c + omega_nu
    return (jnp.log10(omega_b / omega_m) - jnp.log10(sbar)) / sbarz


def build_smart_coarse_grid(
    z_min: float,
    z_max: float,
    N_coarse: int,
    z_feature: Optional[float] = None,
    min_spacing: float = 1e-4,
    N_left: Optional[int] = None,
) -> Array:
    """Construct a fixed-shape coarse grid with an optional shared feature node."""
    if N_coarse < 5:
        raise ValueError("N_coarse must be at least 5 for Akima interpolation.")
    if z_feature is None:
        return jnp.linspace(z_min, z_max, N_coarse)

    if N_left is None:
        N_left = _baryonic_left_nodes(N_coarse)
    N_right = N_coarse - N_left + 1
    if N_left < 5 or N_right < 5:
        raise ValueError("Each baryonic coarse-grid segment requires at least 5 nodes.")

    z_feature = _clip_baryonic_feature(z_min, z_max, z_feature, min_spacing)
    z_min = jnp.asarray(z_min)
    z_max = jnp.asarray(z_max)

    z_left = jnp.linspace(z_min, z_feature, N_left)
    z_right = jnp.linspace(z_feature, z_max, N_right)
    return jnp.concatenate((z_left, z_right[1:]))


def _baryonic_left_nodes(N_coarse: int) -> int:
    return int(round((N_coarse - 1) * 5.0 / 8.0)) + 1


def _clip_baryonic_feature(z_min, z_max, z_feature, min_spacing):
    z_min = jnp.asarray(z_min)
    z_max = jnp.asarray(z_max)
    z_feature = jnp.asarray(z_feature)
    dtype = jnp.result_type(z_min, z_max, z_feature)
    margin = jnp.maximum(
        jnp.asarray(min_spacing, dtype=dtype),
        jnp.asarray(0.05, dtype=dtype) * (z_max - z_min),
    )
    return jnp.clip(z_feature, z_min + margin, z_max - margin)


def hmcode_pmm_baryonic_smart(
    input_params: Optional[Union[Array, dict]] = None,
    z_fine: Optional[Union[float, Array]] = None,
    N_coarse: int = 50,
    T_AGN: float = 10.0**7.8,
    *,
    linear_pmm_emu: TransferFunctionEmulator,
    linear_pcb_emu: TransferFunctionEmulator,
    nM: int = 128,
    k_out: Optional[Array] = None,
    **kwargs,
) -> tuple[Array, Array]:
    """
    Evaluate baryonic HMCode2020 non-linear matter power spectrum using an ergonomically
    constructed smart coarse redshift grid that explicitly places a node at the baryonic feature redshift.

    Parameters:
    -----------
    input_params: Array or dict, optional
        Cosmological parameters.
    z_fine: float or Array, optional
        Target fine redshift grid (or single redshift) for output.
    N_coarse: int, optional
        Total number of coarse grid points (default: 50).
    T_AGN: float, optional
        Baryon feedback temperature in Kelvin (default: 10^7.8 K).
    """
    if z_fine is None:
        raise ValueError("Missing required parameter 'z_fine'.")

    z_arr = jnp.atleast_1d(z_fine)
    is_scalar = jnp.ndim(z_fine) == 0 or z_arr.shape[0] == 1

    if is_scalar or z_arr.shape[0] <= N_coarse or z_arr.shape[0] < 5:
        return hmcode_pmm_from_emulator(
            input_params=input_params,
            z=z_fine,
            linear_pmm_emu=linear_pmm_emu,
            linear_pcb_emu=linear_pcb_emu,
            T_AGN=T_AGN,
            nM=nM,
            k_out=k_out,
            **kwargs,
        )

    z_min, z_max = jnp.min(z_arr), jnp.max(z_arr)

    # Predict feature point
    z_feature = predict_baryonic_discontinuity(
        input_params=input_params, T_AGN=T_AGN, **kwargs
    )

    z_feature = _clip_baryonic_feature(z_min, z_max, z_feature, 1.0e-4)
    n_left = _baryonic_left_nodes(N_coarse)
    z_coarse = build_smart_coarse_grid(
        z_min,
        z_max,
        N_coarse,
        z_feature=z_feature,
        N_left=n_left,
    )

    # Run coarse evaluation + Akima interpolation onto z_fine
    return hmcode_pmm_from_emulator_fast(
        input_params=input_params,
        z_coarse=z_coarse,
        z_fine=z_fine,
        linear_pmm_emu=linear_pmm_emu,
        linear_pcb_emu=linear_pcb_emu,
        T_AGN=T_AGN,
        nM=nM,
        k_out=k_out,
        piecewise_z_feature=z_feature,
        piecewise_split_index=n_left,
        **kwargs,
    )


def hmcode_pmm_dmo_smart(
    input_params: Optional[Union[Array, dict]] = None,
    z_fine: Optional[Union[float, Array]] = None,
    N_coarse: int = 50,
    *,
    linear_pmm_emu: TransferFunctionEmulator,
    linear_pcb_emu: TransferFunctionEmulator,
    nM: int = 128,
    k_out: Optional[Array] = None,
    **kwargs,
) -> tuple[Array, Array]:
    """
    Evaluate dark matter only HMCode2020 non-linear matter power spectrum using an ergonomically
    constructed smart coarse redshift grid.

    Parameters:
    -----------
    input_params: Array or dict, optional
        Cosmological parameters.
    z_fine: float or Array, optional
        Target fine redshift grid (or single redshift) for output.
    N_coarse: int, optional
        Total number of coarse grid points (default: 50).
    """
    if z_fine is None:
        raise ValueError("Missing required parameter 'z_fine'.")

    z_arr = jnp.atleast_1d(z_fine)
    is_scalar = jnp.ndim(z_fine) == 0 or z_arr.shape[0] == 1

    if is_scalar or z_arr.shape[0] <= N_coarse or z_arr.shape[0] < 5:
        return hmcode_pmm_from_emulator(
            input_params=input_params,
            z=z_fine,
            linear_pmm_emu=linear_pmm_emu,
            linear_pcb_emu=linear_pcb_emu,
            T_AGN=None,
            nM=nM,
            k_out=k_out,
            **kwargs,
        )

    z_min, z_max = jnp.min(z_arr), jnp.max(z_arr)

    # Build smart coarse grid (linear for DMO)
    z_coarse = jnp.linspace(z_min, z_max, N_coarse)

    # Run coarse evaluation + Akima interpolation onto z_fine
    return hmcode_pmm_from_emulator_fast(
        input_params=input_params,
        z_coarse=z_coarse,
        z_fine=z_fine,
        linear_pmm_emu=linear_pmm_emu,
        linear_pcb_emu=linear_pcb_emu,
        T_AGN=None,
        nM=nM,
        k_out=k_out,
        piecewise_z_feature=None,
        **kwargs,
    )
