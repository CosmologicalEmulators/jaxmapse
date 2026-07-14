from jaxace.background import (
    D_f_z,
    D_z,
    E_a,
    E_z,
    Ωm_a,
    a_z,
    dA_z,
    dL_z,
    dlogEdloga,
    f_z,
    r_z,
    w0waCDMCosmology,
)

try:
    from jax.tree_util import register_pytree_node
    try:
        register_pytree_node(
            w0waCDMCosmology,
            lambda x: (
                (x.ln10As, x.ns, x.h, x.omega_b, x.omega_c, x.omega_k, x.m_nu, x.w0, x.wa),
                None
            ),
            lambda aux_data, children: w0waCDMCosmology(*children)
        )
    except ValueError:
        pass
except Exception:
    pass

from .halofit import (
    HalofitCosmology,
    halofit_background,
    halofit_cosmology,
    halofit_Pmm,
    halofit_pmm,
    halofit_pmm_from_params,
)
from .builtins import (
    BUILTIN_POSTPROCESSING,
    BUILTIN_PREPROCESSING,
    LOAD_PRESETS,
    lcdm_transfer_function,
    postprocessing_identity,
    postprocessing_lcdm_transfer_ratio,
    preprocessing_identity,
    preprocessing_drop_primordial_parameters,
)
from .hmcode import (
    HMCodeCosmology,
    hmcode_boost,
    hmcode_Pmm,
    hmcode_Pmm_jax,
    hmcode_pmm,
    hmcode_pmm_jax,
    hmcode_pmm_fast,
    hmcode_boost_fast,
)
from .jaxmapse import (
    DEFAULT_EMULATOR_ARTIFACT,
    TransferFunctionEmulator,
    default_artifacts_toml,
    artifact_path,
    load_emulator,
    load_trained_emulators,
    halofit_pmm_from_emulator,
    get_halofit_pmm,
    hmcode_pmm_from_emulator,
    get_hmcode_pmm,
    hmcode_pmm_from_emulator_fast,
    get_hmcode_pmm_fast,
)
from .primordial import primordial_Pk

__version__ = "0.1.1"

# Constants matching Mapse.jl
c_0 = 2.99792458e5  # Speed of light in km/s

__all__ = [
    "DEFAULT_EMULATOR_ARTIFACT",
    "TransferFunctionEmulator",
    "BUILTIN_POSTPROCESSING",
    "BUILTIN_PREPROCESSING",
    "LOAD_PRESETS",
    "lcdm_transfer_function",
    "default_artifacts_toml",
    "artifact_path",
    "HalofitCosmology",
    "load_emulator",
    "load_trained_emulators",
    "halofit_pmm_from_emulator",
    "get_halofit_pmm",
    "hmcode_pmm_from_emulator",
    "get_hmcode_pmm",
    "hmcode_pmm_from_emulator_fast",
    "get_hmcode_pmm_fast",
    "halofit_background",
    "halofit_cosmology",
    "halofit_pmm",
    "halofit_Pmm",
    "halofit_pmm_from_params",
    "HMCodeCosmology",
    "hmcode_pmm",
    "hmcode_pmm_jax",
    "hmcode_Pmm",
    "hmcode_Pmm_jax",
    "hmcode_boost",
    "hmcode_pmm_fast",
    "hmcode_boost_fast",
    "postprocessing_identity",
    "postprocessing_lcdm_transfer_ratio",
    "primordial_Pk",
    "preprocessing_identity",
    "preprocessing_drop_primordial_parameters",
    "c_0",
    "w0waCDMCosmology",
    "a_z",
    "E_a",
    "E_z",
    "dlogEdloga",
    "Ωm_a",
    "D_z",
    "f_z",
    "D_f_z",
    "r_z",
    "dA_z",
    "dL_z",
]
