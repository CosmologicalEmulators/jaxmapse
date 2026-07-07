import os
import warnings
from typing import Dict, Optional

from fetch_artifacts import load_artifacts
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
    postprocessing_identity,
    postprocessing_linear_pk_mnuw0wacdm_sym_ratio,
    preprocessing_identity,
    preprocessing_linear_pk_mnuw0wacdm,
)
from .hmcode import (
    HMCodeCosmology,
    hmcode_boost,
    hmcode_Pmm,
    hmcode_Pmm_jax,
    hmcode_pmm,
    hmcode_pmm_jax,
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
    "default_artifacts_toml",
    "artifact_path",
    "HalofitCosmology",
    "load_emulator",
    "load_trained_emulators",
    "halofit_pmm_from_emulator",
    "get_halofit_pmm",
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
    "postprocessing_identity",
    "postprocessing_linear_pk_mnuw0wacdm_sym_ratio",
    "primordial_Pk",
    "preprocessing_identity",
    "preprocessing_linear_pk_mnuw0wacdm",
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

