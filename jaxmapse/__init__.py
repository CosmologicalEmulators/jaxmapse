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
from .jaxmapse import (
    DEFAULT_EMULATOR_ARTIFACT,
    LinearPkEmulator,
    NonLinearBoostPkEmulator,
    PkEmulator,
    default_artifacts_toml,
    load_emulator,
    load_emulator_from_artifact,
    load_pk_emulator,
    load_pk_emulator_from_artifact,
)
from .primordial import primordial_Pk

__version__ = "0.1.0"

# Constants matching Mapse.jl
c_0 = 2.99792458e5  # Speed of light in km/s

__all__ = [
    "DEFAULT_EMULATOR_ARTIFACT",
    "LinearPkEmulator",
    "NonLinearBoostPkEmulator",
    "PkEmulator",
    "default_artifacts_toml",
    "HalofitCosmology",
    "load_emulator",
    "load_emulator_from_artifact",
    "load_pk_emulator",
    "load_pk_emulator_from_artifact",
    "halofit_background",
    "halofit_cosmology",
    "halofit_pmm",
    "halofit_Pmm",
    "halofit_pmm_from_params",
    "primordial_Pk",
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
    "trained_emulators",
]

# Artifact management and auto-loading
# Initialize the trained_emulators dictionary
# Format: { "emulator_name": PkEmulator_instance }
trained_emulators: Dict[str, Optional[PkEmulator]] = {}

# Path to the packaged Artifacts.toml registry.
_ARTIFACTS_TOML = default_artifacts_toml()

# Global artifact manager
_artifact_manager = None


def _get_artifact_manager():
    """Get or create the artifact manager singleton."""
    global _artifact_manager
    if _artifact_manager is None:
        if _ARTIFACTS_TOML.exists():
            _artifact_manager = load_artifacts(_ARTIFACTS_TOML)
        else:
            warnings.warn(f"Artifacts.toml not found at {_ARTIFACTS_TOML}")
    return _artifact_manager


# Load emulators on import (unless disabled)
if not os.environ.get("JAXMAPSE_NO_AUTO_DOWNLOAD"):
    manager = _get_artifact_manager()
    if manager is not None:
        for model_name in manager.artifacts:
            try:
                # Load the full PkEmulator from the artifact
                # Note: Artifacts.toml path is handled inside load_pk_emulator_from_artifact
                # via our default logic, but passing it explicitly is safer.
                trained_emulators[model_name] = load_pk_emulator_from_artifact(
                    model_name, artifacts_toml=str(_ARTIFACTS_TOML)
                )
            except Exception as e:
                warnings.warn(f"Failed to load {model_name} emulator: {e}")
                trained_emulators[model_name] = None
else:
    # Create empty structure when auto-download is disabled
    manager = _get_artifact_manager()
    if manager is not None:
        for model_name in manager.artifacts:
            trained_emulators[model_name] = None
