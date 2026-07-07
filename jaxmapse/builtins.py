"""Package-defined preprocessing and postprocessing functions for MAPSE artifacts.

Artifacts may name these functions in ``nn_setup.json`` with
``preprocessing_name`` and ``postprocessing_name``.  If no name is present,
``jaxmapse`` falls back to loading the legacy ``preprocessing.py`` and
``postprocessing.py`` files from the artifact directory.
"""

from __future__ import annotations

from typing import Callable, Dict

import jax.numpy as jnp
from jaxtyping import Array

from .primordial import primordial_Pk


def preprocessing_identity(params: Array) -> Array:
    """Return emulator inputs unchanged."""
    return params


def postprocessing_identity(input_params: Array, output: Array, D: Array, emu) -> Array:
    """Return raw emulator outputs unchanged."""
    return output


def preprocessing_linear_pk_mnuw0wacdm(params: Array) -> Array:
    """Preprocess ``mnuw0wacdm`` linear-P(k) parameters.

    Input order is ``[ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa]``.
    Linear MAPSE networks receive all parameters except ``ln10As`` and ``ns``.
    """
    return params[2:]


def postprocessing_linear_pk_mnuw0wacdm_sym_ratio(
    params: Array, output: Array, D: Array, emu
) -> Array:
    """Postprocess ``mnuw0wacdm`` sym-ratio linear-P(k) outputs.

    Applies the primordial spectrum, growth-factor scaling, and the analytic
    ``DIFF**2`` correction used by the matching MAPSE artifacts.
    """
    ln10As = params[0]
    ns = params[1]
    As = jnp.exp(ln10As) * 1e-10

    omega_b = params[3]
    omega_c = params[4]
    m_nu = params[5]

    k = emu.k_grid
    p_prim = primordial_Pk(As, ns, k)

    log10_k = jnp.log10(k)
    omega_nu = m_nu / 93.14
    delta_omega = omega_c + omega_nu - omega_b
    omega_m = omega_b + omega_c + omega_nu

    inner_cos_den = (1.1964213875807956**-2.3661897652294015) / jnp.cos(
        log10_k / -1.8173117588773222
    )
    inner_cos = jnp.cos(log10_k / inner_cos_den)

    term_1 = (
        ((0.731102574104348**log10_k) + delta_omega) / 0.17522861267519874
    ) ** log10_k
    term_2 = (63.65597287231169 ** (log10_k + 0.0472474783701488)) * (
        (0.9899093975978591 ** (log10_k / (inner_cos / 0.20037856443385513)))
        / (delta_omega**0.7767030041348179)
    )

    diff = jnp.exp(
        0.4971733969600907
        + (
            -24.849067935704547
            - jnp.log(term_1 + term_2 + (0.14823981687164764 * omega_m))
        )
    )

    return (output * diff**2) * (D**2) * p_prim


BUILTIN_PREPROCESSING: Dict[str, Callable] = {
    "identity": preprocessing_identity,
    "linear_pk_mnuw0wacdm": preprocessing_linear_pk_mnuw0wacdm,
}

BUILTIN_POSTPROCESSING: Dict[str, Callable] = {
    "identity": postprocessing_identity,
    "linear_pk_mnuw0wacdm_sym_ratio": postprocessing_linear_pk_mnuw0wacdm_sym_ratio,
}

LOAD_PRESETS = {
    "identity": {
        "preprocessing_name": "identity",
        "postprocessing_name": "identity",
    },
    "mnuw0wacdm_linear": {
        "preprocessing_name": "linear_pk_mnuw0wacdm",
        "postprocessing_name": "linear_pk_mnuw0wacdm_sym_ratio",
    },
}

