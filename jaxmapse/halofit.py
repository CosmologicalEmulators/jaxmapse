"""JAX-native Takahashi/Bird Halofit utilities.

The public API follows the jaxmapse redshift convention: vector-redshift power
spectra have shape ``(len(z), len(k))``. Background quantities are explicit
inputs; compute them outside the Halofit kernel and pass ``omega_m_z`` and
``omega_v_z`` directly.
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jaxace import background as _background
from jaxtyping import Array
import numpy as np

_HALOFIT_OMEGA_GAMMA_H2 = 2.469e-5
_HALOFIT_MNU_TO_OMEGA_NU_H2 = 93.14
_HALOFIT_CLASS_N_UR = 2.033
_HALOFIT_MASSLESS_NU_FACTOR = 0.22710731766023898
_HALOFIT_JAXACE_N_EFF = 3.044
_HALOFIT_NEWTON_STEPS = 8


class HalofitCosmology(NamedTuple):
    """Background parameters required by the Halofit kernel.

    Densities are present-day density fractions. ``m_nu`` is the summed massive
    neutrino mass in eV. The fields are JAX pytrees, so this object can be used
    as a JIT argument.
    """

    omega_m0: Array
    omega_nu0: Array
    h: Array
    w0: Array
    wa: Array
    m_nu: Array
    omega_r0: Array
    omega_lambda0: Array


def halofit_cosmology(
    input_params: Array,
    *,
    omega_r0: Optional[float] = None,
    omega_lambda0: Optional[float] = None,
    n_ur: float = _HALOFIT_CLASS_N_UR,
) -> HalofitCosmology:
    """Build a :class:`HalofitCosmology` from MAPSE ``mnuw0wacdm`` parameters.

    The expected order is ``[ln10As, ns, H0, omega_b, omega_c, Mnu, w0, wa]``.
    ``H0`` may be supplied either as km/s/Mpc or as dimensionless ``h``.
    """

    params = jnp.asarray(input_params)
    if params.ndim != 1 or params.shape[0] != 8:
        raise ValueError(
            "halofit_cosmology expects flat mnuw0wacdm parameters in order "
            "[ln10As, ns, H0, omega_b, omega_c, Mnu, w0, wa]. "
            "Curved/HMCODE parameter vectors are not accepted by this flat "
            "Halofit helper."
        )
    h_raw = params[2]
    h = jnp.where(h_raw > 10.0, h_raw / 100.0, h_raw)
    omega_b = params[3]
    omega_c = params[4]
    m_nu = params[5]
    w0 = params[6]
    wa = params[7]

    omega_nu0 = (m_nu / _HALOFIT_MNU_TO_OMEGA_NU_H2) / h**2
    omega_m0 = (omega_b + omega_c) / h**2 + omega_nu0
    inferred_omega_r0 = (
        _HALOFIT_OMEGA_GAMMA_H2 / h**2 * (1.0 + _HALOFIT_MASSLESS_NU_FACTOR * n_ur)
    )
    omega_r0_value = inferred_omega_r0 if omega_r0 is None else jnp.asarray(omega_r0)
    omega_lambda0_value = (
        1.0 - omega_m0 - omega_r0_value
        if omega_lambda0 is None
        else jnp.asarray(omega_lambda0)
    )

    return HalofitCosmology(
        omega_m0=omega_m0,
        omega_nu0=omega_nu0,
        h=h,
        w0=w0,
        wa=wa,
        m_nu=m_nu,
        omega_r0=omega_r0_value,
        omega_lambda0=omega_lambda0_value,
    )


def halofit_background(cosmology: HalofitCosmology, z: Array) -> tuple[Array, Array]:
    """Compute native JAX background arrays for Halofit.

    Returns ``(omega_m_z, omega_v_z)`` for scalar or vector redshift ``z``.
    This helper uses the same jaxace background convention used elsewhere in
    jaxmapse. The matter term includes CDM+baryons plus the jaxace massive
    neutrino energy-density contribution. This is intentionally not forced to
    equal the simple non-relativistic density bookkeeping stored in
    :class:`HalofitCosmology` at exactly ``z=0``; explicit background arrays are
    the source of truth for the Halofit kernel.

    The calculation is intentionally separate from :func:`halofit_pmm`, so
    callers can still inject CLASS/CAMB/emulator background arrays when they
    want a controlled comparison.
    """

    z_arr = jnp.asarray(z)
    a = 1.0 / (1.0 + z_arr)
    omega_gamma0 = _HALOFIT_OMEGA_GAMMA_H2 / cosmology.h**2
    omega_cb0 = cosmology.omega_m0 - cosmology.omega_nu0

    e2 = (
        _background.E_a(
            a,
            omega_cb0,
            cosmology.h,
            mν=cosmology.m_nu,
            w0=cosmology.w0,
            wa=cosmology.wa,
        )
        ** 2
    )
    omega_nu_e2 = _background.ΩνE2(
        a, omega_gamma0, cosmology.m_nu, _HALOFIT_JAXACE_N_EFF
    )
    omega_nu0_e2 = _background.ΩνE2(
        1.0, omega_gamma0, cosmology.m_nu, _HALOFIT_JAXACE_N_EFF
    )
    omega_lambda0 = 1.0 - omega_gamma0 - omega_cb0 - omega_nu0_e2
    rho_de = _background.rhoDE_a(a, cosmology.w0, cosmology.wa)

    omega_m_z = (omega_cb0 * a**-3 + omega_nu_e2) / e2
    omega_v_z = omega_lambda0 * rho_de / e2
    return omega_m_z, omega_v_z


def _halofit_integrate_columns(logk: Array, y_kz: Array) -> Array:
    """Integrate each redshift column over log(k) with Cartwright Simpson rule."""

    n = logk.shape[0]
    last_simpson = n if n % 2 == 1 else n - 1

    x0 = logk[0 : last_simpson - 2 : 2, None]
    x1 = logk[1 : last_simpson - 1 : 2, None]
    x2 = logk[2:last_simpson:2, None]
    y0 = y_kz[0 : last_simpson - 2 : 2, :]
    y1 = y_kz[1 : last_simpson - 1 : 2, :]
    y2 = y_kz[2:last_simpson:2, :]

    h0 = x1 - x0
    h1 = x2 - x1
    simpson_terms = (
        (h0 + h1)
        / 6.0
        * (
            (2.0 - h1 / h0) * y0
            + ((h0 + h1) ** 2 / (h0 * h1)) * y1
            + (2.0 - h0 / h1) * y2
        )
    )
    total = jnp.sum(simpson_terms, axis=0, keepdims=True)

    if n % 2 == 0:
        total = total + (logk[-1] - logk[-2]) * (y_kz[-1:, :] + y_kz[-2:-1, :]) / 2.0

    return total


def _halofit_sigma2_derivs_columns(logk: Array, k: Array, pk_lin_kz: Array, r_z: Array):
    k_col = k[:, None]
    k_r2 = (k_col * r_z) ** 2
    exp_term = jnp.exp(-k_r2)
    integrand_pre = (k_col**3) * pk_lin_kz / (2.0 * jnp.pi**2)

    sig2 = _halofit_integrate_columns(logk, integrand_pre * exp_term)
    dsig2dr = (
        -2.0
        * r_z
        * _halofit_integrate_columns(logk, integrand_pre * (k_col**2) * exp_term)
    )
    is_sig_invalid = (sig2 <= 0.0) | ~jnp.isfinite(sig2)
    d1 = jnp.where(is_sig_invalid, jnp.nan, dsig2dr * r_z / sig2)

    d2sig2dr2 = _halofit_integrate_columns(
        logk, integrand_pre * (k_col**2) * exp_term * (-2.0 + 4.0 * k_r2)
    )
    d2 = jnp.where(is_sig_invalid, jnp.nan, (r_z**2 / sig2) * d2sig2dr2 + d1 - d1**2)

    return sig2, d1, d2


def _halofit_rnl_columns(logk: Array, k: Array, pk_lin_kz: Array) -> Array:
    lr = jnp.zeros((1, pk_lin_kz.shape[1]), dtype=pk_lin_kz.dtype)
    for _ in range(_HALOFIT_NEWTON_STEPS):
        sig2, d1, _ = _halofit_sigma2_derivs_columns(logk, k, pk_lin_kz, jnp.exp(lr))
        is_invalid = (sig2 <= 0.0) | ~jnp.isfinite(sig2) | (d1 == 0.0) | ~jnp.isfinite(d1)
        step_raw = jnp.where(is_invalid, jnp.nan, jnp.log(sig2) / d1)
        # Guard against a finite but tiny d1 producing an Inf or Nan step.
        step = jnp.where(~jnp.isfinite(step_raw), jnp.nan, step_raw)
        lr = lr - step
    return jnp.exp(lr)


def _halofit_power_columns(
    cosmology: HalofitCosmology,
    pk_lin_kz: Array,
    k: Array,
    z: Array,
    rnl: Array,
    neff: Array,
    cur: Array,
    omega_m_z: Array,
    omega_v_z: Array,
) -> Array:
    k_col = k[:, None]
    z_row = z[None, :]
    omega_m = omega_m_z[None, :]
    omega_v = omega_v_z[None, :]

    anorm = 1.0 / (2.0 * jnp.pi**2)
    opz = 1.0 + z_row
    wz = cosmology.w0 + cosmology.wa * (z_row / opz)
    f_nu = cosmology.omega_nu0 / cosmology.omega_m0
    y = k_col * rnl

    gam = 0.1971 - 0.0843 * neff + 0.8460 * cur
    an = 10.0 ** (
        1.5222
        + 2.8553 * neff
        + 2.3706 * neff**2
        + 0.9903 * neff**3
        + 0.2250 * neff**4
        - 0.6038 * cur
        + 0.1749 * omega_v * (1.0 + wz)
    )
    bn = 10.0 ** (
        -0.5642
        + 0.5864 * neff
        + 0.5716 * neff**2
        - 1.5474 * cur
        + 0.2279 * omega_v * (1.0 + wz)
    )
    cn = 10.0 ** (0.3698 + 2.0404 * neff + 0.8161 * neff**2 + 0.5869 * cur)
    x_mu = 0.0
    x_nu = 10.0 ** (5.2105 + 3.6902 * neff)
    alpha = jnp.abs(6.0835 + 1.3373 * neff - 0.1959 * neff**2 - 5.5274 * cur)
    beta = (
        2.0379
        - 0.7354 * neff
        + 0.3157 * neff**2
        + 1.2490 * neff**3
        + 0.3980 * neff**4
        - 0.1682 * cur
        + f_nu * (1.081 + 0.395 * neff**2)
    )

    use_de = jnp.abs(1.0 - omega_m) > 0.01
    denom = jnp.where(use_de, 1.0 - omega_m, 1.0)
    frac = omega_v / denom
    f1a = omega_m ** (-0.0732)
    f2a = omega_m ** (-0.1423)
    f3a = omega_m**0.0725
    f1b = omega_m ** (-0.0307)
    f2b = omega_m ** (-0.0585)
    f3b = omega_m**0.0743
    f1_de = frac * f1b + (1.0 - frac) * f1a
    f2_de = frac * f2b + (1.0 - frac) * f2a
    f3_de = frac * f3b + (1.0 - frac) * f3a
    f1 = jnp.where(use_de, f1_de, 1.0)
    f2 = jnp.where(use_de, f2_de, 1.0)
    f3 = jnp.where(use_de, f3_de, 1.0)

    pk_halo_raw = (
        an * y ** (f1 * 3.0) / (1.0 + bn * y**f2 + (f3 * cn * y) ** (3.0 - gam))
    )
    pk_halo_dim = pk_halo_raw / (1.0 + x_mu / y + x_nu / y**2)
    pk_halo_dim = pk_halo_dim * (1.0 + f_nu * 0.977)

    delta2_lin = (k_col**3) * pk_lin_kz * anorm
    kh = k_col / cosmology.h
    delta2_linaa = delta2_lin * (1.0 + f_nu * 47.48 * kh**2 / (1.0 + 1.5 * kh**2))
    pk_quasi_dim = (
        delta2_lin
        * (1.0 + delta2_linaa) ** beta
        / (1.0 + delta2_linaa * alpha)
        * jnp.exp(-y / 4.0 - y**2 / 8.0)
    )

    return (pk_halo_dim + pk_quasi_dim) / (k_col**3 * anorm)


def _halofit_pmm_kz(
    cosmology: HalofitCosmology,
    z: Array,
    k: Array,
    pk_lin_kz: Array,
    omega_m_z: Array,
    omega_v_z: Array,
) -> Array:
    logk = jnp.log(k)
    rnl = _halofit_rnl_columns(logk, k, pk_lin_kz)
    _, d1, d2 = _halofit_sigma2_derivs_columns(logk, k, pk_lin_kz, rnl)
    neff = -3.0 - d1
    cur = -d2
    return _halofit_power_columns(
        cosmology, pk_lin_kz, k, z, rnl, neff, cur, omega_m_z, omega_v_z
    )


def halofit_pmm(
    cosmology: HalofitCosmology,
    z: Array,
    k: Array,
    pk_lin_mm_z: Array,
    omega_m_z: Array,
    omega_v_z: Array,
) -> Array:
    """Compute nonlinear total-matter ``Pmm`` with JAX-native Halofit.

    Parameters
    ----------
    cosmology
        Halofit background parameter container.
    z
        Scalar or vector of redshifts.
    k
        Wavenumber grid in ``1/Mpc``.
    pk_lin_mm_z
        Linear total-matter power. For vector ``z`` this follows the jaxmapse
        convention ``(len(z), len(k))``. For scalar ``z`` this is ``(len(k),)``.
    omega_m_z, omega_v_z
        Matter and dark-energy density fractions evaluated at ``z``. These are
        explicit inputs so background calculations can live outside the JAX
        Halofit kernel.

    Returns
    -------
    Array
        Nonlinear total-matter power with the same redshift orientation as
        ``pk_lin_mm_z``: ``(len(z), len(k))`` for vector ``z`` or ``(len(k),)``
        for scalar ``z``.
    """

    z_input = jnp.asarray(z)
    z_arr = jnp.atleast_1d(z_input)
    k_arr = jnp.asarray(k)
    pk_arr = jnp.asarray(pk_lin_mm_z)
    omega_m_input = jnp.asarray(omega_m_z)
    omega_v_input = jnp.asarray(omega_v_z)
    omega_m_arr = jnp.atleast_1d(omega_m_input)
    omega_v_arr = jnp.atleast_1d(omega_v_input)

    if not isinstance(pk_arr, jax.core.Tracer):
        if np.any(np.isnan(pk_arr)) or np.any(np.isinf(pk_arr)):
            raise ValueError("pk_lin_mm_z must be finite.")
        if np.any(pk_arr <= 0.0):
            raise ValueError("pk_lin_mm_z must be strictly positive.")

    if k_arr.ndim != 1:
        raise ValueError("k must be a one-dimensional grid.")
    if z_input.ndim > 1:
        raise ValueError("z must be scalar or one-dimensional.")
    if omega_m_input.ndim != z_input.ndim or omega_v_input.ndim != z_input.ndim:
        raise ValueError(
            "omega_m_z and omega_v_z must have the same scalar/vector shape as z."
        )
    if omega_m_arr.shape[0] != z_arr.shape[0] or omega_v_arr.shape[0] != z_arr.shape[0]:
        raise ValueError("omega_m_z and omega_v_z lengths must match z.")

    if pk_arr.ndim == 1:
        if z_input.ndim != 0:
            raise ValueError("one-dimensional pk_lin_mm_z is only valid for scalar z.")
        if pk_arr.shape[0] != k_arr.shape[0]:
            raise ValueError("pk_lin_mm_z length must match k for scalar z.")
        pk_kz = pk_arr[:, None]
        out_kz = _halofit_pmm_kz(
            cosmology, z_arr, k_arr, pk_kz, omega_m_arr, omega_v_arr
        )
        return out_kz[:, 0]

    if pk_arr.ndim != 2:
        raise ValueError("pk_lin_mm_z must be one- or two-dimensional.")
    if z_input.ndim == 0:
        raise ValueError("two-dimensional pk_lin_mm_z requires vector z.")
    if pk_arr.shape != (z_arr.shape[0], k_arr.shape[0]):
        raise ValueError("pk_lin_mm_z must have shape (len(z), len(k)) for vector z.")

    pk_kz = jnp.swapaxes(pk_arr, 0, 1)
    out_kz = _halofit_pmm_kz(cosmology, z_arr, k_arr, pk_kz, omega_m_arr, omega_v_arr)
    return jnp.swapaxes(out_kz, 0, 1)


def halofit_pmm_from_params(
    input_params: Array,
    z: Array,
    k: Array,
    pk_lin_mm_z: Array,
    omega_m_z: Array,
    omega_v_z: Array,
    **cosmology_kwargs,
) -> Array:
    """Convenience wrapper constructing ``HalofitCosmology`` from MAPSE params."""

    cosmology = halofit_cosmology(input_params, **cosmology_kwargs)
    return halofit_pmm(cosmology, z, k, pk_lin_mm_z, omega_m_z, omega_v_z)


# Compatibility alias matching the Julia/Mapse.jl spelling.
halofit_Pmm = halofit_pmm
