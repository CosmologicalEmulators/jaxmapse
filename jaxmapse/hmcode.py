"""JAX-native HMCode2020 nonlinear matter power implementation.

This module mirrors the embedded HMCode2020 port in Mapse.jl. Public arrays use
jaxmapse's redshift-first convention: for vector redshifts, input spectra have
shape ``(len(z), len(k_support))`` and outputs have shape ``(len(z), len(k))``.

The public :func:`hmcode_pmm` function validates Python inputs, then dispatches to
:func:`hmcode_pmm_jax`, a pure-JAX kernel that can be used under ``jax.jit`` for
fixed input shapes and static ``nM``/feedback choices.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

jax.config.update("jax_enable_x64", True)

RHO_CRITICAL = 2.77536627245708e11  # Msun/h / (Mpc/h)^3
DV0 = 18.0 * jnp.pi**2
DC0 = (3.0 / 20.0) * (12.0 * jnp.pi) ** (2.0 / 3.0)
ND_HMCODE = 2.853
ST_A = 0.2161599867112559


class HMCodeCosmology(NamedTuple):
    """Cosmology container for the HMCode2020 implementation.

    Densities are present-day fractions. The object is a JAX pytree, so it can be
    passed to ``jax.jit``-compiled HMCode kernels.
    """

    Omega_m: float
    Omega_b: float
    h: float
    n_s: float
    sigma_8: float
    w0: float
    wa: float
    Omega_nu: float = 0.0
    Omega_k: float = 0.0


class HMCodeParams(NamedTuple):
    R_nl: Array
    n_eff: Array
    sigma_v: Array
    Delta_v: Array
    delta_c: Array
    eta: Array
    A: Array
    f_damp: Array
    k_star: Array
    B: Array
    k_damp: Array


def _as_1d(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or one-dimensional.")
    return arr


def _validate_inputs(z, k_out, k_support, pk_mm, pk_cb):
    if len(k_support) < 2 or len(k_out) < 2:
        raise ValueError("HMCode requires at least two k values.")
    if np.any(k_support <= 0.0) or np.any(k_out <= 0.0):
        raise ValueError("HMCode k grids must be strictly positive.")
    if np.any(np.diff(k_support) <= 0.0) or np.any(np.diff(k_out) <= 0.0):
        raise ValueError("HMCode k grids must be sorted in ascending order.")
    if k_out[0] < k_support[0] or k_out[-1] > k_support[-1]:
        raise ValueError("output k range must lie inside the support k range.")
    expected = (len(z), len(k_support))
    if pk_mm.shape != expected:
        raise ValueError(
            f"pk_mm_z must have shape (len(z), len(k_support))={expected}; "
            f"got {pk_mm.shape}."
        )
    if pk_cb is not None and pk_cb.shape != expected:
        raise ValueError(
            f"pk_cb_z must have shape (len(z), len(k_support))={expected}; "
            f"got {pk_cb.shape}."
        )
    if np.any(pk_mm <= 0.0) or (pk_cb is not None and np.any(pk_cb <= 0.0)):
        raise ValueError("HMCode linear spectra must be strictly positive.")


def _hmcode_mass_steps(nM, accuracy):
    if accuracy <= 0.0:
        raise ValueError("HMCode accuracy must be positive.")
    if nM is None:
        return max(16, int(np.ceil(256 * accuracy)))
    if accuracy != 1.0:
        raise ValueError("Pass either accuracy or nM, not both.")
    if nM < 2:
        raise ValueError("HMCode nM must be at least 2.")
    return int(nM)


def _trapz(y, x, axis=-1):
    dx = jnp.diff(x)
    return jnp.sum(0.5 * dx * (jnp.take(y, jnp.arange(1, y.shape[axis]), axis=axis) + jnp.take(y, jnp.arange(0, y.shape[axis] - 1), axis=axis)), axis=axis)


def _loglog_interp(logx, logy, x):
    return jnp.exp(jnp.interp(jnp.log(x), logx, logy))


def _interp_matrix_from_support_jax(k_support, pk_zk, k_out):
    logk = jnp.log(k_support)
    logpk = jnp.log(pk_zk)
    return jax.vmap(lambda row: _loglog_interp(logk, row, k_out))(logpk)


def _tophat_k(x):
    x = jnp.asarray(x)
    return jnp.where(
        jnp.abs(x) < 1.0e-5,
        1.0 - x**2 / 10.0,
        3.0 * (jnp.sin(x) - x * jnp.cos(x)) / x**3,
    )


def _comoving_matter_density(omega_m):
    return RHO_CRITICAL * omega_m


def _lagrangian_radius(mass, omega_m):
    return (3.0 * mass / (4.0 * jnp.pi * _comoving_matter_density(omega_m))) ** (
        1.0 / 3.0
    )


def _scale_factor(z):
    return 1.0 / (1.0 + z)


def _redshift(a):
    return -1.0 + 1.0 / a


def _w(a, cosmo: HMCodeCosmology, lcdm=False):
    w0 = -1.0 if lcdm else cosmo.w0
    wa = 0.0 if lcdm else cosmo.wa
    return w0 + (1.0 - a) * wa


def _x_w(a, cosmo: HMCodeCosmology, lcdm=False):
    w0 = -1.0 if lcdm else cosmo.w0
    wa = 0.0 if lcdm else cosmo.wa
    return a ** (-3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * (1.0 - a))


def _hubble2(a, cosmo: HMCodeCosmology, lcdm=False):
    om_w = 1.0 - cosmo.Omega_m if lcdm else 1.0 - cosmo.Omega_m - cosmo.Omega_k
    om = 1.0 if lcdm else 1.0 - cosmo.Omega_k
    return cosmo.Omega_m * a ** -3 + om_w * _x_w(a, cosmo, lcdm) + (1.0 - om) * a ** -2


def _omega_m_a(a, cosmo: HMCodeCosmology, lcdm=False):
    return cosmo.Omega_m * a ** -3 / _hubble2(a, cosmo, lcdm)


def _ah(a, cosmo: HMCodeCosmology, lcdm=False):
    om_w = 1.0 - cosmo.Omega_m if lcdm else 1.0 - cosmo.Omega_m - cosmo.Omega_k
    return -0.5 * (
        cosmo.Omega_m * a ** -3
        + (1.0 + 3.0 * _w(a, cosmo, lcdm)) * om_w * _x_w(a, cosmo, lcdm)
    )


def _growth_rhs(a, u, cosmo, lcdm):
    d_val, v_val = u
    fv = -(2.0 + _ah(a, cosmo, lcdm) / _hubble2(a, cosmo, lcdm)) * v_val / a
    fd = 1.5 * _omega_m_a(a, cosmo, lcdm) * d_val / a**2
    return jnp.array([v_val, fv + fd])


def _rk4_step(u, a_pair, cosmo, lcdm):
    a0, a1 = a_pair
    h = a1 - a0
    k1 = _growth_rhs(a0, u, cosmo, lcdm)
    k2 = _growth_rhs(a0 + 0.5 * h, u + 0.5 * h * k1, cosmo, lcdm)
    k3 = _growth_rhs(a0 + 0.5 * h, u + 0.5 * h * k2, cosmo, lcdm)
    k4 = _growth_rhs(a1, u + h * k3, cosmo, lcdm)
    return u + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _growth_tables_jax(cosmo: HMCodeCosmology, lcdm=False, na=2000):
    a_init = 1.0e-4
    a_grid = jnp.linspace(a_init, 1.0, na)
    f_init = 1.0 - _omega_m_a(a_init, cosmo, lcdm)
    d0 = a_init ** (1.0 - 3.0 * f_init / 5.0)
    v0 = (1.0 - 3.0 * f_init / 5.0) * a_init ** (-3.0 * f_init / 5.0)
    u0 = jnp.array([d0, v0])

    def scan_step(u, a_pair):
        u_next = _rk4_step(u, a_pair, cosmo, lcdm)
        return u_next, u_next

    _, states = jax.lax.scan(scan_step, u0, (a_grid[:-1], a_grid[1:]))
    growth = jnp.concatenate([u0[None, :], states], axis=0)[:, 0]
    integrand = growth / a_grid
    increments = 0.5 * jnp.diff(a_grid) * (integrand[1:] + integrand[:-1])
    agrowth = jnp.concatenate(
        [jnp.array([growth[0]]), growth[0] + jnp.cumsum(increments)]
    )
    return a_grid, growth, agrowth


def _f_mead(x, y, p0, p1, p2, p3):
    return p0 + p1 * (1.0 - x) + p2 * (1.0 - x) ** 2 + p3 * (1.0 - y)


def _dc_mead(a, om_m, f_nu, g, G):
    p10, p11, p12, p13 = -0.0069, -0.0208, 0.0312, 0.0021
    p20, p21, p22, p23 = 0.0001, -0.0647, -0.0417, 0.0646
    dc = 1.0
    dc += _f_mead(g / a, G / a, p10, p11, p12, p13) * jnp.log10(om_m)
    dc += _f_mead(g / a, G / a, p20, p21, p22, p23)
    return dc * DC0 * (1.0 - 0.041 * f_nu)


def _dv_mead(a, om_m, f_nu, g, G):
    p30, p31, p32, p33 = -0.79, -10.17, 2.51, 6.51
    p40, p41, p42, p43 = -1.89, 0.38, 18.8, -15.87
    logom = jnp.log10(om_m)
    dv = 1.0
    dv += _f_mead(g / a, G / a, p30, p31, p32, p33) * logom
    dv += _f_mead(g / a, G / a, p40, p41, p42, p43) * logom**2
    return dv * DV0 * (1.0 + 0.763 * f_nu)


def _derivative_from_samples_jax(x, xs, fs):
    ix = jnp.argmin(jnp.abs(xs - x))
    i0 = jnp.clip(ix - 1, 0, xs.shape[0] - 3)
    idx = i0 + jnp.arange(3)
    x0, x1, x2 = jnp.take(xs, idx)
    f0, f1, f2 = jnp.take(fs, idx)
    return (
        f0 * (2.0 * x - x1 - x2) / ((x0 - x1) * (x0 - x2))
        + f1 * (2.0 * x - x0 - x2) / ((x1 - x0) * (x1 - x2))
        + f2 * (2.0 * x - x0 - x1) / ((x2 - x0) * (x2 - x1))
    )


def _tk_eh_nowiggle(k, h, wm, wb, t_cmb=2.725):
    rb = wb / wm
    s = 44.5 * jnp.log(9.83 / wm) / jnp.sqrt(1.0 + 10.0 * wb**0.75)
    alpha = 1.0 - 0.328 * jnp.log(431.0 * wm) * rb + 0.38 * jnp.log(22.3 * wm) * rb**2
    gamma = (wm / h) * (alpha + (1.0 - alpha) / (1.0 + (0.43 * k * s * h) ** 4))
    q = k * (t_cmb / 2.7) ** 2 / gamma
    L = jnp.log(2.0 * jnp.e + 1.8 * q)
    C = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return L / (L + C * q**2)


def _reflect_indices(idx, n):
    period = 2 * n
    imod = jnp.mod(idx, period)
    return jnp.where(imod < n, imod, period - imod - 1)


def _gaussian_filter1d_reflect_jax(y, sigma, truncate=4.0):
    n = y.shape[0]
    offsets = jnp.arange(-(n - 1), n)
    mask = jnp.abs(offsets) <= truncate * sigma
    weights = jnp.exp(-0.5 * (offsets / sigma) ** 2) * mask
    weights = weights / jnp.sum(weights)
    base = jnp.arange(n)[:, None]
    idx = _reflect_indices(base + offsets[None, :], n).astype(jnp.int32)
    return jnp.sum(y[idx] * weights[None, :], axis=1)


def _pk_wiggle_jax(k, pk_lin, h, omega_m_h2, omega_b_h2, n_s):
    dlnk = jnp.log(k[1] / k[0])
    sigma = 0.25 / dlnk
    tk_nw = _tk_eh_nowiggle(k, h, omega_m_h2, omega_b_h2)
    pk_nw = k**n_s * tk_nw**2
    ratio = pk_lin / pk_nw
    smooth = _gaussian_filter1d_reflect_jax(ratio, sigma)
    return pk_lin - smooth * pk_nw


def _sigma_grid_jax(k_support, pk_cb_zk, r_grid):
    logk = jnp.log(k_support)
    W = _tophat_k(r_grid[:, None] * k_support[None, :])
    integrand = (
        k_support[None, None, :] ** 3
        * pk_cb_zk[:, None, :]
        * W[None, :, :] ** 2
        / (2.0 * jnp.pi**2)
    )
    sigma2 = _trapz(integrand, logk, axis=-1)
    return jnp.sqrt(jnp.maximum(sigma2, 0.0))


def _sigma_v_jax(k_support, pk_mm_zk):
    logk = jnp.log(k_support)
    integrand = pk_mm_zk * k_support[None, :]
    sigma2 = _trapz(integrand, logk, axis=-1) / (2.0 * jnp.pi**2)
    return jnp.sqrt(jnp.maximum(sigma2, 0.0)) / jnp.sqrt(3.0)


def _inverse_interp_decreasing(x_decreasing, y_increasing, x):
    return jnp.interp(x, x_decreasing[::-1], y_increasing[::-1])


def _compute_params_jax(z, a_grid, growth, agrowth, sigma_zm, r_grid, k_support, pk_mm_zk, cosmo):
    f_nu = cosmo.Omega_nu / cosmo.Omega_m
    a = _scale_factor(z)
    om_mz = _omega_m_a(a, cosmo, False)
    g = jnp.interp(a, a_grid, growth)
    G = jnp.interp(a, a_grid, agrowth)
    dc = _dc_mead(a, om_mz, f_nu, g, G)
    dv = _dv_mead(a, om_mz, f_nu, g, G)

    logR = jnp.log(r_grid)
    logsigma = jnp.log(sigma_zm)
    logdc = jnp.log(dc)
    rnl = jax.vmap(lambda ls, ldc: jnp.exp(_inverse_interp_decreasing(ls, logR, ldc)))(
        logsigma, logdc
    )
    s8 = jax.vmap(lambda ls: jnp.exp(jnp.interp(jnp.log(8.0), logR, ls)))(logsigma)
    sigma_v = _sigma_v_jax(k_support, pk_mm_zk)
    neff = -3.0 - 2.0 * jax.vmap(_derivative_from_samples_jax, in_axes=(0, None, 0))(
        jnp.log(rnl), logR, logsigma
    )

    return HMCodeParams(
        R_nl=rnl,
        n_eff=neff,
        sigma_v=sigma_v,
        Delta_v=dv,
        delta_c=dc,
        eta=0.1281 * s8 ** (-0.3644),
        A=1.875 * (1.603) ** neff,
        f_damp=0.2696 * s8**0.9403,
        k_star=0.05618 * s8 ** (-1.013),
        B=jnp.full_like(z, 5.196),
        k_damp=0.05699 * s8 ** (-1.089),
    )


def _params_notweaks(params: HMCodeParams):
    zeros = jnp.zeros_like(params.eta)
    ones = jnp.ones_like(params.eta)
    return HMCodeParams(
        R_nl=params.R_nl,
        n_eff=params.n_eff,
        sigma_v=params.sigma_v,
        Delta_v=params.Delta_v,
        delta_c=params.delta_c,
        eta=zeros,
        A=ones,
        f_damp=zeros,
        k_star=params.k_star,
        B=jnp.full_like(params.B, 4.0),
        k_damp=zeros,
    )


def _compute_weights_jax(M, nu_zm, amp):
    p_st, q_st = 0.3, 0.707
    g = ST_A * (1.0 + (q_st * nu_zm**2) ** (-p_st)) * jnp.exp(-q_st * nu_zm**2 / 2.0)
    wt0 = 0.5 * (nu_zm[:, 1:2] - nu_zm[:, 0:1])
    wtm = 0.5 * (nu_zm[:, 2:] - nu_zm[:, :-2])
    wt1 = 0.5 * (nu_zm[:, -1:] - nu_zm[:, -2:-1])
    wt = jnp.concatenate([wt0, wtm, wt1], axis=1)
    return (g / M[None, :]) * amp[None, :] ** 2 * wt


def _evalpoly(x, coeffs):
    y = jnp.asarray(coeffs[-1], dtype=x.dtype)
    for c in reversed(coeffs[:-1]):
        y = y * x + c
    return y


EULER_GAMMA = 0.5772156649015329
SI_SMALL = (
    1.0,
    -0.05555555555555555,
    0.0016666666666666668,
    -2.834467120181406e-5,
    3.0619243582206544e-7,
    -2.27746439867652e-9,
    1.2353110643708935e-11,
    -5.0981091545465446e-14,
    1.6537983849091297e-16,
    -4.326650129802279e-19,
    9.32044812542441e-22,
    -1.6818131176655147e-24,
    2.5787801137537893e-27,
    -3.401366616220572e-30,
    3.8999872022233505e-33,
    -3.922984005011348e-36,
    3.489798672961197e-39,
    -2.7650265596091116e-42,
    1.9636378862575867e-45,
    -1.257043527311165e-48,
    7.291002017420499e-52,
    -3.849327599400454e-55,
    1.8577001882628452e-58,
    -8.22686917863956e-62,
    3.3550504251358757e-65,
    -1.264109733422975e-68,
)
CI_INT_SMALL = (
    -0.25,
    0.010416666666666666,
    -0.0002314814814814815,
    3.1001984126984127e-6,
    -2.755731922398589e-8,
    1.7397297489890083e-10,
    -8.193389712664089e-13,
    2.9871733327421158e-15,
    -8.677337204770125e-18,
    2.0551588116560825e-20,
    -4.043996087477533e-23,
    6.715573212900493e-26,
    -9.53690870471076e-29,
    1.1713890132392279e-31,
    -1.2566625429386353e-34,
    1.1876221108921073e-37,
    -9.962228045650476e-41,
    7.467278517462879e-44,
    -5.031482118527058e-47,
    3.0640435978209645e-50,
    -1.6946206503074855e-53,
    8.549642911891504e-57,
    -3.9506856555684327e-60,
    1.6782241814065079e-63,
    -6.575898833266316e-67,
)
P_F_RAT1 = (0.9999999996217391, 364.510603386319, 44218.54804128844, 2246756.9405961153, 49315316.72305596, 431867952.7967028, 1184799251.9992545, 455732675.9379532)
Q_F_RAT1 = (1.0, 366.5106027322935, 44927.56981497069, 2328535.488220404, 53117852.01722826, 503353106.6724187, 1657528501.5623176, 1174653283.7041042)
R_G_RAT1 = (0.999999999204849, 513.8550487530732, 92293.48345259381, 7407134.186234174, 281423561.62841356, 4928089035.773462, 35524762685.554024, 79194271662.05495, 17942522624.4139)
S_G_RAT1 = (1.0, 519.8550470881487, 95292.61550812594, 7921545.967976676, 319775677.90347815, 6227313470.243901, 54570971054.996445, 182417501666.45703, 154071481488.65445)
P_F_ASYM_NUM = (1.9999999999999978, 2220.611938043496, 847490.0762398824, 139592679.54823944, 10197205463.267975, 302298652645.2408, 2750405380428.847, 2181898970468.7498)
P_F_ASYM_DEN = (1.0, 1122.3059690217168, 436852.7097485132, 74654702.14065616, 5858003475.188747, 201579803792.09885, 2622914185768.9644, 8785290733498.676)
R_G_ASYM_NUM = (5.999999999999999, 9652.774604499714, 5607762.699656884, 1502266771.8927317, 196442710647.33087, 121913682811632.5, 3192438989864569.5, 2.5876053010027484e16, 1.2754978896268878e16)
R_G_ASYM_DEN = (1.0, 1628.7957674166142, 966363.0319578709, 268397347.5095067, 37388510548.052925, 2602858566615.2144, 85134283716949.72, 1130407936162795.2, 4251984147948980.0)


def _si_fast(x):
    small = x * _evalpoly(x * x, SI_SMALL)
    t = x * x
    invt = 1.0 / t
    sx, cx = jnp.sin(x), jnp.cos(x)
    f_rat = (_evalpoly(invt, P_F_RAT1) / _evalpoly(invt, Q_F_RAT1)) / x
    g_rat = (_evalpoly(invt, R_G_RAT1) / _evalpoly(invt, S_G_RAT1)) * invt
    f_asym = (1.0 - _evalpoly(invt, P_F_ASYM_NUM) * invt / _evalpoly(invt, P_F_ASYM_DEN)) / x
    g_asym = (1.0 - _evalpoly(invt, R_G_ASYM_NUM) * invt / _evalpoly(invt, R_G_ASYM_DEN)) * invt
    f = jnp.where(t <= 144.0, f_rat, f_asym)
    g = jnp.where(t <= 144.0, g_rat, g_asym)
    large = jnp.pi / 2.0 - f * cx - g * sx
    return jnp.where(x <= 4.0, small, large)


def _ci_fast(x):
    small = EULER_GAMMA + jnp.log(x) + x * x * _evalpoly(x * x, CI_INT_SMALL)
    t = x * x
    invt = 1.0 / t
    sx, cx = jnp.sin(x), jnp.cos(x)
    f_rat = (_evalpoly(invt, P_F_RAT1) / _evalpoly(invt, Q_F_RAT1)) / x
    g_rat = (_evalpoly(invt, R_G_RAT1) / _evalpoly(invt, S_G_RAT1)) * invt
    f_asym = (1.0 - _evalpoly(invt, P_F_ASYM_NUM) * invt / _evalpoly(invt, P_F_ASYM_DEN)) / x
    g_asym = (1.0 - _evalpoly(invt, R_G_ASYM_NUM) * invt / _evalpoly(invt, R_G_ASYM_DEN)) * invt
    f = jnp.where(t <= 144.0, f_rat, f_asym)
    g = jnp.where(t <= 144.0, g_rat, g_asym)
    large = f * sx - g * cx
    return jnp.where(x <= 4.0, small, large)


def _wnfw_fast_jax(x, c, ln1pc):
    x_plus = x * (1.0 + 1.0 / c)
    x_minus = x / c
    dsi = _si_fast(x_plus) - _si_fast(x_minus)
    dci = _ci_fast(x_plus) - _ci_fast(x_minus)
    sinc_xp = jnp.sin(x) / x_plus
    norm = ln1pc - c / (1.0 + c)
    return (dsi * jnp.sin(x_minus) + dci * jnp.cos(x_minus) - sinc_xp) / norm


def _feedback_parameters_jax(t_agn):
    theta = jnp.log10(t_agn / 10.0**7.8)
    return (
        3.44 - 0.496 * theta,
        -0.0671 - 0.0371 * theta,
        10.0 ** (13.87 + 1.81 * theta),
        -0.108 + 0.195 * theta,
        (2.01 - 0.3 * theta) * 1.0e-2,
        0.409 + 0.0224 * theta,
    )


def _assemble_pass_jax(k, z, cosmo, M, R, params, sigma_zm, nu_zm, pk_lin_zk, pk_wig_zk, a_grid, growth, growth_lcdm, tweaks=True, include_feedback=False, t_agn=10.0**7.8):
    rhom = _comoving_matter_density(cosmo.Omega_m)
    om_m, om_b, om_nu = cosmo.Omega_m, cosmo.Omega_b, cosmo.Omega_nu
    om_c = om_m - om_b - om_nu
    f_nu = om_nu / om_m
    feedback_profile = include_feedback and not tweaks
    amp_factor = 1.0 if feedback_profile else 1.0 - f_nu
    amp = M * amp_factor / rhom
    w1h_zm = _compute_weights_jax(M, nu_zm, amp)
    fb_B0, fb_Bz, fb_Mb0, fb_Mbz, fb_f0, fb_fz = _feedback_parameters_jax(t_agn)
    ac = _scale_factor(10.0)
    g_ac = jnp.interp(ac, a_grid, growth)
    g_lcdm_ac = jnp.interp(ac, a_grid, growth_lcdm)
    logR = jnp.log(R)

    def one_z(zz, sigma_m, nu_m, w1h_m, pk_lin, pk_wig, dc, dv, eta, alpha, kdamp, fdamp, kstar, sigv, B_base):
        a_obs = _scale_factor(zz)
        g_obs = jnp.interp(a_obs, a_grid, growth)
        dolag = (g_ac / g_lcdm_ac) * (jnp.interp(a_obs, a_grid, growth_lcdm) / g_obs)
        rc = _lagrangian_radius(0.01 * M, om_m)
        sig_rc = jnp.exp(jnp.interp(jnp.log(rc), logR, jnp.log(sigma_m)))
        g_target = g_obs * dc / sig_rc
        af = jnp.interp(g_target, growth, a_grid)
        zf = jnp.where(g_target >= g_obs, zz, _redshift(af))
        rv = R / dv ** (1.0 / 3.0)
        B = fb_B0 * 10.0 ** (zz * fb_Bz) if feedback_profile else B_base
        conc = B * (1.0 + zf) / (1.0 + zz) * dolag
        ln1pc = jnp.log1p(conc)
        rv_eff = rv * nu_m**eta

        W = jax.vmap(lambda rve, cc, ll: _wnfw_fast_jax(k * rve, cc, ll))(
            rv_eff, conc, ln1pc
        )
        if feedback_profile:
            Mb = fb_Mb0 * 10.0 ** (zz * fb_Mbz)
            fstar = jnp.minimum(fb_f0 * 10.0 ** (zz * fb_fz), om_b / om_m)
            fg = (om_b / om_m - fstar) * (M / Mb) ** 2 / (1.0 + (M / Mb) ** 2)
            W = (om_c / om_m + fg)[:, None] * W + fstar

        I1h = (W**2).T @ w1h_m * rhom
        x4 = (k / kstar) ** 4
        p1h = x4 / (1.0 + x4) * I1h
        if tweaks:
            pk_dwl = pk_lin - (1.0 - jnp.exp(-(k * sigv) ** 2)) * pk_wig
            y = (k / kdamp) ** ND_HMCODE
            p2h = pk_dwl * (1.0 - fdamp * y / (1.0 + y))
            return (p2h**alpha + p1h**alpha) ** (1.0 / alpha)
        return pk_lin + p1h

    return jax.vmap(one_z)(
        z,
        sigma_zm,
        nu_zm,
        w1h_zm,
        pk_lin_zk,
        pk_wig_zk,
        params.delta_c,
        params.Delta_v,
        params.eta,
        params.A,
        params.k_damp,
        params.f_damp,
        params.k_star,
        params.sigma_v,
        params.B,
    )


@partial(jax.jit, static_argnames=("nM", "include_feedback"))
def hmcode_pmm_jax(
    cosmo: HMCodeCosmology,
    z: Array,
    k: Array,
    k_support: Array,
    pk_mm_z: Array,
    pk_cb_z: Array,
    *,
    T_AGN: float = 10.0**7.8,
    Mmin: float = 1.0,
    Mmax: float = 1.0e18,
    nM: int = 256,
    include_feedback: bool = True,
) -> Array:
    """Pure-JAX HMCode2020 kernel.

    This function assumes validated vector-redshift inputs with spectra shaped
    ``(len(z), len(k_support))``. ``nM`` and ``include_feedback`` are static JIT
    arguments.
    """
    z = jnp.asarray(z)
    k = jnp.asarray(k)
    k_support = jnp.asarray(k_support)
    pk_mm_z = jnp.asarray(pk_mm_z)
    pk_cb_z = jnp.asarray(pk_cb_z)

    M = jnp.exp(jnp.linspace(jnp.log(Mmin), jnp.log(Mmax), nM))
    R = _lagrangian_radius(M, cosmo.Omega_m)
    pk_mm_out = _interp_matrix_from_support_jax(k_support, pk_mm_z, k)
    sigma_zm = _sigma_grid_jax(k_support, pk_cb_z, R)
    a_grid, growth, agrowth = _growth_tables_jax(cosmo, lcdm=False)
    _, growth_lcdm, _ = _growth_tables_jax(cosmo, lcdm=True)
    params = _compute_params_jax(z, a_grid, growth, agrowth, sigma_zm, R, k_support, pk_mm_z, cosmo)
    nu_zm = params.delta_c[:, None] / sigma_zm

    omh2 = cosmo.Omega_m * cosmo.h**2
    obh2 = cosmo.Omega_b * cosmo.h**2
    pk_wig = jax.vmap(lambda row: _pk_wiggle_jax(k, row, cosmo.h, omh2, obh2, cosmo.n_s))(
        pk_mm_out
    )
    base = _assemble_pass_jax(
        k,
        z,
        cosmo,
        M,
        R,
        params,
        sigma_zm,
        nu_zm,
        pk_mm_out,
        pk_wig,
        a_grid,
        growth,
        growth_lcdm,
        tweaks=True,
        include_feedback=False,
        t_agn=T_AGN,
    )
    if include_feedback:
        params_not = _params_notweaks(params)
        den = _assemble_pass_jax(
            k,
            z,
            cosmo,
            M,
            R,
            params_not,
            sigma_zm,
            nu_zm,
            pk_mm_out,
            pk_wig,
            a_grid,
            growth,
            growth_lcdm,
            tweaks=False,
            include_feedback=False,
            t_agn=T_AGN,
        )
        num = _assemble_pass_jax(
            k,
            z,
            cosmo,
            M,
            R,
            params_not,
            sigma_zm,
            nu_zm,
            pk_mm_out,
            pk_wig,
            a_grid,
            growth,
            growth_lcdm,
            tweaks=False,
            include_feedback=True,
            t_agn=T_AGN,
        )
        base = base * (num / den)
    return base


def hmcode_pmm(
    cosmo: HMCodeCosmology,
    z: Array,
    k: Array,
    pk_mm_z: Array,
    pk_cb_z: Optional[Array] = None,
    *,
    k_support: Optional[Array] = None,
    pk_cb_support_z: Optional[Array] = None,
    T_AGN: Optional[float] = 10.0**7.8,
    Mmin: float = 1.0,
    Mmax: float = 1.0e18,
    nM: Optional[int] = None,
    accuracy: float = 1.0,
) -> Array:
    """Compute HMCode2020 nonlinear total-matter ``Pmm``.

    For vector redshifts, spectra are shaped ``(len(z), len(k_support))`` and
    output is ``(len(z), len(k))``. If ``k_support`` is omitted, ``k`` is used as
    both support and output grid. ``pk_cb_z``/``pk_cb_support_z`` supplies the
    cold+baryon linear spectrum used for HMCode internal σ(R) quantities.
    """
    scalar_z = np.asarray(z).ndim == 0
    z_arr = _as_1d(z, "z")
    k_out = np.asarray(k, dtype=float)
    k_sup = k_out if k_support is None else np.asarray(k_support, dtype=float)
    pk_mm = np.asarray(pk_mm_z, dtype=float)
    if pk_mm.ndim == 1:
        pk_mm = pk_mm[None, :]
    if pk_cb_support_z is not None and pk_cb_z is not None:
        raise ValueError("Pass either pk_cb_z or pk_cb_support_z, not both.")
    pk_cb = pk_cb_support_z if pk_cb_support_z is not None else pk_cb_z
    pk_cb = pk_mm if pk_cb is None else np.asarray(pk_cb, dtype=float)
    if pk_cb.ndim == 1:
        pk_cb = pk_cb[None, :]
    _validate_inputs(z_arr, k_out, k_sup, pk_mm, pk_cb)
    nM_eff = _hmcode_mass_steps(nM, accuracy)
    out = hmcode_pmm_jax(
        cosmo,
        jnp.asarray(z_arr),
        jnp.asarray(k_out),
        jnp.asarray(k_sup),
        jnp.asarray(pk_mm),
        jnp.asarray(pk_cb),
        T_AGN=10.0**7.8 if T_AGN is None else float(T_AGN),
        Mmin=float(Mmin),
        Mmax=float(Mmax),
        nM=nM_eff,
        include_feedback=T_AGN is not None,
    )
    return out[0] if scalar_z else out


def hmcode_boost(
    cosmo: HMCodeCosmology,
    z: Array,
    k: Array,
    pk_mm_z: Array,
    pk_cb_z: Optional[Array] = None,
    **kwargs,
) -> Array:
    """Return HMCode nonlinear boost ``Pmm_nl / Pmm_lin``."""
    scalar_z = np.asarray(z).ndim == 0
    k_support = kwargs.get("k_support")
    k_out = np.asarray(k, dtype=float)
    k_sup = k_out if k_support is None else np.asarray(k_support, dtype=float)
    pk_mm = np.asarray(pk_mm_z, dtype=float)
    if pk_mm.ndim == 1:
        pk_mm = pk_mm[None, :]
    pk_nl = hmcode_pmm(cosmo, z, k, pk_mm_z, pk_cb_z=pk_cb_z, **kwargs)
    pk_lin_out = _interp_matrix_from_support_jax(
        jnp.asarray(k_sup), jnp.asarray(pk_mm), jnp.asarray(k_out)
    )
    boost = pk_nl / (pk_lin_out[0] if scalar_z else pk_lin_out)
    return boost


hmcode_Pmm = hmcode_pmm
hmcode_Pmm_jax = hmcode_pmm_jax
