#!/usr/bin/env python3
"""Benchmark jaxmapse linear + nonlinear predictions against CAMB and CLASS.

The benchmark samples cosmologies from the HMCode-trained jaxmapse emulator prior
and compares, for each cosmology:

1. Linear Pmm/Pcb:
   - jaxmapse emulator vs CAMB
   - jaxmapse emulator vs CLASS
   - CLASS vs CAMB

2. Nonlinear HMCode2020 feedback:
   - JAX HMCode with jaxmapse-emulated linear Pmm/Pcb
   - JAX HMCode with CLASS linear Pmm/Pcb sampled on the same support grid
   - CAMB native HMCode2020 feedback
   - CLASS native HMCode2020 feedback

3. Nonlinear Halofit:
   - JAX Halofit with jaxmapse-emulated linear Pmm
   - JAX Halofit with CLASS linear Pmm sampled on the same support grid
   - CAMB native Takahashi/Bird Halofit
   - CLASS native Halofit

All spectra are compared in h-units: k [h/Mpc], P [(Mpc/h)^3]. The HMCode
artifact linear spectra are in CLASS physical units, so the script converts

    k_h = k_artifact / h,    P_h = P_artifact * h**3.

Examples
--------
python scripts/benchmark_jaxmapse_boltzmann.py --n-cosmologies 20
python scripts/benchmark_jaxmapse_boltzmann.py --n-cosmologies 20 --tail none
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import camb
import jax
import jax.numpy as jnp
import numpy as np
from camb import model
from classy import Class

import jaxmapse
from jaxmapse import HMCodeCosmology, hmcode_pmm_jax, w0waCDMCosmology
from jaxmapse.halofit import (
    halofit_background,
    halofit_cosmology,
    halofit_pmm,
)

jax.config.update("jax_enable_x64", True)

HM_EMULATOR = "trained_mapse_class_hmcode_mnuw0waOkcdm"
MNU_CONV = 93.14
N_UR_CLASS = 2.0308  # CLASS recommendation for one massive nu with T_ncdm=0.71611 and Neff≈3.044.
T_NCDM_CLASS = 0.71611  # Gives Mnu/omega_nu≈93.14 eV for active massive neutrinos.
N_UR_CAMB = None  # Let CAMB use its default nnu=3.044 convention for num_massive_neutrinos=1.


@contextlib.contextmanager
def silence_fds(enabled: bool = True):
    """Silence verbose Fortran/C output from CLASS/CAMB when desired."""
    if not enabled:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(devnull)


@dataclass(frozen=True)
class CosmoPoint:
    idx: int
    ln10As: float
    ns: float
    H0: float
    ombh2: float
    omch2: float
    mnu: float
    w0: float
    wa: float
    omega_k: float
    log10T: float

    @property
    def h(self) -> float:
        return self.H0 / 100.0

    @property
    def As(self) -> float:
        return math.exp(self.ln10As) * 1.0e-10

    @property
    def omega_b(self) -> float:
        return self.ombh2 / self.h**2

    @property
    def omega_nu(self) -> float:
        return self.mnu / (MNU_CONV * self.h**2)

    @property
    def omega_m(self) -> float:
        return self.omega_b + self.omch2 / self.h**2 + self.omega_nu

    @property
    def omega_de(self) -> float:
        return 1.0 - self.omega_m - self.omega_k

    @property
    def p10(self) -> jnp.ndarray:
        return jnp.array(
            [
                self.ln10As,
                self.ns,
                self.H0,
                self.ombh2,
                self.omch2,
                self.mnu,
                self.w0,
                self.wa,
                self.omega_k,
                self.log10T,
            ]
        )

    @property
    def p8(self) -> jnp.ndarray:
        return jnp.array(
            [
                self.ln10As,
                self.ns,
                self.H0,
                self.ombh2,
                self.omch2,
                self.mnu,
                self.w0,
                self.wa,
            ]
        )


def sample_cosmologies(emu, n: int, seed: int, shrink: float, flat_only: bool) -> list[CosmoPoint]:
    """Sample deterministic points from the HMCode boost emulator prior.

    The boost in_minmax rows are:
    z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa, Omega_k, log10T_AGN.
    """
    bounds = np.asarray(emu.boost.in_minmax, dtype=float)
    param_bounds = bounds[1:, :].copy()
    lo, hi = param_bounds[:, 0], param_bounds[:, 1]
    lo = lo + shrink * (hi - lo)
    hi = hi - shrink * (hi - lo)

    rng = np.random.default_rng(seed)
    points: list[CosmoPoint] = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        attempts += 1
        u = rng.random(10)
        vals = lo + u * (hi - lo)
        if flat_only:
            vals[8] = 0.0  # Omega_k
        cp = CosmoPoint(len(points), *map(float, vals))
        # Avoid pathological points where dark energy density is non-positive today.
        if cp.omega_de <= 0.02:
            continue
        points.append(cp)
    if len(points) != n:
        raise RuntimeError(f"Only generated {len(points)} valid cosmologies after {attempts} attempts")
    return points


def interp_log_rows(values_zk: np.ndarray, k_src: np.ndarray, k_tgt: np.ndarray) -> np.ndarray:
    return np.vstack(
        [np.exp(np.interp(np.log(k_tgt), np.log(k_src), np.log(row))) for row in values_zk]
    )


def extend_powerlaw(k: np.ndarray, pk_zk: np.ndarray, kmax: float, n_extra: int, nfit: int = 12):
    if k[-1] >= kmax or n_extra <= 0:
        return k, pk_zk
    step = k[-1] / k[-2]
    k_extra = np.geomspace(k[-1] * step, kmax, n_extra)
    logk = np.log(k)
    logp = np.log(pk_zk)
    x = logk[-nfit:]
    y = logp[:, -nfit:]
    xm = x.mean()
    ym = y.mean(axis=1, keepdims=True)
    slope = np.sum((x - xm)[None, :] * (y - ym), axis=1) / np.sum((x - xm) ** 2)
    slope = np.clip(slope, -6.0, -1.0)
    tail = pk_zk[:, -1, None] * (k_extra[None, :] / k[-1]) ** slope[:, None]
    return np.concatenate([k, k_extra]), np.concatenate([pk_zk, tail], axis=1)


def maybe_extend(k: np.ndarray, pk_zk: np.ndarray, tail: str, kmax: float, n_extra: int):
    if tail == "none":
        return k, pk_zk
    if tail == "powerlaw":
        return extend_powerlaw(k, pk_zk, kmax, n_extra)
    raise ValueError(f"Unknown tail method {tail}")


def relerr_stats(x: np.ndarray, ref: np.ndarray, k: np.ndarray, band: tuple[float, float]):
    mask = (k >= band[0]) & (k <= band[1])
    err = np.abs(x[:, mask] - ref[:, mask]) / np.abs(ref[:, mask])
    return {
        "max": float(np.max(err)),
        "mean": float(np.mean(err)),
        "p95": float(np.quantile(err, 0.95)),
    }


def class_params(cp: CosmoPoint, nonlinear: str, accurate_ncdm: bool = False) -> dict:
    params = {
        "output": "mPk",
        "non linear": nonlinear,
        "P_k_max_1/Mpc": 1000.0,
        "z_max_pk": 5.0,
        "A_s": cp.As,
        "n_s": cp.ns,
        "H0": cp.H0,
        "omega_b": cp.ombh2,
        "omega_cdm": cp.omch2,
        "N_ur": N_UR_CLASS,
        "N_ncdm": 1,
        "m_ncdm": cp.mnu,
        "T_ncdm": T_NCDM_CLASS,
        "Omega_fld": cp.omega_de,
        "w0_fld": cp.w0,
        "wa_fld": cp.wa,
        "Omega_k": cp.omega_k,
    }
    if accurate_ncdm:
        # Disable the massive-neutrino fluid approximation for better CAMB/CLASS
        # linear-P(k) matching over the emulator prior. This is much slower but
        # reduced the transition-band mean disagreement by ~2-3x in diagnostics.
        params.update({"ncdm_fluid_approximation": 3, "l_max_ncdm": 40})
    if nonlinear == "hmcode":
        params.update(
            {
                "hmcode_version": "2020_baryonic_feedback",
                "log10T_heat_hmcode": cp.log10T,
            }
        )
    return params


def compute_class(cp: CosmoPoint, z: np.ndarray, k_support: np.ndarray, k_out: np.ndarray, quiet: bool, accurate_ncdm: bool = False):
    """Return CLASS linear support grids and native HMCode/Halofit on k_out."""
    with silence_fds(quiet):
        cc_hm = Class()
        cc_hm.set(class_params(cp, "hmcode", accurate_ncdm=accurate_ncdm))
        cc_hm.compute()

    def grid(cc: Class, k_h: np.ndarray, nonlinear=False, cb=False):
        out = np.empty((len(z), len(k_h)))
        with silence_fds(quiet):
            for iz, zz in enumerate(z):
                for ik, kh in enumerate(k_h):
                    km = kh * cp.h
                    if cb:
                        val = cc.pk_cb_lin(float(km), float(zz))
                    elif nonlinear:
                        val = cc.pk(float(km), float(zz))
                    else:
                        val = cc.pk_lin(float(km), float(zz))
                    out[iz, ik] = val * cp.h**3
        return out

    pmm = grid(cc_hm, k_support)
    pcb = grid(cc_hm, k_support, cb=True)
    hm = grid(cc_hm, k_out, nonlinear=True)
    with silence_fds(quiet):
        cc_hm.struct_cleanup()
        cc_hm.empty()

    with silence_fds(quiet):
        cc_hf = Class()
        cc_hf.set(class_params(cp, "halofit", accurate_ncdm=accurate_ncdm))
        cc_hf.compute()
    hf = grid(cc_hf, k_out, nonlinear=True)
    with silence_fds(quiet):
        cc_hf.struct_cleanup()
        cc_hf.empty()
    return pmm, pcb, hm, hf


def camb_params(cp: CosmoPoint, nonlinear_model: str):
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=cp.H0,
        ombh2=cp.ombh2,
        omch2=cp.omch2,
        mnu=cp.mnu,
        omk=cp.omega_k,
        num_massive_neutrinos=1,
    )
    pars.InitPower.set_params(As=cp.As, ns=cp.ns)
    pars.set_dark_energy(w=cp.w0, wa=cp.wa, dark_energy_model="ppf")
    pars.set_matter_power(redshifts=list(z_global), kmax=1000.0)
    pars.NonLinear = model.NonLinear_both
    if nonlinear_model == "hmcode":
        pars.NonLinearModel.set_params(
            halofit_version="mead2020_feedback",
            HMCode_logT_AGN=cp.log10T,
        )
    elif nonlinear_model == "halofit":
        pars.NonLinearModel.set_params(halofit_version="takahashi")
    else:
        raise ValueError(nonlinear_model)
    return pars


# Set by main before CAMB helper calls. This avoids passing z through CAMBparams builder repeatedly.
z_global: np.ndarray


def compute_camb(cp: CosmoPoint, z: np.ndarray, k_support: np.ndarray, k_out: np.ndarray, quiet: bool):
    global z_global
    z_global = z
    with silence_fds(quiet):
        pars_hm = camb_params(cp, "hmcode")
        camb.get_results(pars_hm)
        lin_mm = camb.get_matter_power_interpolator(
            pars_hm,
            nonlinear=False,
            hubble_units=True,
            k_hunit=True,
            kmax=1000.0,
            zmax=float(np.max(z)),
            var1="delta_tot",
            var2="delta_tot",
        )
        lin_cb = camb.get_matter_power_interpolator(
            pars_hm,
            nonlinear=False,
            hubble_units=True,
            k_hunit=True,
            kmax=1000.0,
            zmax=float(np.max(z)),
            var1="delta_nonu",
            var2="delta_nonu",
        )
        nl_hm = camb.get_matter_power_interpolator(
            pars_hm,
            nonlinear=True,
            hubble_units=True,
            k_hunit=True,
            kmax=1000.0,
            zmax=float(np.max(z)),
            var1="delta_tot",
            var2="delta_tot",
        )

    pmm = np.vstack([lin_mm.P(float(zz), k_support) for zz in z])
    pcb = np.vstack([lin_cb.P(float(zz), k_support) for zz in z])
    hm = np.vstack([nl_hm.P(float(zz), k_out) for zz in z])

    with silence_fds(quiet):
        pars_hf = camb_params(cp, "halofit")
        camb.get_results(pars_hf)
        nl_hf = camb.get_matter_power_interpolator(
            pars_hf,
            nonlinear=True,
            hubble_units=True,
            k_hunit=True,
            kmax=1000.0,
            zmax=float(np.max(z)),
            var1="delta_tot",
            var2="delta_tot",
        )
    hf = np.vstack([nl_hf.P(float(zz), k_out) for zz in z])
    return pmm, pcb, hm, hf


def hmcode_cosmology(cp: CosmoPoint) -> HMCodeCosmology:
    return HMCodeCosmology(
        cp.omega_m,
        cp.omega_b,
        cp.h,
        cp.ns,
        0.8,  # only retained for compatibility; HMCode infers amplitudes from P(k).
        cp.w0,
        cp.wa,
        cp.omega_nu,
        cp.omega_k,
    )


def jax_hmcode(cp: CosmoPoint, z, k_out, k_support, pmm_support, pcb_support):
    return np.asarray(
        jax.block_until_ready(
            hmcode_pmm_jax(
                hmcode_cosmology(cp),
                jnp.asarray(z),
                jnp.asarray(k_out),
                jnp.asarray(k_support),
                jnp.asarray(pmm_support),
                jnp.asarray(pcb_support),
                nM=96,
                include_feedback=True,
            )
        )
    )


def jax_halofit(cp: CosmoPoint, z, k_support, pmm_support, k_out):
    cpar = halofit_cosmology(cp.p8)
    omega_m_z, omega_v_z = halofit_background(cpar, jnp.asarray(z))
    pk_support = np.asarray(
        jax.block_until_ready(
            halofit_pmm(
                cpar,
                jnp.asarray(z),
                jnp.asarray(k_support),
                jnp.asarray(pmm_support),
                omega_m_z,
                omega_v_z,
            )
        )
    )
    return interp_log_rows(pk_support, k_support, k_out)


def add_metrics(rows: list[dict], cosmo_idx: int, category: str, comparison: str, x, ref, k, bands):
    for band_name, band in bands.items():
        st = relerr_stats(x, ref, k, band)
        rows.append(
            {
                "cosmo_idx": cosmo_idx,
                "category": category,
                "comparison": comparison,
                "band": band_name,
                **st,
            }
        )


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["category"], row["comparison"], row["band"])
        groups.setdefault(key, []).append(row)
    out = []
    for (category, comparison, band), vals in sorted(groups.items()):
        for metric in ["mean", "p95", "max"]:
            arr = np.array([v[metric] for v in vals], dtype=float)
            out.append(
                {
                    "category": category,
                    "comparison": comparison,
                    "band": band,
                    "metric": metric,
                    "n": len(arr),
                    "median_over_cosmologies": float(np.median(arr)),
                    "mean_over_cosmologies": float(np.mean(arr)),
                    "p95_over_cosmologies": float(np.quantile(arr, 0.95)),
                    "max_over_cosmologies": float(np.max(arr)),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cosmologies", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prior-shrink", type=float, default=0.05)
    ap.add_argument("--z-count", type=int, default=30)
    ap.add_argument("--k-count", type=int, default=128)
    ap.add_argument("--z-max", type=float, default=3.5)
    ap.add_argument("--k-min", type=float, default=1.0e-4)
    ap.add_argument("--k-max", type=float, default=10.0)
    ap.add_argument("--tail", choices=["none", "powerlaw"], default="powerlaw")
    ap.add_argument("--tail-kmax", type=float, default=1000.0)
    ap.add_argument("--tail-n-extra", type=int, default=64)
    ap.add_argument("--output-dir", type=Path, default=Path("benchmark_results/jaxmapse_boltzmann"))
    ap.add_argument("--quiet-solvers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--class-accurate-ncdm", action="store_true", help="Use CLASS full massive-neutrino hierarchy (ncdm_fluid_approximation=3, l_max_ncdm=40). Slower but improves linear CAMB/CLASS agreement.")
    args = ap.parse_args()

    emu = jaxmapse.trained_emulators[HM_EMULATOR]
    if emu is None:
        raise RuntimeError(f"{HM_EMULATOR} failed to load")

    z = np.linspace(0.0, args.z_max, args.z_count)
    k_out = np.geomspace(args.k_min, args.k_max, args.k_count)
    bands = {
        "transition_0p1_1": (0.1, 1.0),
        "wide_1em3_10": (1.0e-3, 10.0),
    }

    cosmologies = sample_cosmologies(
        emu, args.n_cosmologies, args.seed, args.prior_shrink, flat_only=True
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    failure_rows: list[dict] = []

    t_all = time.perf_counter()
    for cp in cosmologies:
        t0 = time.perf_counter()
        print(
            f"[{cp.idx+1:02d}/{len(cosmologies):02d}] "
            f"H0={cp.H0:.2f} ombh2={cp.ombh2:.4f} omch2={cp.omch2:.4f} "
            f"mnu={cp.mnu:.3f} w0={cp.w0:.2f} wa={cp.wa:.2f} logT={cp.log10T:.2f}",
            flush=True,
        )
        try:
            # Emulator linear spectra in physical CLASS units -> h-units.
            growth_cosmo = w0waCDMCosmology(
                ln10As=cp.ln10As,
                ns=cp.ns,
                h=cp.h,
                omega_b=cp.ombh2,
                omega_c=cp.omch2,
                m_nu=cp.mnu,
                w0=cp.w0,
                wa=cp.wa,
                omega_k=0.0,
            )
            D = growth_cosmo.D_z(jnp.asarray(z))
            pmm_emu_phys = np.asarray(
                jax.block_until_ready(emu.get_linear_pmm(cp.p10, jnp.asarray(z), D))
            )
            pcb_emu_phys = np.asarray(
                jax.block_until_ready(emu.get_linear_pkcb(cp.p10, jnp.asarray(z), D))
            )
            k_emu_h_all = np.asarray(emu.linear_pmm.k_grid) / cp.h
            support_mask = k_emu_h_all >= args.k_min
            k_native = k_emu_h_all[support_mask]
            pmm_emu_native = pmm_emu_phys[:, support_mask] * cp.h**3
            pcb_emu_native = pcb_emu_phys[:, support_mask] * cp.h**3

            # Nonlinear support is either native emulator support or extrapolated.
            k_support, pmm_emu_support = maybe_extend(
                k_native, pmm_emu_native, args.tail, args.tail_kmax, args.tail_n_extra
            )
            _, pcb_emu_support = maybe_extend(
                k_native, pcb_emu_native, args.tail, args.tail_kmax, args.tail_n_extra
            )

            # CLASS linear quantities are needed on the nonlinear support grid.
            # Since the power-law tail appends points after k_native, the first
            # len(k_native) columns are exactly the native emulator grid used for
            # linear comparisons.
            pmm_class_support, pcb_class_support, class_hm, class_hf = compute_class(
                cp, z, k_support, k_out, args.quiet_solvers, accurate_ncdm=args.class_accurate_ncdm
            )
            pmm_class_native = pmm_class_support[:, : len(k_native)]
            pcb_class_native = pcb_class_support[:, : len(k_native)]

            # CAMB linear quantities are only needed on the native grid for the
            # linear comparison; CAMB native nonlinear references are on k_out.
            pmm_camb_native, pcb_camb_native, camb_hm, camb_hf = compute_camb(
                cp, z, k_native, k_out, args.quiet_solvers
            )

            # JAX nonlinear kernels.
            jax_hm_emu = jax_hmcode(cp, z, k_out, k_support, pmm_emu_support, pcb_emu_support)
            jax_hm_class = jax_hmcode(cp, z, k_out, k_support, pmm_class_support, pcb_class_support)
            jax_hf_emu = jax_halofit(cp, z, k_support, pmm_emu_support, k_out)
            jax_hf_class = jax_halofit(cp, z, k_support, pmm_class_support, k_out)

            # Linear comparisons on native emulator grid.
            add_metrics(metric_rows, cp.idx, "linear_pmm", "jaxmapse_emu_vs_camb", pmm_emu_native, pmm_camb_native, k_native, bands)
            add_metrics(metric_rows, cp.idx, "linear_pmm", "jaxmapse_emu_vs_class", pmm_emu_native, pmm_class_native, k_native, bands)
            add_metrics(metric_rows, cp.idx, "linear_pmm", "class_vs_camb", pmm_class_native, pmm_camb_native, k_native, bands)
            add_metrics(metric_rows, cp.idx, "linear_pcb", "jaxmapse_emu_vs_camb", pcb_emu_native, pcb_camb_native, k_native, bands)
            add_metrics(metric_rows, cp.idx, "linear_pcb", "jaxmapse_emu_vs_class", pcb_emu_native, pcb_class_native, k_native, bands)
            add_metrics(metric_rows, cp.idx, "linear_pcb", "class_vs_camb", pcb_class_native, pcb_camb_native, k_native, bands)

            # HMCode comparisons on k_out.
            for ref_name, ref in [("camb_native", camb_hm), ("class_native", class_hm)]:
                add_metrics(metric_rows, cp.idx, "hmcode", f"jaxmapse_linear_vs_{ref_name}", jax_hm_emu, ref, k_out, bands)
                add_metrics(metric_rows, cp.idx, "hmcode", f"class_linear_vs_{ref_name}", jax_hm_class, ref, k_out, bands)
            add_metrics(metric_rows, cp.idx, "hmcode", "camb_native_vs_class_native", camb_hm, class_hm, k_out, bands)
            add_metrics(metric_rows, cp.idx, "hmcode", "jaxmapse_linear_vs_class_linear", jax_hm_emu, jax_hm_class, k_out, bands)

            # Halofit comparisons on k_out.
            for ref_name, ref in [("camb_native", camb_hf), ("class_native", class_hf)]:
                add_metrics(metric_rows, cp.idx, "halofit", f"jaxmapse_linear_vs_{ref_name}", jax_hf_emu, ref, k_out, bands)
                add_metrics(metric_rows, cp.idx, "halofit", f"class_linear_vs_{ref_name}", jax_hf_class, ref, k_out, bands)
            add_metrics(metric_rows, cp.idx, "halofit", "camb_native_vs_class_native", camb_hf, class_hf, k_out, bands)
            add_metrics(metric_rows, cp.idx, "halofit", "jaxmapse_linear_vs_class_linear", jax_hf_emu, jax_hf_class, k_out, bands)

            print(f"    ok in {time.perf_counter()-t0:.1f}s", flush=True)
        except Exception as exc:  # Keep going across prior points.
            failure_rows.append({"cosmo_idx": cp.idx, "error": repr(exc)})
            print(f"    FAILED: {exc!r}", flush=True)

    summary_rows = summarize(metric_rows)
    write_csv(args.output_dir / "per_cosmology_metrics.csv", metric_rows)
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows)
    write_csv(args.output_dir / "failures.csv", failure_rows)

    print(f"\nCompleted in {time.perf_counter()-t_all:.1f}s")
    print(f"Successful cosmology metrics: {len({r['cosmo_idx'] for r in metric_rows})}/{len(cosmologies)}")
    print(f"Failures: {len(failure_rows)}")
    print(f"Wrote: {args.output_dir / 'per_cosmology_metrics.csv'}")
    print(f"Wrote: {args.output_dir / 'summary_metrics.csv'}")

    # Compact report: mean-error summaries, p95 over cosmologies.
    print("\nCompact summary: metric=mean, value=p95_over_cosmologies")
    for row in summary_rows:
        if row["metric"] == "mean" and row["band"] == "transition_0p1_1":
            print(
                f"{row['category']:12s} {row['comparison']:36s} "
                f"median={row['median_over_cosmologies']:.3e} "
                f"p95={row['p95_over_cosmologies']:.3e} "
                f"max={row['max_over_cosmologies']:.3e}"
            )


if __name__ == "__main__":
    main()
