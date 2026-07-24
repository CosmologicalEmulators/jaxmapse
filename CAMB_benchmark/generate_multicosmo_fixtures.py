"""Generate the committed ten-cosmology CAMB HMCode reference fixtures."""

import json
from pathlib import Path

import camb
import numpy as np


ROOT = Path(__file__).parent / "multicosmo"
CASES_PATH = ROOT / "cases.json"


def camb_spectrum(params, log10_t_agn, z, k_phys, *, feedback):
    ln10_as, ns, h0, omega_b, omega_c, mnu, w0, wa = params
    if not w0 + wa < 0.0:
        raise ValueError(f"CAMB/emulator trust condition violated: w0 + wa = {w0 + wa}")

    camb_params = camb.CAMBparams()
    camb_params.set_cosmology(
        H0=h0,
        ombh2=omega_b,
        omch2=omega_c,
        mnu=mnu,
        omk=0.0,
    )
    camb_params.InitPower.set_params(As=np.exp(ln10_as) * 1.0e-10, ns=ns)
    camb_params.set_dark_energy(w=w0, wa=wa, dark_energy_model="ppf")
    camb_params.set_matter_power(redshifts=[0.0], kmax=50.0, nonlinear=True)
    if feedback:
        camb_params.NonLinearModel.set_params(
            halofit_version="mead2020_feedback",
            HMCode_logT_AGN=log10_t_agn,
        )
    else:
        camb_params.NonLinearModel.set_params(halofit_version="mead2020")

    interpolator = camb.get_matter_power_interpolator(
        camb_params,
        nonlinear=True,
        hubble_units=False,
        k_hunit=False,
        kmax=50.0,
        zmax=4.0,
        zs=z,
    )
    return interpolator.P(z, k_phys)


def main():
    manifest = json.loads(CASES_PATH.read_text())
    grid = manifest["grid"]
    z = np.linspace(grid["z_min"], grid["z_max"], grid["n_z"])
    k_h = np.logspace(
        np.log10(grid["k_h_min"]),
        np.log10(grid["k_h_max"]),
        grid["n_k"],
    )

    for index, case in enumerate(manifest["cases"]):
        params = np.asarray(case["params"], dtype=np.float64)
        w0, wa = params[6:8]
        if not w0 + wa < 0.0:
            raise ValueError(f"{case['name']}: w0 + wa must be negative")
        k_phys = k_h * params[2] / 100.0
        print(
            f"[{index + 1:02d}/10] {case['name']}: "
            f"w0={w0:.3f}, wa={wa:.3f}, w0+wa={w0 + wa:.3f}"
        )
        dmo = camb_spectrum(
            params,
            case["log10_T_AGN"],
            z,
            k_phys,
            feedback=False,
        )
        feedback = camb_spectrum(
            params,
            case["log10_T_AGN"],
            z,
            k_phys,
            feedback=True,
        )
        fixture = np.concatenate((dmo, feedback), axis=1)
        output = ROOT / f"case_{index:02d}_{case['name']}.txt"
        header = (
            f"CAMB {camb.__version__}; params={case['params']}; "
            f"log10_T_AGN={case['log10_T_AGN']}; "
            "columns 0:128 DMO, columns 128:256 feedback; units Mpc^3"
        )
        np.savetxt(output, fixture, header=header)
        print(f"         saved {output.name} {fixture.shape}")


if __name__ == "__main__":
    main()
