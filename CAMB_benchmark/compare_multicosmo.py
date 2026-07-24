"""Compare direct and smart jaxmapse HMCode against ten saved CAMB fixtures."""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

import jaxmapse  # noqa: E402


FIXTURE_ROOT = Path(__file__).parent / "multicosmo"
N_COARSE = 24
N_MASS = 128


def error_statistics(prediction, reference):
    relative = np.abs(prediction - reference) / np.maximum(np.abs(reference), 1.0e-30)
    index = np.unravel_index(np.argmax(relative), relative.shape)
    return {
        "max": float(relative[index]),
        "mean": float(np.mean(relative)),
        "rms": float(np.sqrt(np.mean(relative**2))),
        "p99": float(np.quantile(relative, 0.99)),
        "fraction_gt_1pct": float(np.mean(relative > 0.01)),
        "fraction_gt_5pct": float(np.mean(relative > 0.05)),
        "fraction_gt_10pct": float(np.mean(relative > 0.10)),
        "index": [int(index[0]), int(index[1])],
    }


def evaluate_case(case, fixture, pmm, pcb, z_fine, k_h):
    params = jnp.asarray(case["params"])
    log_temperature = case["log10_T_AGN"]
    k_out = k_h * params[2] / 100.0
    reference_dmo = fixture[:, : k_h.shape[0]]
    reference_feedback = fixture[:, k_h.shape[0] :]

    _, direct_dmo = jaxmapse.hmcode_pmm_from_emulator(
        params,
        z_fine,
        linear_pmm_emu=pmm,
        linear_pcb_emu=pcb,
        T_AGN=None,
        nM=N_MASS,
        k_out=k_out,
    )
    _, direct_feedback = jaxmapse.hmcode_pmm_from_emulator(
        params,
        z_fine,
        linear_pmm_emu=pmm,
        linear_pcb_emu=pcb,
        T_AGN=10.0**log_temperature,
        nM=N_MASS,
        k_out=k_out,
    )
    _, smart_dmo = jaxmapse.hmcode_pmm_dmo_smart(
        params,
        z_fine=z_fine,
        N_coarse=N_COARSE,
        linear_pmm_emu=pmm,
        linear_pcb_emu=pcb,
        nM=N_MASS,
        k_out=k_out,
    )
    _, smart_feedback = jaxmapse.hmcode_pmm_baryonic_smart(
        params,
        z_fine=z_fine,
        N_coarse=N_COARSE,
        T_AGN=10.0**log_temperature,
        linear_pmm_emu=pmm,
        linear_pcb_emu=pcb,
        nM=N_MASS,
        k_out=k_out,
    )
    outputs = {
        "direct_dmo": np.asarray(direct_dmo),
        "smart_dmo": np.asarray(smart_dmo),
        "direct_feedback": np.asarray(direct_feedback),
        "smart_feedback": np.asarray(smart_feedback),
    }
    for name, output in outputs.items():
        if output.shape != reference_dmo.shape:
            raise ValueError(f"{case['name']} {name}: unexpected shape {output.shape}")
        if not np.all(np.isfinite(output)) or not np.all(output > 0.0):
            raise ValueError(f"{case['name']} {name}: non-finite or non-positive output")

    return {
        "name": case["name"],
        "params": case["params"],
        "log10_T_AGN": log_temperature,
        "w0_plus_wa": case["params"][6] + case["params"][7],
        "z_feature": float(
            jaxmapse.predict_baryonic_discontinuity(
                params, T_AGN=10.0**log_temperature
            )
        ),
        "direct_dmo_vs_camb": error_statistics(outputs["direct_dmo"], reference_dmo),
        "smart_dmo_vs_direct": error_statistics(
            outputs["smart_dmo"], outputs["direct_dmo"]
        ),
        "smart_dmo_vs_camb": error_statistics(outputs["smart_dmo"], reference_dmo),
        "direct_feedback_vs_camb": error_statistics(
            outputs["direct_feedback"], reference_feedback
        ),
        "smart_feedback_vs_direct": error_statistics(
            outputs["smart_feedback"], outputs["direct_feedback"]
        ),
        "smart_feedback_vs_camb": error_statistics(
            outputs["smart_feedback"], reference_feedback
        ),
    }


def print_results(results):
    columns = (
        "direct_dmo_vs_camb",
        "smart_dmo_vs_direct",
        "smart_dmo_vs_camb",
        "direct_feedback_vs_camb",
        "smart_feedback_vs_direct",
        "smart_feedback_vs_camb",
    )
    labels = ("D/C DMO", "S/D DMO", "S/C DMO", "D/C FB", "S/D FB", "S/C FB")
    print("\nMaximum relative errors [%]")
    print(f"{'case':30s} {'zfeat':>7s} " + " ".join(f"{label:>10s}" for label in labels))
    print("-" * 108)
    for result in results:
        values = " ".join(f"{100.0 * result[key]['max']:10.4f}" for key in columns)
        print(f"{result['name']:30s} {result['z_feature']:7.3f} {values}")

    print("\nWorst cases")
    for label, key in zip(labels, columns):
        worst = max(results, key=lambda result: result[key]["max"])
        stats = worst[key]
        print(
            f"{label:10s}: max={100.0 * stats['max']:.6f}% "
            f"mean={100.0 * stats['mean']:.6f}% "
            f"rms={100.0 * stats['rms']:.6f}% "
            f"p99={100.0 * stats['p99']:.6f}% "
            f">1%={100.0 * stats['fraction_gt_1pct']:.3f}% "
            f">5%={100.0 * stats['fraction_gt_5pct']:.3f}% "
            f"case={worst['name']} index={tuple(stats['index'])}"
        )


def main():
    jax.config.update("jax_enable_x64", True)
    manifest = json.loads((FIXTURE_ROOT / "cases.json").read_text())
    grid = manifest["grid"]
    z_fine = jnp.linspace(grid["z_min"], grid["z_max"], grid["n_z"])
    k_h = jnp.logspace(
        np.log10(grid["k_h_min"]),
        np.log10(grid["k_h_max"]),
        grid["n_k"],
    )
    artifact = jaxmapse.artifact_path(jaxmapse.DEFAULT_EMULATOR_ARTIFACT)
    pmm = jaxmapse.load_emulator(artifact / "Pk_lin_mm")
    pcb = jaxmapse.load_emulator(artifact / "Pk_lin_cb")

    results = []
    for index, case in enumerate(manifest["cases"]):
        if not case["params"][6] + case["params"][7] < 0.0:
            raise ValueError(f"{case['name']}: w0 + wa must be negative")
        fixture_path = FIXTURE_ROOT / f"case_{index:02d}_{case['name']}.txt"
        print(f"[{index + 1:02d}/{len(manifest['cases'])}] {case['name']}")
        result = evaluate_case(
            case,
            np.loadtxt(fixture_path),
            pmm,
            pcb,
            z_fine,
            k_h,
        )
        results.append(result)

    print_results(results)
    output = FIXTURE_ROOT / "comparison_results.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved detailed results to {output}")


if __name__ == "__main__":
    main()
