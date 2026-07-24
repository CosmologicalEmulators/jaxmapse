import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import jaxmapse
from CAMB_benchmark.compare_multicosmo import evaluate_case


CAMB_MAX_RELATIVE_ERROR = 0.011
SMART_MAX_RELATIVE_ERROR = 0.005


def test_multicosmology_hmcode_accuracy_against_camb():
    fixture_root = Path(__file__).parents[1] / "CAMB_benchmark" / "multicosmo"
    manifest = json.loads((fixture_root / "cases.json").read_text())
    grid = manifest["grid"]
    assert len(manifest["cases"]) == 10

    z_fine = jnp.linspace(grid["z_min"], grid["z_max"], grid["n_z"])
    k_h = jnp.logspace(
        np.log10(grid["k_h_min"]),
        np.log10(grid["k_h_max"]),
        grid["n_k"],
    )
    artifact = jaxmapse.artifact_path(jaxmapse.DEFAULT_EMULATOR_ARTIFACT)
    pmm = jaxmapse.load_emulator(artifact / "Pk_lin_mm")
    pcb = jaxmapse.load_emulator(artifact / "Pk_lin_cb")

    for index, case in enumerate(manifest["cases"]):
        assert case["params"][6] + case["params"][7] < 0.0
        fixture = np.loadtxt(
            fixture_root / f"case_{index:02d}_{case['name']}.txt"
        )
        assert fixture.shape == (grid["n_z"], 2 * grid["n_k"])
        assert np.all(np.isfinite(fixture))
        assert np.all(fixture > 0.0)

        result = evaluate_case(case, fixture, pmm, pcb, z_fine, k_h)
        for key, statistics in result.items():
            if key.endswith(("_vs_camb", "_vs_direct")):
                for metric in (
                    "max",
                    "mean",
                    "rms",
                    "p99",
                    "fraction_gt_1pct",
                    "fraction_gt_5pct",
                    "fraction_gt_10pct",
                ):
                    assert np.isfinite(statistics[metric])

                ceiling = (
                    SMART_MAX_RELATIVE_ERROR
                    if key.endswith("_vs_direct")
                    else CAMB_MAX_RELATIVE_ERROR
                )
                assert statistics["max"] < ceiling, (
                    f"{case['name']} {key} maximum relative error "
                    f"{statistics['max']:.6%} exceeds {ceiling:.3%}"
                )
