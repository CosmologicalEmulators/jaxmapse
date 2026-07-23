from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmapse import (
    DEFAULT_EMULATOR_ARTIFACT,
    artifact_path,
    hmcode_pmm_baryonic_smart,
    hmcode_pmm_dmo_smart,
    hmcode_pmm_from_emulator,
    hmcode_pmm_from_emulator_fast,
    load_emulator,
)


PARAMS = jnp.array([3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0])
Z_FINE = jnp.linspace(0.0, 3.5, 150)
N_COARSE_VALUES = (24, 32, 40)
CAMB_MAX_RELATIVE_ERROR = 0.0051


@pytest.fixture(scope="module")
def linear_emulators():
    root = artifact_path(DEFAULT_EMULATOR_ARTIFACT)
    return load_emulator(root / "Pk_lin_mm"), load_emulator(root / "Pk_lin_cb")


@pytest.fixture(scope="module")
def camb_references():
    root = Path(__file__).parents[1] / "CAMB_benchmark"
    return (
        np.loadtxt(root / "camb_pk_hmcode_dmo.txt"),
        np.loadtxt(root / "camb_pk_hmcode_fb.txt"),
    )


@pytest.mark.parametrize("n_coarse", N_COARSE_VALUES)
@pytest.mark.parametrize("mode", ("dmo", "feedback"))
def test_smart_hmcode_matches_camb(
    linear_emulators, camb_references, mode, n_coarse
):
    pmm, pcb = linear_emulators
    reference = camb_references[0 if mode == "dmo" else 1]
    h = PARAMS[2] / 100.0
    k_out = 10.0 ** jnp.linspace(-3.0, 1.0, 128) * h

    if mode == "dmo":
        _, prediction = hmcode_pmm_dmo_smart(
            PARAMS,
            z_fine=Z_FINE,
            N_coarse=n_coarse,
            linear_pmm_emu=pmm,
            linear_pcb_emu=pcb,
            nM=128,
            k_out=k_out,
        )
    else:
        _, prediction = hmcode_pmm_baryonic_smart(
            PARAMS,
            z_fine=Z_FINE,
            N_coarse=n_coarse,
            T_AGN=10.0**7.8,
            linear_pmm_emu=pmm,
            linear_pcb_emu=pcb,
            nM=128,
            k_out=k_out,
        )

    relative_error = np.abs(np.asarray(prediction) - reference) / np.maximum(
        np.abs(reference), 1.0e-30
    )
    max_error = np.max(relative_error)
    assert max_error < CAMB_MAX_RELATIVE_ERROR, (
        f"{mode} N_coarse={n_coarse} max CAMB error {max_error:.6%} exceeds "
        f"{CAMB_MAX_RELATIVE_ERROR:.3%}"
    )


def _pipeline(path, params, pmm, pcb):
    if path == "direct":
        return hmcode_pmm_from_emulator(
            params,
            jnp.linspace(0.0, 1.0, 3),
            linear_pmm_emu=pmm,
            linear_pcb_emu=pcb,
            T_AGN=10.0**7.8,
            nM=16,
        )[1]
    if path == "fixed_fast":
        return hmcode_pmm_from_emulator_fast(
            params,
            z_coarse=jnp.linspace(0.0, 1.0, 5),
            z_fine=jnp.linspace(0.0, 1.0, 12),
            linear_pmm_emu=pmm,
            linear_pcb_emu=pcb,
            T_AGN=None,
            nM=16,
        )[1]
    if path == "dmo_smart":
        return hmcode_pmm_dmo_smart(
            params,
            z_fine=jnp.linspace(0.0, 1.0, 12),
            N_coarse=5,
            linear_pmm_emu=pmm,
            linear_pcb_emu=pcb,
            nM=16,
        )[1]
    raise AssertionError(f"Unknown pipeline: {path}")


@pytest.mark.parametrize("path", ("direct", "fixed_fast", "dmo_smart"))
def test_end_to_end_pipeline_compiles_once_and_remains_dynamic(linear_emulators, path):
    pmm, pcb = linear_emulators
    trace_count = 0

    def pipeline(params):
        nonlocal trace_count
        trace_count += 1
        return _pipeline(path, params, pmm, pcb)

    compiled = jax.jit(pipeline).lower(PARAMS).compile()
    prediction1 = compiled(PARAMS)
    prediction2 = compiled(PARAMS.at[0].add(0.01))
    prediction1.block_until_ready()
    prediction2.block_until_ready()

    assert trace_count == 1
    assert prediction1.shape == prediction2.shape
    assert jnp.all(jnp.isfinite(prediction1))
    assert jnp.all(jnp.isfinite(prediction2))
    assert not jnp.allclose(prediction1, prediction2)


@pytest.mark.parametrize("path", ("direct", "fixed_fast", "dmo_smart"))
def test_end_to_end_reverse_gradients_match_finite_differences(
    linear_emulators, path
):
    pmm, pcb = linear_emulators

    def loss(params):
        prediction = _pipeline(path, params, pmm, pcb)
        return jnp.mean(jnp.log(prediction))

    compiled_loss = jax.jit(loss)
    reverse_gradient = jax.jit(jax.grad(loss))(PARAMS)
    reverse_gradient.block_until_ready()

    steps = 1.0e-5 * jnp.maximum(jnp.abs(PARAMS), 1.0)
    finite_difference_gradient = []
    for index in range(PARAMS.shape[0]):
        upper = compiled_loss(PARAMS.at[index].add(steps[index]))
        lower = compiled_loss(PARAMS.at[index].add(-steps[index]))
        finite_difference_gradient.append((upper - lower) / (2.0 * steps[index]))
    finite_difference_gradient = jnp.stack(finite_difference_gradient)
    finite_difference_gradient.block_until_ready()

    assert reverse_gradient.shape == PARAMS.shape
    assert jnp.all(jnp.isfinite(reverse_gradient))
    assert jnp.any(reverse_gradient != 0.0)
    np.testing.assert_allclose(
        np.asarray(reverse_gradient),
        np.asarray(finite_difference_gradient),
        rtol=5.0e-3,
        atol=2.0e-5,
    )
