from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmapse import (
    DEFAULT_EMULATOR_ARTIFACT,
    artifact_path,
    build_smart_coarse_grid,
    hmcode_pmm_baryonic_smart,
    hmcode_pmm_dmo_smart,
    hmcode_pmm_from_emulator,
    hmcode_pmm_from_emulator_fast,
    load_emulator,
    piecewise_akima_interpolation,
    predict_baryonic_discontinuity,
)


PARAMS = jnp.array([3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0])
Z_FINE = jnp.linspace(0.0, 3.5, 150)
N_COARSE_VALUES = (24, 32, 40)
CAMB_MAX_RELATIVE_ERROR = 0.0051


def test_baryonic_grid_is_fixed_shape_dynamic_and_differentiable():
    n_coarse = 24
    n_left = 15

    grid_fn = jax.jit(
        lambda feature: build_smart_coarse_grid(
            0.0, 3.5, n_coarse, z_feature=feature, N_left=n_left
        )
    )
    grid1 = grid_fn(jnp.array(2.0))
    grid2 = grid_fn(jnp.array(2.2))
    grid3 = grid_fn(jnp.array(10.0))

    assert grid1.shape == (n_coarse,)
    assert grid2.shape == grid1.shape == grid3.shape
    assert jnp.all(jnp.diff(grid1) > 0.0)
    assert jnp.all(jnp.diff(grid2) > 0.0)
    assert jnp.all(jnp.diff(grid3) > 0.0)
    assert grid1[n_left - 1] == 2.0
    assert grid2[n_left - 1] == 2.2
    np.testing.assert_allclose(grid3[n_left - 1], 0.95 * 3.5, rtol=0.0, atol=1.0e-14)
    assert not jnp.allclose(grid1, grid2)

    gradient = jax.grad(lambda feature: jnp.sum(grid_fn(feature)))(jnp.array(2.0))
    step = 1.0e-5
    finite_difference = (
        jnp.sum(grid_fn(jnp.array(2.0 + step)))
        - jnp.sum(grid_fn(jnp.array(2.0 - step)))
    ) / (2.0 * step)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1.0e-8, atol=1.0e-9)


def test_piecewise_akima_has_static_slices_and_is_differentiable():
    n_left = 7
    z_feature = jnp.array(2.0)
    z_left = jnp.linspace(0.0, z_feature, n_left)
    z_right = jnp.linspace(z_feature, 3.5, 6)
    z_coarse = jnp.concatenate((z_left, z_right[1:]))
    values = jnp.stack((jnp.sin(z_coarse), jnp.cos(z_coarse)), axis=1)

    interpolate = jax.jit(
        lambda data, feature: piecewise_akima_interpolation(
            data,
            z_coarse,
            z_coarse,
            feature,
            split_index=n_left,
        )
    )
    prediction = interpolate(values, z_feature)

    np.testing.assert_allclose(prediction, values, rtol=1.0e-12, atol=1.0e-12)
    gradient = jax.grad(lambda data: jnp.sum(interpolate(data, z_feature)))(values)
    assert gradient.shape == values.shape
    assert jnp.all(jnp.isfinite(gradient))


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
    if path == "baryonic_smart":
        return hmcode_pmm_baryonic_smart(
            params,
            z_fine=jnp.linspace(0.0, 3.5, 20),
            N_coarse=12,
            T_AGN=10.0**7.8,
            linear_pmm_emu=pmm,
            linear_pcb_emu=pcb,
            nM=16,
        )[1]
    raise AssertionError(f"Unknown pipeline: {path}")


@pytest.mark.parametrize(
    "path", ("direct", "fixed_fast", "dmo_smart", "baryonic_smart")
)
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


@pytest.mark.parametrize(
    "path", ("direct", "fixed_fast", "dmo_smart", "baryonic_smart")
)
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

    steps = 1.0e-6 * jnp.maximum(jnp.abs(PARAMS), 1.0)
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


def test_baryonic_smart_dynamic_temperature_reuses_graph_and_has_correct_gradient(
    linear_emulators,
):
    pmm, pcb = linear_emulators
    trace_count = 0

    def pipeline(params, log_temperature, z_fine):
        nonlocal trace_count
        trace_count += 1
        return hmcode_pmm_baryonic_smart(
            params,
            z_fine=z_fine,
            N_coarse=12,
            T_AGN=10.0**log_temperature,
            linear_pmm_emu=pmm,
            linear_pcb_emu=pcb,
            nM=16,
        )[1]

    z_fine1 = jnp.linspace(0.0, 3.5, 20)
    z_fine2 = 3.5 * jnp.linspace(0.0, 1.0, 20) ** 1.1
    compiled = jax.jit(pipeline).lower(PARAMS, jnp.array(7.8), z_fine1).compile()
    prediction1 = compiled(PARAMS, jnp.array(7.8), z_fine1)
    prediction2 = compiled(PARAMS, jnp.array(8.0), z_fine1)
    prediction3 = compiled(PARAMS, jnp.array(7.8), z_fine2)
    prediction1.block_until_ready()
    prediction2.block_until_ready()
    prediction3.block_until_ready()

    assert trace_count == 1
    assert prediction1.shape == prediction2.shape == prediction3.shape
    assert jnp.all(jnp.isfinite(prediction1))
    assert jnp.all(jnp.isfinite(prediction2))
    assert jnp.all(jnp.isfinite(prediction3))
    assert not jnp.allclose(prediction1, prediction2)
    assert not jnp.allclose(prediction1, prediction3)

    def loss(log_temperature):
        return jnp.mean(jnp.log(pipeline(PARAMS, log_temperature, z_fine1)))

    log_temperature = jnp.array(7.8)
    reverse_gradient = jax.jit(jax.grad(loss))(log_temperature)
    step = 1.0e-5
    finite_difference = (
        jax.jit(loss)(log_temperature + step)
        - jax.jit(loss)(log_temperature - step)
    ) / (2.0 * step)
    reverse_gradient.block_until_ready()
    finite_difference.block_until_ready()

    assert jnp.isfinite(reverse_gradient)
    assert reverse_gradient != 0.0
    np.testing.assert_allclose(
        reverse_gradient, finite_difference, rtol=5.0e-3, atol=2.0e-5
    )


def test_baryonic_smart_reuses_compilation_across_large_feature_shifts(
    linear_emulators,
):
    pmm, pcb = linear_emulators
    z_fine = jnp.linspace(0.0, 3.5, 50)
    n_coarse = 24
    n_left = 15

    low_feature_params = PARAMS.at[3].set(0.0202).at[4].set(0.178)
    high_feature_params = PARAMS.at[3].set(0.0248).at[4].set(0.082)
    cases = (
        (PARAMS, jnp.array(7.8)),
        (low_feature_params, jnp.array(7.6)),
        (high_feature_params, jnp.array(8.2)),
    )
    trace_count = 0

    def evaluate(params, log_temperature):
        temperature = 10.0**log_temperature
        feature = predict_baryonic_discontinuity(params, T_AGN=temperature)
        grid = build_smart_coarse_grid(
            z_fine[0],
            z_fine[-1],
            n_coarse,
            z_feature=feature,
            N_left=n_left,
        )
        prediction = hmcode_pmm_baryonic_smart(
            params,
            z_fine=z_fine,
            N_coarse=n_coarse,
            T_AGN=temperature,
            linear_pmm_emu=pmm,
            linear_pcb_emu=pcb,
            nM=16,
        )[1]
        return feature, grid, prediction

    def traced_evaluate(params, log_temperature):
        nonlocal trace_count
        trace_count += 1
        return evaluate(params, log_temperature)

    compiled = jax.jit(traced_evaluate).lower(*cases[0]).compile()
    compiled_results = [compiled(*case) for case in cases]
    jax.block_until_ready(compiled_results)

    assert trace_count == 1
    features = jnp.stack([result[0] for result in compiled_results])
    assert features[0] - features[1] > 0.4
    assert features[2] - features[0] > 0.4
    assert features[2] - features[1] > 0.8

    for (_, grid, prediction), case in zip(compiled_results, cases):
        assert grid.shape == (n_coarse,)
        assert prediction.shape == (len(z_fine), pmm.k_grid.shape[0])
        assert jnp.all(jnp.diff(grid) > 0.0)
        assert jnp.all(jnp.isfinite(prediction))
        np.testing.assert_allclose(grid[n_left - 1], evaluate(*case)[0], rtol=0.0, atol=1.0e-12)
        reference = evaluate(*case)[2]
        np.testing.assert_allclose(prediction, reference, rtol=1.0e-12, atol=1.0e-12)

    def loss(params, log_temperature):
        return jnp.mean(jnp.log(evaluate(params, log_temperature)[2]))

    compiled_loss = jax.jit(loss)
    compiled_gradient = jax.jit(jax.grad(loss, argnums=(0, 1))).lower(*cases[0]).compile()
    for params, log_temperature in cases[1:]:
        params_gradient, temperature_gradient = compiled_gradient(params, log_temperature)
        jax.block_until_ready((params_gradient, temperature_gradient))
        assert jnp.all(jnp.isfinite(params_gradient))
        assert jnp.isfinite(temperature_gradient)

        finite_differences = []
        reverse_gradients = []
        for index in (3, 4):
            step = 1.0e-6 * jnp.maximum(jnp.abs(params[index]), 1.0)
            finite_differences.append(
                (
                    compiled_loss(params.at[index].add(step), log_temperature)
                    - compiled_loss(params.at[index].add(-step), log_temperature)
                )
                / (2.0 * step)
            )
            reverse_gradients.append(params_gradient[index])

        temperature_step = 1.0e-6
        finite_differences.append(
            (
                compiled_loss(params, log_temperature + temperature_step)
                - compiled_loss(params, log_temperature - temperature_step)
            )
            / (2.0 * temperature_step)
        )
        reverse_gradients.append(temperature_gradient)
        np.testing.assert_allclose(
            np.asarray(reverse_gradients),
            np.asarray(finite_differences),
            rtol=5.0e-4,
            atol=2.0e-5,
        )
