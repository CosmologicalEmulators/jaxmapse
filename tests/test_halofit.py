import jax
import jax.numpy as jnp
import pytest

from jaxmapse import (
    PkEmulator,
    halofit_background,
    halofit_cosmology,
    halofit_pmm,
    halofit_pmm_from_params,
)


def _synthetic_case(nk=33, nz=100):
    params = jnp.array([3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0])
    cosmology = halofit_cosmology(params)
    k = jnp.exp(jnp.linspace(jnp.log(1.0e-4), jnp.log(10.0), nk))
    z = jnp.linspace(0.0, 2.0, nz)
    pk_lin_zk = (
        1.0e4
        * (k[None, :] / 0.1) ** 0.96
        * jnp.exp(-k[None, :] / 5.0)
        / (1.0 + z[:, None]) ** 2
    )
    ez2 = cosmology.omega_m0 * (1.0 + z) ** 3 + cosmology.omega_lambda0
    omega_m_z = cosmology.omega_m0 * (1.0 + z) ** 3 / ez2
    omega_v_z = cosmology.omega_lambda0 / ez2
    return params, cosmology, z, k, pk_lin_zk, omega_m_z, omega_v_z


class AnalyticLinearPmm:
    def __init__(self, k_grid):
        self.k_grid = k_grid

    def get_Pk(self, input_params, z, D):
        k = self.k_grid
        z_arr = jnp.asarray(z)
        d_arr = jnp.asarray(D)
        base = 1.0e4 * (k / 0.1) ** 0.96 * jnp.exp(-k / 5.0)
        if z_arr.ndim == 0:
            return base * d_arr**2
        return base[None, :] * d_arr[:, None] ** 2


def test_halofit_pmm_jit_vectorized_z100_shape_and_finiteness():
    _, cosmology, z, k, pk_lin_zk, omega_m_z, omega_v_z = _synthetic_case()

    run = jax.jit(halofit_pmm)
    pk_nl_zk = run(cosmology, z, k, pk_lin_zk, omega_m_z, omega_v_z)

    assert pk_nl_zk.shape == (len(z), len(k))
    assert jnp.all(jnp.isfinite(pk_nl_zk))
    assert jnp.all(pk_nl_zk > 0.0)


def test_halofit_background_is_jittable_and_finite():
    _, cosmology, z, *_ = _synthetic_case()

    omega_m_z, omega_v_z = jax.jit(halofit_background)(cosmology, z)

    assert omega_m_z.shape == z.shape
    assert omega_v_z.shape == z.shape
    assert jnp.all(jnp.isfinite(omega_m_z))
    assert jnp.all(jnp.isfinite(omega_v_z))
    assert jnp.all(omega_m_z > 0.0)
    assert jnp.all(omega_v_z > 0.0)


def test_halofit_pmm_vectorized_matches_per_redshift_vmap():
    _, cosmology, z, k, pk_lin_zk, omega_m_z, omega_v_z = _synthetic_case(nk=33, nz=11)

    pk_batch = halofit_pmm(cosmology, z, k, pk_lin_zk, omega_m_z, omega_v_z)

    def one_redshift(z_i, pk_i, omega_m_i, omega_v_i):
        return halofit_pmm(cosmology, z_i, k, pk_i, omega_m_i, omega_v_i)

    pk_vmap = jax.vmap(one_redshift)(z, pk_lin_zk, omega_m_z, omega_v_z)

    assert jnp.allclose(pk_batch, pk_vmap, rtol=1.0e-10, atol=1.0e-10)


def test_halofit_pmm_from_params_matches_cosmology_call():
    params, cosmology, z, k, pk_lin_zk, omega_m_z, omega_v_z = _synthetic_case(
        nk=41, nz=5
    )

    pk_from_cosmology = halofit_pmm(cosmology, z, k, pk_lin_zk, omega_m_z, omega_v_z)
    pk_from_params = halofit_pmm_from_params(
        params, z, k, pk_lin_zk, omega_m_z, omega_v_z
    )

    assert jnp.allclose(pk_from_params, pk_from_cosmology, rtol=1.0e-12, atol=1.0e-12)


def test_halofit_pmm_is_differentiable_wrt_linear_amplitude():
    _, cosmology, z, k, pk_lin_zk, omega_m_z, omega_v_z = _synthetic_case(nk=33, nz=7)

    def loss(log_amp):
        pk_nl = halofit_pmm(
            cosmology, z, k, pk_lin_zk * jnp.exp(log_amp), omega_m_z, omega_v_z
        )
        return jnp.mean(jnp.log(pk_nl))

    grad = jax.grad(loss)(jnp.array(0.0))

    assert jnp.isfinite(grad)
    assert grad != 0.0


def test_halofit_pmm_matches_mapse_jl_regression_values():
    _, cosmology, z, k, pk_lin_zk, omega_m_z, omega_v_z = _synthetic_case(nk=33, nz=17)

    pk_nl = halofit_pmm(cosmology, z, k, pk_lin_zk, omega_m_z, omega_v_z)
    selected = jnp.array(
        [
            pk_nl[0, 0],
            pk_nl[0, 10],
            pk_nl[4, 20],
            pk_nl[8, 32],
            pk_nl[16, 5],
            pk_nl[16, 32],
        ]
    )
    # Generated from Mapse.jl Halofit on the same synthetic input. jaxmapse uses
    # the transposed (z, k) convention relative to Mapse.jl's (k, z).
    reference = jnp.array(
        [
            13.180006910521557,
            413.8807129429551,
            2360.434709393714,
            4.6169480499214245e-09,
            8.230805014366817,
            2.120606766218583e-06,
        ]
    )

    assert jnp.allclose(selected, reference, rtol=1.0e-12, atol=1.0e-12)


def test_pk_emulator_get_halofit_pmm_low_effort_api_matches_manual_call():
    params, cosmology, z, k, _, _, _ = _synthetic_case(nk=33, nz=9)
    linear = AnalyticLinearPmm(k)
    emu = PkEmulator(linear_pmm=linear, linear_pkcb=linear, boost=None)

    k_api, pk_api = emu.get_halofit_pmm(params, z)

    # Manual equivalent using the same public low-level pieces.
    # The fake linear emulator uses D^2 scaling, so choose D consistently with
    # the API's native growth/background path by taking it from the API inputs.
    from jaxace.background import w0waCDMCosmology

    growth_cosmology = w0waCDMCosmology(
        ln10As=params[0],
        ns=params[1],
        h=params[2] / 100.0,
        omega_b=params[3],
        omega_c=params[4],
        m_nu=params[5],
        w0=params[6],
        wa=params[7],
    )
    d_z = growth_cosmology.D_z(z)
    pk_lin = linear.get_Pk(params, z, d_z)
    omega_m_z, omega_v_z = halofit_background(cosmology, z)
    pk_manual = halofit_pmm(cosmology, z, k, pk_lin, omega_m_z, omega_v_z)

    assert jnp.allclose(k_api, k)
    assert pk_api.shape == (len(z), len(k))
    assert jnp.allclose(pk_api, pk_manual, rtol=1.0e-12, atol=1.0e-12)


def test_halofit_pmm_rejects_inconsistent_scalar_vector_shapes():
    _, cosmology, z, k, pk_lin_zk, omega_m_z, omega_v_z = _synthetic_case(nk=17, nz=3)

    with pytest.raises(
        ValueError, match="two-dimensional pk_lin_mm_z requires vector z"
    ):
        halofit_pmm(cosmology, 0.0, k, pk_lin_zk, omega_m_z[0], omega_v_z[0])

    with pytest.raises(ValueError, match="one-dimensional pk_lin_mm_z is only valid"):
        halofit_pmm(cosmology, z, k, pk_lin_zk[0], omega_m_z, omega_v_z)

    with pytest.raises(ValueError, match="same scalar/vector shape as z"):
        halofit_pmm(cosmology, z, k, pk_lin_zk, omega_m_z[0], omega_v_z[0])


def test_halofit_cosmology_rejects_non_flat_hmcode_parameter_vectors():
    params = jnp.array(
        [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0, 0.01, 7.8]
    )

    with pytest.raises(ValueError, match="Curved/HMCODE parameter vectors"):
        halofit_cosmology(params)


def test_halofit_background_z0_convention_is_documented_close_but_not_identical():
    params, cosmology, *_ = _synthetic_case(nk=17, nz=3)
    omega_m0, omega_v0 = halofit_background(cosmology, jnp.array(0.0))

    assert jnp.isfinite(omega_m0)
    assert jnp.isfinite(omega_v0)
    assert abs(float(omega_v0 - cosmology.omega_lambda0)) < 1.0e-4


def test_halofit_pmm_matches_class_reference_table():
    import numpy as np

    data = np.loadtxt("tests/data/halofit_class_reference.txt")
    z_values = np.unique(data[:, 0])
    k_values = data[data[:, 0] == z_values[0], 1]
    pk_lin_zk = np.stack([data[data[:, 0] == z, 2] for z in z_values])
    omega_m_z = np.array([data[data[:, 0] == z, 3][0] for z in z_values])
    omega_v_z = np.array([data[data[:, 0] == z, 4][0] for z in z_values])
    pk_nl_ref = np.stack([data[data[:, 0] == z, 5] for z in z_values])

    params = jnp.array([3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0])
    pk_nl = halofit_pmm(
        halofit_cosmology(params),
        jnp.asarray(z_values),
        jnp.asarray(k_values),
        jnp.asarray(pk_lin_zk),
        jnp.asarray(omega_m_z),
        jnp.asarray(omega_v_z),
    )

    assert jnp.allclose(pk_nl, jnp.asarray(pk_nl_ref), rtol=6.0e-3, atol=0.0)
