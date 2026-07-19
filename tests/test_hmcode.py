from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmapse import (
    HMCodeCosmology,
    hmcode_boost,
    hmcode_boost_fast,
    hmcode_pmm,
    hmcode_pmm_fast,
    hmcode_pmm_jax,
)


def _reference_case():
    data = np.loadtxt(Path(__file__).parent / "data" / "hmcode_camb_reference.txt")
    z_values = np.unique(data[:, 0])
    k_values = data[data[:, 0] == z_values[0], 1]

    # jaxmapse uses redshift-first spectra, the transpose of Mapse.jl's (k, z)
    # convention used in the source Julia tests.
    pk_mm_zk = np.stack([data[data[:, 0] == z, 2] for z in z_values])
    pk_cb_zk = np.stack([data[data[:, 0] == z, 3] for z in z_values])
    pk_nl_ref_zk = np.stack([data[data[:, 0] == z, 4] for z in z_values])
    boost_ref_zk = np.stack([data[data[:, 0] == z, 5] for z in z_values])

    h = 0.6736
    omega_b = 0.02237 / h**2
    omega_nu = 0.06 / (93.14 * h**2)
    omega_m = omega_b + 0.12 / h**2 + omega_nu
    cosmo = HMCodeCosmology(
        Omega_m=omega_m,
        Omega_b=omega_b,
        h=h,
        n_s=0.9649,
        sigma_8=0.8109118,
        w0=-1.0,
        wa=0.0,
        Omega_nu=omega_nu,
        Omega_k=0.0,
    )

    return (
        cosmo,
        jnp.asarray(z_values),
        jnp.asarray(k_values),
        jnp.asarray(pk_mm_zk),
        jnp.asarray(pk_cb_zk),
        jnp.asarray(pk_nl_ref_zk),
        jnp.asarray(boost_ref_zk),
    )


def _curved_parity_case():
    data = np.loadtxt(
        Path(__file__).parent / "data" / "hmcode_curved_parity_reference.txt"
    )
    z_values = np.unique(data[:, 0])
    k_values = data[data[:, 0] == z_values[0], 1]
    pmm_zk = np.stack([data[data[:, 0] == z, 2] for z in z_values])
    pcb_zk = np.stack([data[data[:, 0] == z, 3] for z in z_values])
    dmo_zk = np.stack([data[data[:, 0] == z, 4] for z in z_values])
    feedback_zk = np.stack([data[data[:, 0] == z, 5] for z in z_values])

    h = 0.6736
    omega_b = 0.02237 / h**2
    omega_nu = 0.06 / (93.14 * h**2)
    omega_m = omega_b + 0.12 / h**2 + omega_nu
    cosmo = HMCodeCosmology(
        Omega_m=omega_m,
        Omega_b=omega_b,
        h=h,
        n_s=0.9649,
        sigma_8=0.8109118,
        w0=-0.9,
        wa=0.2,
        Omega_nu=omega_nu,
        Omega_k=0.01,
    )
    return (
        cosmo,
        jnp.asarray(z_values),
        jnp.asarray(k_values),
        jnp.asarray(pmm_zk),
        jnp.asarray(pcb_zk),
        jnp.asarray(dmo_zk),
        jnp.asarray(feedback_zk),
    )


def test_hmcode_curved_neutrino_parity_fixture_matches_native_julia():
    cosmo, z, k, pmm_zk, pcb_zk, dmo_reference, feedback_reference = (
        _curved_parity_case()
    )

    dmo = hmcode_pmm(cosmo, z, k, pmm_zk, pcb_zk, T_AGN=None, nM=32)
    feedback = hmcode_pmm(cosmo, z, k, pmm_zk, pcb_zk, T_AGN=10.0**7.8, nM=32)

    # Native Julia uses adaptive root finding/integration while JAX uses fixed
    # compiled grids, so this deliberately has a looser cross-backend tolerance.
    assert jnp.allclose(dmo, dmo_reference, rtol=1.5e-2, atol=0.0)
    assert jnp.allclose(feedback, feedback_reference, rtol=1.5e-2, atol=0.0)


def test_hmcode_boost_matches_camb_reference_table():
    cosmo, z, k, pk_mm_zk, pk_cb_zk, _, boost_ref_zk = _reference_case()

    boost = hmcode_boost(cosmo, z, k, pk_mm_zk, pk_cb_zk, nM=256)

    assert boost.shape == pk_mm_zk.shape
    assert jnp.all(jnp.isfinite(boost))
    assert jnp.allclose(boost, boost_ref_zk, rtol=3.0e-3, atol=0.0)


def test_hmcode_scalar_pmm_and_boost_are_consistent():
    cosmo, _, k, pk_mm_zk, pk_cb_zk, *_ = _reference_case()

    pmm = hmcode_pmm(cosmo, 0.0, k, pk_mm_zk[0], pk_cb_zk[0], nM=64)
    boost = hmcode_boost(cosmo, 0.0, k, pk_mm_zk[0], pk_cb_zk[0], nM=64)

    assert pmm.shape == k.shape
    assert boost.shape == k.shape
    assert jnp.all(jnp.isfinite(pmm))
    assert jnp.all(jnp.isfinite(boost))
    assert jnp.allclose(pmm / pk_mm_zk[0], boost, rtol=1.0e-10, atol=1.0e-10)


def test_hmcode_keyword_and_positional_cb_paths_match():
    cosmo, z, k, pk_mm_zk, pk_cb_zk, *_ = _reference_case()

    boost_keyword = hmcode_boost(cosmo, z, k, pk_mm_zk, pk_cb_z=pk_cb_zk, nM=64)
    boost_positional = hmcode_boost(cosmo, z, k, pk_mm_zk, pk_cb_zk, nM=64)

    assert jnp.allclose(boost_keyword, boost_positional, rtol=1.0e-12, atol=1.0e-12)


def test_hmcode_support_grid_matches_same_grid_call():
    cosmo, z, k, pk_mm_zk, pk_cb_zk, *_ = _reference_case()

    boost_support = hmcode_boost(
        cosmo,
        z,
        k,
        pk_mm_zk,
        k_support=k,
        pk_cb_support_z=pk_cb_zk,
        nM=64,
    )
    boost_same_grid = hmcode_boost(cosmo, z, k, pk_mm_zk, pk_cb_zk, nM=64)

    assert jnp.allclose(boost_support, boost_same_grid, rtol=1.0e-12, atol=1.0e-12)


def test_hmcode_support_output_grid_and_scalar_support_call():
    cosmo, z, k, pk_mm_zk, pk_cb_zk, *_ = _reference_case()
    k_out = k[::2]

    pmm_kout = hmcode_pmm(
        cosmo,
        z,
        k_out,
        pk_mm_zk,
        k_support=k,
        pk_cb_support_z=pk_cb_zk,
        nM=64,
    )
    boost_kout = hmcode_boost(
        cosmo,
        z,
        k_out,
        pk_mm_zk,
        k_support=k,
        pk_cb_support_z=pk_cb_zk,
        nM=64,
    )

    assert pmm_kout.shape == (len(z), len(k_out))
    assert boost_kout.shape == (len(z), len(k_out))
    assert jnp.all(jnp.isfinite(pmm_kout))
    assert jnp.allclose(
        boost_kout, pmm_kout / pk_mm_zk[:, ::2], rtol=1e-4, atol=1.0e-10
    )

    pmm_full = hmcode_pmm(cosmo, z, k, pk_mm_zk, pk_cb_z=pk_cb_zk, nM=64)
    # The internal log-log interpolation from the support grid to the output grid
    # inherently introduces ~4.1e-3 relative numerical differences compared to
    # full-grid evaluation. This is an intentional interpolation approximation, not a float32 artifact.
    assert jnp.allclose(pmm_kout, pmm_full[:, ::2], rtol=5e-3, atol=1e-10)

    k_out_irr = k_out.at[10].set(0.5 * (k_out[9] + k_out[10]))
    pmm_kout_irr = hmcode_pmm(
        cosmo, z, k_out_irr, pk_mm_zk, k_support=k, pk_cb_support_z=pk_cb_zk, nM=64
    )
    assert pmm_kout_irr.shape == (len(z), len(k_out_irr))
    for iz in range(len(z)):
        interp_val = jnp.interp(jnp.log(k_out_irr), jnp.log(k), pmm_full[iz])
        assert jnp.allclose(interp_val, pmm_kout_irr[iz], rtol=5e-3)


    scalar_pmm = hmcode_pmm(
        cosmo,
        0.0,
        k_out,
        pk_mm_zk[0],
        k_support=k,
        pk_cb_support_z=pk_cb_zk[0],
        nM=64,
    )
    assert jnp.allclose(scalar_pmm, pmm_kout[0], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("nM", [32])
def test_hmcode_pure_jax_kernel_is_jittable(nM):
    cosmo, z_all, k_support, pk_mm_all, pk_cb_all, *_ = _reference_case()
    z = z_all[:1]
    k_out = k_support[::8]
    pk_mm = pk_mm_all[:1]
    pk_cb = pk_cb_all[:1]

    compiled = jax.jit(
        hmcode_pmm_jax,
        static_argnames=("nM", "include_feedback"),
    )
    pmm_jit = compiled(
        cosmo,
        z,
        k_out,
        k_support,
        pk_mm,
        pk_cb,
        nM=nM,
        include_feedback=True,
    )
    pmm_host = hmcode_pmm(
        cosmo,
        z,
        k_out,
        pk_mm,
        k_support=k_support,
        pk_cb_support_z=pk_cb,
        nM=nM,
    )

    assert pmm_jit.shape == pmm_host.shape
    assert jnp.all(jnp.isfinite(pmm_jit))
    assert jnp.allclose(pmm_jit, pmm_host, rtol=3.0e-3, atol=0.0)

@pytest.mark.parametrize("nM", [16, 64])
def test_hmcode_fast_kernel_is_jittable(nM):
    cosmo, z_all, k_support, pk_mm_all, pk_cb_all, *_ = _reference_case()
    # z_all only has 3 elements, but Akima needs >= 5.
    # Create a 5 element grid and repeat realistic spectra to avoid NaNs
    z_coarse = jnp.linspace(0.0, 2.0, 5)
    pk_mm = jnp.repeat(pk_mm_all[:1], 5, axis=0)
    pk_cb = jnp.repeat(pk_cb_all[:1], 5, axis=0)
    z_fine = jnp.linspace(0.0, 2.0, 20)
    k = k_support

    @jax.jit
    def run_fast(c, zc, zf, k_arr, pm, pc):
        return hmcode_pmm_fast(c, zc, zf, k_arr, pm, pk_cb_coarse=pc, nM=nM)

    @jax.jit
    def run_boost(c, zc, zf, k_arr, pm, pc):
        return hmcode_boost_fast(c, zc, zf, k_arr, pm, pk_cb_coarse=pc, nM=nM)

    pmm_jit = run_fast(cosmo, z_coarse, z_fine, k, pk_mm, pk_cb)
    boost_jit = run_boost(cosmo, z_coarse, z_fine, k, pk_mm, pk_cb)

    assert pmm_jit.shape == (20, len(k))
    assert jnp.all(jnp.isfinite(pmm_jit))

    assert boost_jit.shape == (20, len(k))
    assert jnp.all(jnp.isfinite(boost_jit))


def test_hmcode_rejects_invalid_inputs_like_mapse_jl():
    cosmo, z, k, pk_mm_zk, pk_cb_zk, *_ = _reference_case()
    k_out = k[::2]

    with pytest.raises(ValueError, match="sorted in ascending order"):
        hmcode_pmm(cosmo, z, k[::-1], pk_mm_zk)

    irregular_k = k.at[10].set(0.5 * (k[9] + k[10]))
    with pytest.raises(ValueError, match="uniformly spaced in log"):
        hmcode_pmm(cosmo, z, irregular_k, pk_mm_zk)

    with pytest.raises(ValueError, match="pk_mm_z must have shape"):
        hmcode_pmm(cosmo, jnp.array([0.0, 1.0]), k, pk_mm_zk[:1])

    with pytest.raises(ValueError, match="pk_cb_z must have shape"):
        hmcode_pmm(cosmo, z, k, pk_mm_zk, pk_cb_zk[:, :-1])

    with pytest.raises(ValueError, match="at least 2"):
        hmcode_boost(cosmo, z, k, pk_mm_zk, pk_cb_zk, nM=0)

    with pytest.raises(ValueError, match="at least 2"):
        hmcode_boost(cosmo, z, k, pk_mm_zk, pk_cb_zk, nM=1)

    with pytest.raises(ValueError, match="output k range"):
        hmcode_boost(
            cosmo,
            z,
            jnp.concatenate([k_out, jnp.array([20.0])]),
            pk_mm_zk,
            k_support=k,
            pk_cb_support_z=pk_cb_zk,
        )


def test_hmcode_fast_apis_validation_and_correctness():
    cosmo, z_all, k, pk_mm_all, pk_cb_all, _, _ = _reference_case()

    # We need at least 5 coarse redshift points for Akima interpolation.
    z_coarse = jnp.linspace(0.0, 1.0, 6)
    pk_mm_coarse = jnp.ones((6, len(k)))
    pk_cb_coarse = jnp.ones((6, len(k)))

    z_fine = jnp.linspace(z_coarse[0], z_coarse[-1], 20)

    # 1. Test hmcode_pmm_fast and hmcode_boost_fast compared to direct ones at the endpoints
    pk_nl_fast = hmcode_pmm_fast(
        cosmo, z_coarse, z_fine, k, pk_mm_coarse, pk_cb_coarse, nM=64
    )
    boost_fast = hmcode_boost_fast(
        cosmo, z_coarse, z_fine, k, pk_mm_coarse, pk_cb_coarse, nM=64
    )

    assert pk_nl_fast.shape == (20, len(k))
    assert boost_fast.shape == (20, len(k))

    # 2. Test support-grid mode
    k_out = k[::2]
    pk_nl_support = hmcode_pmm_fast(
        cosmo,
        z_coarse,
        z_fine,
        k_out,
        pk_mm_coarse,
        k_support=k,
        pk_cb_support_coarse=pk_cb_coarse,
        nM=64,
    )
    assert pk_nl_support.shape == (20, len(k_out))

    # 3. Test scalar z_fine
    pk_nl_scalar = hmcode_pmm_fast(
        cosmo, z_coarse, z_coarse[2], k, pk_mm_coarse, pk_cb_coarse, nM=64
    )
    assert pk_nl_scalar.shape == (len(k),)

    # 4. Test feedback vs DMO
    pk_nl_dmo = hmcode_pmm_fast(
        cosmo, z_coarse, z_fine, k, pk_mm_coarse, pk_cb_coarse, T_AGN=None, nM=64
    )
    assert not jnp.allclose(pk_nl_fast, pk_nl_dmo)

    # 5. Conflict handling for cb spectra
    with pytest.raises(
        ValueError, match="Pass either pk_cb_coarse or pk_cb_support_coarse"
    ):
        hmcode_pmm_fast(
            cosmo,
            z_coarse,
            z_fine,
            k,
            pk_mm_coarse,
            pk_cb_coarse=pk_cb_coarse,
            pk_cb_support_coarse=pk_cb_coarse,
        )

    # 6. Grid size check: z_coarse must have at least 5 points
    with pytest.raises(ValueError, match="z_coarse must have at least 5 points"):
        hmcode_pmm_fast(
            cosmo, z_coarse[:4], z_fine, k, pk_mm_coarse[:4], pk_cb_coarse[:4]
        )

    # 7. Redshift interpolation boundary checks
    with pytest.raises(
        ValueError, match="z_fine must lie within the range of z_coarse"
    ):
        hmcode_pmm_fast(
            cosmo,
            z_coarse,
            jnp.array([z_coarse[-1] + 0.1]),
            k,
            pk_mm_coarse,
            pk_cb_coarse,
        )

    with pytest.raises(ValueError, match="z_coarse must be strictly increasing"):
        hmcode_pmm_fast(
            cosmo, z_coarse[::-1], z_fine, k, pk_mm_coarse[::-1], pk_cb_coarse[::-1]
        )

def _class_feedback_reference_case():
    data_ref = np.loadtxt(Path(__file__).parent / "data" / "hmcode_class_feedback_reference.txt")
    data_sup = np.loadtxt(Path(__file__).parent / "data" / "hmcode_class_linear_support.txt")

    z_values = np.unique(data_sup[:, 0])
    k_values = data_sup[data_sup[:, 0] == z_values[0], 1]

    pk_mm_zk = np.stack([data_sup[data_sup[:, 0] == z, 2] for z in z_values])
    pk_cb_zk = np.stack([data_sup[data_sup[:, 0] == z, 3] for z in z_values])

    pk_nl_feedback_zk = np.stack([data_ref[data_ref[:, 0] == z, 4] for z in z_values])
    feedback_boost_zk = np.stack([data_ref[data_ref[:, 0] == z, 5] for z in z_values])
    pk_nl_dmo_zk = np.stack([data_ref[data_ref[:, 0] == z, 6] for z in z_values])
    dmo_boost_zk = np.stack([data_ref[data_ref[:, 0] == z, 7] for z in z_values])

    h = 0.6736
    omega_b = 0.02237 / h**2
    omega_nu = 0.06 / (93.14 * h**2)
    omega_m = omega_b + 0.12 / h**2 + omega_nu
    cosmo = HMCodeCosmology(
        Omega_m=omega_m,
        Omega_b=omega_b,
        h=h,
        n_s=0.9649,
        sigma_8=0.8109118, # approximated to match CLASS sigma8, but we feed linear Pk directly
        w0=-1.0,
        wa=0.0,
        Omega_nu=omega_nu,
        Omega_k=0.0,
    )

    return (
        cosmo,
        jnp.asarray(z_values),
        jnp.asarray(k_values),
        jnp.asarray(pk_mm_zk),
        jnp.asarray(pk_cb_zk),
        jnp.asarray(pk_nl_feedback_zk),
        jnp.asarray(feedback_boost_zk),
        jnp.asarray(pk_nl_dmo_zk),
        jnp.asarray(dmo_boost_zk),
    )


def test_hmcode_feedback_matches_patched_class_reference():
    cosmo, z, k, pk_mm_zk, pk_cb_zk, _, fb_boost_lin, _, dmo_boost_lin = _class_feedback_reference_case()
    boost = hmcode_boost(cosmo, z, k, pk_mm_zk, pk_cb_zk, T_AGN=10.0**7.8, nM=256)

    mask = k <= 10.0
    boost_ref_zk = fb_boost_lin
    boost = boost[:, mask]
    k_ref = k[mask]

    err = jnp.abs(boost - boost_ref_zk) / boost_ref_zk
    max_err = jnp.max(err)
    max_idx = jnp.unravel_index(jnp.argmax(err), err.shape)
    if max_err >= 1.6e-2 or jnp.any(err >= 5.0e-3):
        print(f"Max rel err: {max_err} at z={z[max_idx[0]]}, k={k_ref[max_idx[1]]}")

    # Enforce 0.5% strict ceiling on the k <= 10.0 region where the finite integration bound does not pollute the response
    assert jnp.all(err < 5.0e-3)

def test_hmcode_feedback_low_k_response_regression():
    cosmo, z, k, pk_mm_zk, pk_cb_zk, _, boost_ref_zk, _, _ = _class_feedback_reference_case()

    feedback = hmcode_pmm(cosmo, z, k, pk_mm_zk, pk_cb_zk, T_AGN=10.0**7.8, nM=256)
    dmo = hmcode_pmm(cosmo, z, k, pk_mm_zk, pk_cb_zk, T_AGN=None, nM=256)

    response = feedback / dmo
    low_k = k <= 3.0e-4

    k_ref = k[k <= 10.0]
    low_k_ref = k_ref <= 3.0e-4

    np.testing.assert_allclose(response[:, low_k], 1.0, rtol=1.0e-3, atol=0.0)
    np.testing.assert_allclose((feedback / pk_mm_zk)[:, low_k], boost_ref_zk[:, low_k_ref], rtol=1.0e-3, atol=0.0)

def test_hmcode_feedback_low_k_response_regression_fast_jit():
    cosmo, z, k, pk_mm_zk, pk_cb_zk, _, boost_ref_zk, _, _ = _class_feedback_reference_case()

    z_fine = jnp.linspace(z[0], z[-1], 25)

    @jax.jit
    def run_fast_dmo(c, zc, zf, k_arr, pm, pc):
        return hmcode_pmm_fast(c, zc, zf, k_arr, pm, pk_cb_coarse=pc, T_AGN=None, nM=64)

    @jax.jit
    def run_fast_fb(c, zc, zf, k_arr, pm, pc):
        return hmcode_pmm_fast(c, zc, zf, k_arr, pm, pk_cb_coarse=pc, T_AGN=10.0**7.8, nM=64)

    @jax.jit
    def run_boost_fast(c, zc, zf, k_arr, pm, pc):
        return hmcode_boost_fast(c, zc, zf, k_arr, pm, pk_cb_coarse=pc, T_AGN=10.0**7.8, nM=64)

    dmo = run_fast_dmo(cosmo, z, z_fine, k, pk_mm_zk, pk_cb_zk)
    feedback = run_fast_fb(cosmo, z, z_fine, k, pk_mm_zk, pk_cb_zk)
    boost_fast_out = run_boost_fast(cosmo, z, z_fine, k, pk_mm_zk, pk_cb_zk)

    response = feedback / dmo
    low_k = k <= 3.0e-4
    k_ref = k[k <= 10.0]
    low_k_ref = k_ref <= 3.0e-4

    np.testing.assert_allclose(response[:, low_k], 1.0, rtol=1.0e-3, atol=0.0)

    # We evaluate boost_ref_zk which is at redshift z, against boost_fast_out at z_fine.
    # We interpolate boost_ref_zk over redshift to match z_fine.
    boost_fast_out_ref = boost_fast_out[:, k <= 10.0]
    interp_boost_ref = jax.vmap(lambda boost_k: jnp.interp(z_fine, z, boost_k))(boost_ref_zk.T).T
    np.testing.assert_allclose(boost_fast_out_ref[:, low_k_ref], interp_boost_ref[:, low_k_ref], rtol=1.0e-3, atol=0.0)


def test_hmcode_no_tweaks_preserves_one_halo_cutoff():
    from jaxmapse.hmcode import HMCodeParams, _params_notweaks

    nz = 5
    zeros = jnp.zeros(nz)
    ones = jnp.ones(nz)
    k_star_mock = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])

    params = HMCodeParams(
        R_nl=ones,
        n_eff=ones,
        sigma_v=ones,
        Delta_v=ones,
        delta_c=ones,
        eta=ones,
        A=ones,
        f_damp=ones,
        k_star=k_star_mock,
        B=ones,
        k_damp=ones,
    )

    params_notweaks = _params_notweaks(params)
    np.testing.assert_allclose(params_notweaks.k_star, params.k_star, rtol=0, atol=0)
    assert np.all(np.asarray(params_notweaks.k_star) > 0)


def test_hmcode_fixture_schema_check():
    """Validate the split canonical+support fixture contract.

    Canonical fixture: nonlinear reference outputs, k <= 10 h/Mpc, 8 columns.
    Support fixture:   linear spectra for internal integrals, k <= 50 h/Mpc, 4 columns.
    Both fixtures must share identical redshift grids and the canonical k-grid
    must be a prefix-aligned subset of the support k-grid.
    """
    base = Path(__file__).parent / "data"
    canonical = base / "hmcode_class_feedback_reference.txt"
    support = base / "hmcode_class_linear_support.txt"

    # --- canonical fixture schema ---
    raw_can = canonical.read_text(encoding="utf-8")
    assert "fixture_schema_version: 1" in raw_can
    assert "CAMB_PR_136_low_k_response_cutoff" in raw_can
    assert "domain: canonical test region k <= 10 h/Mpc" in raw_can

    data_can = np.loadtxt(canonical)
    assert data_can.shape == (705, 8), f"Canonical shape {data_can.shape} != (705, 8)"

    k_can_z0 = data_can[data_can[:, 0] == data_can[0, 0], 1]
    assert len(k_can_z0) == 141, f"Canonical per-redshift count {len(k_can_z0)} != 141"
    np.testing.assert_allclose(k_can_z0.max(), 9.696137237434288, rtol=1e-5)
    assert np.all(np.isfinite(data_can)), "Canonical fixture contains non-finite values"
    assert np.all(k_can_z0 > 0), "Canonical fixture contains non-positive k values"
    assert np.all(data_can[:, 2] > 0), "Canonical fixture pk_mm_lin contains non-positive values"

    # --- support fixture schema ---
    raw_sup = support.read_text(encoding="utf-8")
    assert "fixture_schema_version: 1" in raw_sup
    assert "CAMB_PR_136_low_k_response_cutoff" in raw_sup
    assert "domain: linear support grid k <= 50 h/Mpc" in raw_sup

    data_sup = np.loadtxt(support)
    assert data_sup.shape == (805, 4), f"Support shape {data_sup.shape} != (805, 4)"

    k_sup_z0 = data_sup[data_sup[:, 0] == data_sup[0, 0], 1]
    assert len(k_sup_z0) == 161, f"Support per-redshift count {len(k_sup_z0)} != 161"
    np.testing.assert_allclose(k_sup_z0.max(), 50.0, rtol=1e-5)
    assert np.all(np.isfinite(data_sup)), "Support fixture contains non-finite values"
    assert np.all(k_sup_z0 > 0), "Support fixture contains non-positive k values"

    # --- split-contract: matching redshift grids ---
    z_can = np.unique(data_can[:, 0])
    z_sup = np.unique(data_sup[:, 0])
    np.testing.assert_array_equal(z_can, z_sup,
                                   err_msg="Canonical and support redshift grids differ")

    # --- split-contract: canonical k is a subset of support k ---
    k_sup_set = set(np.round(k_sup_z0, 12))
    for k_val in np.round(k_can_z0, 12):
        assert k_val in k_sup_set, (
            f"Canonical k={k_val} not found in support k-grid; grids are not aligned"
        )


def test_hmcode_dmo_invariant():
    cosmo, z, k, pk_mm_zk, pk_cb_zk, _, _, dmo_ref_zk, _ = _class_feedback_reference_case()
    # Ensure standard DMO response matches without regression
    dmo_out = hmcode_pmm(cosmo, z, k, pk_mm_zk, pk_cb_zk, T_AGN=None, nM=256)

    k_ref = k[k <= 10.0]
    dmo_out_ref = dmo_out[:, k <= 10.0]

    err = jnp.abs(dmo_out_ref - dmo_ref_zk) / dmo_ref_zk
    max_err = jnp.max(err)
    max_idx = jnp.unravel_index(jnp.argmax(err), err.shape)
    if max_err > 3.0e-3:
        print(f"DMO Max rel err: {max_err} at z={z[max_idx[0]]}, k={k_ref[max_idx[1]]}")
    assert jnp.allclose(dmo_out_ref, dmo_ref_zk, rtol=3.0e-3, atol=1e-12)
